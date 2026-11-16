import gc
import heapq
import pickle

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import LabelEncoder

from warpgbm.cuda import node_kernel
from warpgbm.metrics import (
    accuracy_torch,
    expectile_loss_torch,
    log_loss_torch,
    pseudo_huber_loss_torch,
    rmsle_torch,
    softmax,
)

_ERA_MODE_MAP = {
    "max_agreement": 0,
    "threshold": 1,
    "soft": 2,
    "min_gain": 3,
    "pooled": 4,
}


class WarpGBM(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        objective="regression",
        num_bins=10,
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        min_child_weight=None,
        min_split_gain=0.0,
        threads_per_block=64,
        rows_per_thread=4,
        L2_reg=1e-6,
        device="cuda",
        colsample_bytree=1.0,
        random_state=None,
        warm_start=False,
        L1_reg=0.0,
        expectile_alpha=0.5,
        huber_delta=1.0,
        subsample=1.0,
        colsample_bylevel=1.0,
        colsample_bynode=1.0,
        path_smooth=0.0,
        pack_bits="auto",
        prebinned="auto",
        max_leaves=None,
        era_split_mode="max_agreement",
        era_agreement_threshold=0.75,
        era_agreement_power=2.0,
        era_dro_gamma=0.0,
        era_dro_clip_min=0.2,
        era_dro_clip_max=5.0,
        restore_best_iteration=True,
        verbose=1,
    ):
        self._validate_hyperparams(
            objective=objective,
            num_bins=num_bins,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_weight=min_child_weight,
            min_split_gain=min_split_gain,
            threads_per_block=threads_per_block,
            rows_per_thread=rows_per_thread,
            L2_reg=L2_reg,
            colsample_bytree=colsample_bytree,
            L1_reg=L1_reg,
            expectile_alpha=expectile_alpha,
            huber_delta=huber_delta,
            subsample=subsample,
            colsample_bylevel=colsample_bylevel,
            colsample_bynode=colsample_bynode,
            path_smooth=path_smooth,
            pack_bits=pack_bits,
            prebinned=prebinned,
            max_leaves=max_leaves,
            era_split_mode=era_split_mode,
            era_agreement_threshold=era_agreement_threshold,
            era_agreement_power=era_agreement_power,
            era_dro_gamma=era_dro_gamma,
            era_dro_clip_min=era_dro_clip_min,
            era_dro_clip_max=era_dro_clip_max,
        )

        self.objective = objective
        self.num_bins = num_bins
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.min_child_weight = min_child_weight
        self.min_split_gain = min_split_gain
        self.threads_per_block = threads_per_block
        self.rows_per_thread = rows_per_thread
        self.L2_reg = L2_reg
        self.L1_reg = L1_reg
        self.device = device
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.path_smooth = path_smooth
        self.pack_bits = pack_bits
        self.prebinned = prebinned
        self.subsample = subsample
        self.max_leaves = max_leaves
        self.expectile_alpha = expectile_alpha
        self.huber_delta = huber_delta
        self.era_split_mode = era_split_mode
        self.era_agreement_threshold = era_agreement_threshold
        self.era_agreement_power = era_agreement_power
        self.era_dro_gamma = era_dro_gamma
        self.era_dro_clip_min = era_dro_clip_min
        self.era_dro_clip_max = era_dro_clip_max
        self.restore_best_iteration = restore_best_iteration
        self.verbose = verbose
        self.random_state = random_state
        self.warm_start = warm_start

        self.forest = []
        self.bin_edges = None
        self.base_prediction = None
        self.unique_eras = None
        self.num_classes = None
        self.classes_ = None
        self.class_priors_ = None
        self.has_missing_ = False
        self.root_gradient_histogram = None
        self.root_hessian_histogram = None
        self.gradients = None
        self.root_node_indices = None
        self.bin_indices = None
        self.bin_indices_fm = None
        self.Y_gpu = None
        self.num_features = None
        self.num_samples = None
        self.best_bin = torch.tensor([-1], dtype=torch.int32, device=self.device)
        self.label_encoder = None
        self.feature_importance_ = None
        self.per_era_feature_importance_ = None
        self._is_fitted = False
        self._trees_trained = 0
        self._tree_tensors = None
        self._torch_rng = None
        self.best_iteration_ = 0
        self.best_eval_loss_ = float("inf")

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def _validate_hyperparams(self, **kwargs):
        valid_objectives = ["regression", "multiclass", "binary", "expectile", "huber"]
        if kwargs["objective"] not in valid_objectives:
            raise ValueError(
                f"objective must be one of {valid_objectives}, got {kwargs['objective']}."
            )

        int_params = [
            "num_bins",
            "max_depth",
            "n_estimators",
            "threads_per_block",
            "rows_per_thread",
        ]
        float_params = [
            "learning_rate",
            "min_split_gain",
            "L2_reg",
            "L1_reg",
            "colsample_bytree",
            "colsample_bylevel",
            "colsample_bynode",
            "path_smooth",
            "subsample",
            "expectile_alpha",
            "huber_delta",
            "era_agreement_threshold",
            "era_agreement_power",
            "era_dro_gamma",
            "era_dro_clip_min",
            "era_dro_clip_max",
        ]

        for param in int_params:
            if not isinstance(kwargs[param], int) or isinstance(kwargs[param], bool):
                raise TypeError(
                    f"{param} must be an integer, got {type(kwargs[param])}."
                )

        for param in float_params:
            if not isinstance(kwargs[param], (float, int)) or isinstance(kwargs[param], bool):
                raise TypeError(f"{param} must be a float, got {type(kwargs[param])}.")

        if kwargs["min_child_weight"] is not None:
            mcw = kwargs["min_child_weight"]
            if not isinstance(mcw, (int, float)) or isinstance(mcw, bool):
                raise TypeError(f"min_child_weight must be a positive number or None, got {type(mcw)}.")
            if mcw <= 0:
                raise ValueError("min_child_weight must be positive.")

        if not (2 <= kwargs["num_bins"] <= 127):
            raise ValueError("num_bins must be between 2 and 127 inclusive.")
        if kwargs["max_depth"] < 1:
            raise ValueError("max_depth must be at least 1.")
        if not (0.0 < kwargs["learning_rate"] <= 1.0):
            raise ValueError("learning_rate must be in (0.0, 1.0].")
        if kwargs["n_estimators"] <= 0:
            raise ValueError("n_estimators must be positive.")
        if kwargs["min_split_gain"] < 0:
            raise ValueError("min_split_gain must be non-negative.")
        if kwargs["threads_per_block"] <= 0 or kwargs["threads_per_block"] % 32 != 0:
            raise ValueError(
                "threads_per_block should be a positive multiple of 32 (warp size)."
            )
        if not (1 <= kwargs["rows_per_thread"] <= 16):
            raise ValueError(
                "rows_per_thread must be positive between 1 and 16 inclusive."
            )
        if kwargs["L2_reg"] < 0:
            raise ValueError("L2_reg must be non-negative.")
        if kwargs["L1_reg"] < 0:
            raise ValueError("L1_reg must be non-negative.")
        if kwargs["path_smooth"] < 0:
            raise ValueError("path_smooth must be non-negative.")
        if kwargs["colsample_bytree"] <= 0 or kwargs["colsample_bytree"] > 1:
            raise ValueError(
                f"Invalid colsample_bytree: {kwargs['colsample_bytree']}. Must be in (0, 1]."
            )
        if kwargs["colsample_bylevel"] <= 0 or kwargs["colsample_bylevel"] > 1:
            raise ValueError(
                f"Invalid colsample_bylevel: {kwargs['colsample_bylevel']}. Must be in (0, 1]."
            )
        if kwargs["colsample_bynode"] <= 0 or kwargs["colsample_bynode"] > 1:
            raise ValueError(
                f"Invalid colsample_bynode: {kwargs['colsample_bynode']}. Must be in (0, 1]."
            )
        if kwargs["pack_bits"] not in ("auto", True, False):
            raise ValueError("pack_bits must be 'auto', True, or False.")
        if kwargs["pack_bits"] is True and kwargs["num_bins"] > 15:
            raise ValueError("pack_bits=True requires num_bins <= 15.")
        if kwargs["prebinned"] not in ("auto", True, False):
            raise ValueError("prebinned must be 'auto', True, or False.")
        if kwargs["subsample"] <= 0 or kwargs["subsample"] > 1:
            raise ValueError(
                f"Invalid subsample: {kwargs['subsample']}. Must be in (0, 1]."
            )
        if kwargs["max_leaves"] is not None:
            if (
                not isinstance(kwargs["max_leaves"], int)
                or isinstance(kwargs["max_leaves"], bool)
                or kwargs["max_leaves"] < 2
            ):
                raise ValueError("max_leaves must be None or an integer >= 2.")
        if not (0.0 < kwargs["expectile_alpha"] < 1.0):
            raise ValueError("expectile_alpha must be in (0.0, 1.0).")
        if kwargs["huber_delta"] <= 0.0:
            raise ValueError("huber_delta must be positive.")
        if kwargs["era_split_mode"] not in _ERA_MODE_MAP:
            raise ValueError(
                f"era_split_mode must be one of {list(_ERA_MODE_MAP)}, got {kwargs['era_split_mode']}."
            )
        if not (0.0 <= kwargs["era_agreement_threshold"] <= 1.0):
            raise ValueError("era_agreement_threshold must be in [0.0, 1.0].")
        if kwargs["era_agreement_power"] < 0.0:
            raise ValueError("era_agreement_power must be non-negative.")
        if kwargs["era_dro_gamma"] < 0.0:
            raise ValueError("era_dro_gamma must be non-negative.")
        if kwargs["era_dro_clip_min"] <= 0.0 or kwargs["era_dro_clip_max"] < kwargs["era_dro_clip_min"]:
            raise ValueError("Invalid era_dro_clip_min / era_dro_clip_max bounds.")

    def _compute_tree_predictions(self, tree, bin_indices):
        """Compute predictions from a single tree dict on row-major [N, F] bin_indices."""
        num_samples = bin_indices.size(0)
        predictions = torch.zeros(num_samples, device=self.device, dtype=torch.float32)

        def traverse(node, sample_mask):
            if "leaf_value" in node:
                predictions[sample_mask] = node["leaf_value"] * self.learning_rate
            else:
                feature_idx = node["feature"]
                split_bin = node["bin"]
                default_left = node.get("default_left", True)
                col_bins = bin_indices[sample_mask, feature_idx]
                go_left = torch.where(
                    col_bins < 0,
                    torch.tensor(default_left, device=col_bins.device),
                    col_bins <= split_bin,
                )

                left_mask = sample_mask.clone()
                left_mask[sample_mask] = go_left
                right_mask = sample_mask.clone()
                right_mask[sample_mask] = ~go_left

                traverse(node["left"], left_mask)
                traverse(node["right"], right_mask)

        all_samples = torch.ones(num_samples, dtype=torch.bool, device=self.device)
        traverse(tree, all_samples)
        return predictions

    def _compute_softmax_gradients_hessians(self, y_true_encoded):
        """Compute gradients and diagonal hessians for softmax classification."""
        probs = softmax(self.gradients, dim=1)
        n_samples = y_true_encoded.shape[0]
        y_onehot = torch.zeros(n_samples, self.num_classes, device=self.device)
        y_onehot[torch.arange(n_samples, device=self.device), y_true_encoded.long()] = 1.0

        gradients = probs - y_onehot
        hessians = torch.clamp(probs * (1.0 - probs), min=1e-6)
        return gradients, hessians

    def _compute_regression_grad_hess(self):
        """Return (residual = -grad, hessian) for regression/expectile/huber."""
        diff = self.gradients - self.Y_gpu
        if self.objective == "regression":
            res = -diff
            hess = self._base_sample_weight if self._base_sample_weight is not None else self._empty_float
            if self.era_dro_gamma > 0.0 and self.num_eras > 1:
                loss_i = 0.5 * diff * diff
            else:
                loss_i = None
        elif self.objective == "expectile":
            w_down = 2.0 * (1.0 - self.expectile_alpha)
            w_up = 2.0 * self.expectile_alpha
            w = torch.where(diff > 0.0, w_down, w_up)
            res = -w * diff
            hess = w if self._base_sample_weight is None else (w * self._base_sample_weight)
            loss_i = 0.5 * w * diff * diff if (self.era_dro_gamma > 0.0 and self.num_eras > 1) else None
        else:  # pseudo-huber
            inv_d = 1.0 / self.huber_delta
            z = diff * inv_d
            s = torch.rsqrt(1.0 + z * z)
            res = -diff * s
            hess = torch.clamp(s * s * s, min=1e-6)
            if self._base_sample_weight is not None:
                hess = hess * self._base_sample_weight
            loss_i = (
                (self.huber_delta * self.huber_delta) * (torch.sqrt(1.0 + z * z) - 1.0)
                if (self.era_dro_gamma > 0.0 and self.num_eras > 1)
                else None
            )

        if self._base_sample_weight is not None and self.objective == "regression":
            res = res * self._base_sample_weight

        if loss_i is not None:
            era_w = self._compute_era_dro_weights(loss_i)
            sample_w = era_w[self.era_indices]
            res = res * sample_w
            if hess.numel() == 0:
                hess = sample_w
            else:
                hess = hess * sample_w

        return res, hess

    def _compute_era_dro_weights(self, loss_per_sample):
        """Closed-form Gibbs Minimax Group-DRO weights over eras (Sagawa et al. ICLR 2020)."""
        if self._base_sample_weight is not None:
            weighted_loss = loss_per_sample * self._base_sample_weight
            era_loss_sum = torch.zeros(self.num_eras, dtype=torch.float32, device=self.device)
            era_loss_sum.scatter_add_(0, self.era_indices, weighted_loss)
            era_losses = era_loss_sum / self._era_weight_sum
        else:
            era_loss_sum = torch.zeros(self.num_eras, dtype=torch.float32, device=self.device)
            era_loss_sum.scatter_add_(0, self.era_indices, loss_per_sample)
            era_losses = era_loss_sum / self._era_counts

        loss_scale = era_losses.std(unbiased=False) + 1e-8
        shifted_l = (era_losses - era_losses.max()) / loss_scale
        exp_terms = torch.exp(torch.clamp(self.era_dro_gamma * shifted_l, -30.0, 0.0))
        q = self._era_p0 * exp_terms
        q = q / torch.clamp(q.sum(), min=1e-12)
        ratio = torch.clamp(q / self._era_p0, self.era_dro_clip_min, self.era_dro_clip_max)
        mean_w = torch.clamp((ratio * self._era_p0).sum(), min=1e-12)
        return ratio / mean_w

    def validate_fit_params(
        self,
        X,
        y,
        era_id,
        X_eval,
        y_eval,
        eval_every_n_trees,
        early_stopping_rounds,
        eval_metric,
        sample_weight=None,
    ):
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X and y must be numpy arrays.")
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of rows. Got {X.shape[0]} and {y.shape[0]}."
            )

        if sample_weight is not None:
            if not isinstance(sample_weight, np.ndarray):
                raise TypeError("sample_weight must be a numpy array.")
            if sample_weight.ndim != 1 or len(sample_weight) != len(y):
                raise ValueError("sample_weight must be 1D with same length as y.")
            if not np.all(np.isfinite(sample_weight)) or np.any(sample_weight < 0):
                raise ValueError("sample_weight must be finite and non-negative.")

        if era_id is not None:
            if not isinstance(era_id, np.ndarray):
                raise TypeError("era_id must be a numpy array.")
            if era_id.ndim != 1:
                raise ValueError(f"era_id must be 1-dimensional, got shape {era_id.shape}")
            if len(era_id) != len(y):
                raise ValueError(
                    f"era_id must have same length as y. Got {len(era_id)} and {len(y)}."
                )

        eval_args = [X_eval, y_eval, eval_every_n_trees]
        if any(arg is not None for arg in eval_args):
            if X_eval is None or y_eval is None or eval_every_n_trees is None:
                raise ValueError(
                    "If using eval set, X_eval, y_eval, and eval_every_n_trees must all be defined."
                )
            if not isinstance(X_eval, np.ndarray) or not isinstance(y_eval, np.ndarray):
                raise TypeError("X_eval and y_eval must be numpy arrays.")
            if X_eval.ndim != 2:
                raise ValueError(f"X_eval must be 2-dimensional, got shape {X_eval.shape}")
            if y_eval.ndim != 1:
                raise ValueError(f"y_eval must be 1-dimensional, got shape {y_eval.shape}")
            if X_eval.shape[0] != y_eval.shape[0]:
                raise ValueError(
                    f"X_eval and y_eval must have same number of rows. Got {X_eval.shape[0]} and {y_eval.shape[0]}."
                )
            if not isinstance(eval_every_n_trees, int) or eval_every_n_trees <= 0:
                raise ValueError(
                    f"eval_every_n_trees must be a positive integer, got {eval_every_n_trees}."
                )
            if early_stopping_rounds is not None:
                if not isinstance(early_stopping_rounds, int) or early_stopping_rounds <= 0:
                    raise ValueError(
                        f"early_stopping_rounds must be a positive integer, got {early_stopping_rounds}."
                    )
            else:
                early_stopping_rounds = self.n_estimators + 1

            valid_metrics = ["mse", "corr", "rmsle", "logloss", "accuracy", "expectile", "huber"]
            if eval_metric not in valid_metrics:
                raise ValueError(
                    f"Invalid eval_metric: {eval_metric}. Choose from {valid_metrics}."
                )

        return early_stopping_rounds

    def fit(
        self,
        X,
        y,
        era_id=None,
        X_eval=None,
        y_eval=None,
        eval_every_n_trees=None,
        early_stopping_rounds=None,
        eval_metric="mse",
        sample_weight=None,
    ):
        early_stopping_rounds = self.validate_fit_params(
            X, y, era_id, X_eval, y_eval, eval_every_n_trees, early_stopping_rounds, eval_metric, sample_weight
        )

        is_continuing = self.warm_start and self._is_fitted
        if self.random_state is not None and not is_continuing:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.random_state)
                torch.cuda.manual_seed_all(self.random_state)
            self._torch_rng = torch.Generator(device=self.device)
            self._torch_rng.manual_seed(int(self.random_state))
        elif not is_continuing:
            self._torch_rng = None

        if not is_continuing:
            self._is_fitted = False
            self._trees_trained = 0
            self.forest = []
            self.training_loss = []
            self.eval_loss = []
            self.best_iteration_ = 0
            self.best_eval_loss_ = float("inf")
        else:
            if X.shape[1] != self.num_features:
                raise ValueError(
                    f"X has {X.shape[1]} features, but model was trained with {self.num_features} features."
                )
            if self._trees_trained >= self.n_estimators:
                self._log(
                    f"Model already has {self._trees_trained} trees (n_estimators={self.n_estimators}). No additional training needed."
                )
                return self

        self._tree_tensors = None
        if era_id is None:
            era_id = np.ones(X.shape[0], dtype="int32")

        if self.objective in ("multiclass", "binary"):
            if not is_continuing:
                self.label_encoder = LabelEncoder()
                y_encoded = self.label_encoder.fit_transform(y)
                self.classes_ = self.label_encoder.classes_
                self.num_classes = len(self.classes_)
            else:
                y_encoded = self.label_encoder.transform(y)

            if self.objective == "binary" and self.num_classes != 2:
                raise ValueError(f"binary objective requires exactly 2 classes, got {self.num_classes}")
            if self.objective == "multiclass" and self.num_classes < 2:
                raise ValueError(f"multiclass objective requires at least 2 classes, got {self.num_classes}")

            return self._fit_classification(
                X, y_encoded, era_id, X_eval, y_eval,
                eval_every_n_trees, early_stopping_rounds, eval_metric, sample_weight
            )
        else:
            return self._fit_regression(
                X, y, era_id, X_eval, y_eval,
                eval_every_n_trees, early_stopping_rounds, eval_metric, sample_weight
            )

    def _use_pack4(self, num_samples, num_features):
        if self.pack_bits is False:
            return False
        if self.num_bins > 15:
            return False
        if self.pack_bits is True:
            return True
        return (num_samples * num_features) >= 200_000

    def _init_era_and_weight_buffers(self, sample_weight):
        self._empty_float = torch.empty(0, dtype=torch.float32, device=self.device)
        self._empty_bool = torch.empty(0, dtype=torch.bool, device=self.device)
        if sample_weight is not None:
            self._base_sample_weight = torch.from_numpy(
                np.ascontiguousarray(sample_weight, dtype=np.float32)
            ).to(self.device)
        else:
            self._base_sample_weight = None

        if self.num_eras > 1 and self.era_dro_gamma > 0.0:
            if self._base_sample_weight is not None:
                w_sum = torch.zeros(self.num_eras, dtype=torch.float32, device=self.device)
                w_sum.scatter_add_(0, self.era_indices, self._base_sample_weight)
                self._era_weight_sum = torch.clamp(w_sum, min=1e-12)
                self._era_p0 = self._era_weight_sum / torch.clamp(self._era_weight_sum.sum(), min=1e-12)
            else:
                counts = torch.bincount(self.era_indices, minlength=self.num_eras).to(torch.float32)
                self._era_counts = torch.clamp(counts, min=1.0)
                self._era_p0 = self._era_counts / float(self.num_samples)

        self._num_hist_eras = 1 if (self.era_split_mode == "pooled" or self.num_eras == 1) else self.num_eras
        needs_hess = (
            self.objective != "regression"
            or (self.num_eras > 1 and self.era_dro_gamma > 0.0)
            or (self._base_sample_weight is not None)
        )
        self._gh_packed_buf = (
            torch.empty((self.num_samples, 2), dtype=torch.float32, device=self.device)
            if needs_hess
            else None
        )

    def _fit_regression(
        self, X, y, era_id, X_eval, y_eval,
        eval_every_n_trees, early_stopping_rounds, eval_metric, sample_weight=None,
    ):
        self.bin_indices_fm, self.era_indices, self.bin_edges, self.unique_eras, self.Y_gpu = (
            self._preprocess_gpu_data_fm(X, y, era_id)
        )
        self.num_samples, self.num_features = X.shape
        self.num_eras = len(self.unique_eras)
        self.era_indices = self.era_indices.to(dtype=torch.int32)
        self._init_era_and_weight_buffers(sample_weight)

        if self.warm_start and self._is_fitted and self._trees_trained > 0:
            self.gradients = self._predict_binned_fm(
                self.bin_indices_fm, num_samples=self.num_samples, is_packed=self._packed_fm
            )
        else:
            if self._base_sample_weight is not None:
                w_sum = torch.clamp(self._base_sample_weight.sum(), min=1e-12)
                self.base_prediction = ((self.Y_gpu * self._base_sample_weight).sum() / w_sum).item()
            else:
                self.base_prediction = self.Y_gpu.mean().item()
            if self.objective == "expectile" and self.expectile_alpha != 0.5:
                # 3 Newton steps for optimal constant initial expectile
                b_pred = torch.tensor(self.base_prediction, dtype=torch.float32, device=self.device)
                w_down = 2.0 * (1.0 - self.expectile_alpha)
                w_up = 2.0 * self.expectile_alpha
                for _ in range(3):
                    diff = b_pred - self.Y_gpu
                    w = torch.where(diff > 0.0, w_down, w_up)
                    if self._base_sample_weight is not None:
                        w = w * self._base_sample_weight
                    b_pred = (w * self.Y_gpu).sum() / torch.clamp(w.sum(), min=1e-12)
                self.base_prediction = b_pred.item()
            self.gradients = torch.full(
                (self.num_samples,), self.base_prediction, dtype=torch.float32, device=self.device
            )

        self.root_node_indices = torch.arange(self.num_samples, device=self.device, dtype=torch.int32)
        self.feature_indices = torch.arange(self.num_features, device=self.device, dtype=torch.int32)

        if X_eval is not None and y_eval is not None:
            self._eval_num_samples = X_eval.shape[0]
            self._eval_packed_fm = self._use_pack4(self._eval_num_samples, self.num_features)
            self.bin_indices_eval_fm = self._bin_inference_data_fm(X_eval, pack4=self._eval_packed_fm)
            self.Y_gpu_eval = torch.from_numpy(np.ascontiguousarray(y_eval, dtype=np.float32)).to(self.device)
            self.eval_every_n_trees = eval_every_n_trees
            self.early_stopping_rounds = early_stopping_rounds
            self.eval_metric = eval_metric
        else:
            self.bin_indices_eval_fm = None
            self._eval_num_samples = 0
            self._eval_packed_fm = False
            self.Y_gpu_eval = None
            self.eval_every_n_trees = None
            self.early_stopping_rounds = None

        with torch.no_grad():
            self.grow_forest()

        del self.bin_indices_fm, self.Y_gpu, self._base_sample_weight, self._empty_float, self._empty_bool, self._gh_packed_buf
        self.bin_indices_fm = None
        self._gh_packed_buf = None
        if hasattr(self, "bin_indices_eval_fm") and self.bin_indices_eval_fm is not None:
            del self.bin_indices_eval_fm, self.Y_gpu_eval
            self.bin_indices_eval_fm = None
            self.Y_gpu_eval = None
        gc.collect()
        return self

    def _fit_classification(
        self, X, y_encoded, era_id, X_eval, y_eval,
        eval_every_n_trees, early_stopping_rounds, eval_metric, sample_weight=None,
    ):
        self.bin_indices_fm, self.era_indices, self.bin_edges, self.unique_eras, _ = (
            self._preprocess_gpu_data_fm(X, y_encoded, era_id)
        )
        self.num_samples, self.num_features = X.shape
        self.num_eras = len(self.unique_eras)
        self.era_indices = self.era_indices.to(dtype=torch.int32)
        self.Y_gpu = torch.from_numpy(np.ascontiguousarray(y_encoded, dtype=np.int32)).to(self.device)
        self._init_era_and_weight_buffers(sample_weight)

        if not self.warm_start or not self._is_fitted:
            counts = torch.bincount(self.Y_gpu, minlength=self.num_classes).float()
            priors = counts / float(self.num_samples)
            self.class_priors_ = torch.log(priors + 1e-10).cpu().numpy().tolist()

        if self.warm_start and self._is_fitted and self._trees_trained > 0:
            self.gradients = self._predict_raw_multiclass_fm(
                self.bin_indices_fm, num_samples=self.num_samples, is_packed=self._packed_fm
            )
        else:
            self.gradients = torch.zeros(
                (self.num_samples, self.num_classes), dtype=torch.float32, device=self.device
            )
            for k in range(self.num_classes):
                if self.class_priors_[k] != 0.0:
                    self.gradients[:, k] = self.class_priors_[k]

        self.root_node_indices = torch.arange(self.num_samples, device=self.device, dtype=torch.int32)
        self.feature_indices = torch.arange(self.num_features, device=self.device, dtype=torch.int32)

        if X_eval is not None and y_eval is not None:
            self._eval_num_samples = X_eval.shape[0]
            self._eval_packed_fm = self._use_pack4(self._eval_num_samples, self.num_features)
            self.bin_indices_eval_fm = self._bin_inference_data_fm(X_eval, pack4=self._eval_packed_fm)
            y_eval_encoded = self.label_encoder.transform(y_eval)
            self.Y_gpu_eval = torch.from_numpy(np.ascontiguousarray(y_eval_encoded, dtype=np.int32)).to(self.device)
            self.eval_every_n_trees = eval_every_n_trees
            self.early_stopping_rounds = early_stopping_rounds
            self.eval_metric = eval_metric if eval_metric != "mse" else "logloss"
        else:
            self.bin_indices_eval_fm = None
            self._eval_num_samples = 0
            self._eval_packed_fm = False
            self.Y_gpu_eval = None
            self.eval_every_n_trees = None
            self.early_stopping_rounds = None

        with torch.no_grad():
            self.grow_forest_multiclass()

        del self.bin_indices_fm, self.Y_gpu, self._base_sample_weight, self._empty_float, self._empty_bool, self._gh_packed_buf
        self.bin_indices_fm = None
        self._gh_packed_buf = None
        if hasattr(self, "bin_indices_eval_fm") and self.bin_indices_eval_fm is not None:
            del self.bin_indices_eval_fm, self.Y_gpu_eval
            self.bin_indices_eval_fm = None
            self.Y_gpu_eval = None
        gc.collect()
        return self

    def _compute_bin_edges_gpu(self, X_np):
        """Compute quantile bin edges in chunks on GPU while ignoring NaNs."""
        num_samples, num_features = X_np.shape
        bin_edges = torch.zeros(
            (num_features, self.num_bins - 1), dtype=torch.float32, device=self.device
        )
        quantiles = torch.linspace(
            0, 1, self.num_bins + 1, device=self.device, dtype=torch.float32
        )[1:-1]
        has_missing = False
        f_chunk = 256

        for f0 in range(0, num_features, f_chunk):
            f1 = min(f0 + f_chunk, num_features)
            X_blk = torch.from_numpy(
                np.ascontiguousarray(X_np[:, f0:f1], dtype=np.float32)
            ).to(self.device)
            nan_mask = torch.isnan(X_blk)
            if not nan_mask.any():
                edges_blk = torch.quantile(X_blk, quantiles, dim=0).t().contiguous()
                bin_edges[f0:f1] = edges_blk
            else:
                has_missing = True
                for j in range(f1 - f0):
                    col = X_blk[:, j]
                    col_nan = nan_mask[:, j]
                    if not col_nan.any():
                        bin_edges[f0 + j] = torch.quantile(col, quantiles)
                    else:
                        valid = col[~col_nan]
                        if valid.numel() > 0:
                            bin_edges[f0 + j] = torch.quantile(valid, quantiles)
            del X_blk, nan_mask
        return bin_edges, has_missing

    def _is_prebinned_int(self, X_np):
        if not np.issubdtype(X_np.dtype, np.integer):
            return False, False
        if X_np.size == 0:
            return True, False
        max_val = int(X_np.max())
        if max_val >= self.num_bins:
            return False, False
        min_val = int(X_np.min())
        if min_val < -1:
            return False, False
        return True, (min_val < 0)

    def preprocess_gpu_data(self, X_np, Y_np, era_id_np):
        """Row-major [N, F] preprocessing retained for API/test compatibility."""
        with torch.no_grad():
            self.num_samples, self.num_features = X_np.shape
            Y_gpu = torch.from_numpy(np.ascontiguousarray(Y_np, dtype=np.float32)).to(self.device)
            era_id_gpu = torch.from_numpy(np.ascontiguousarray(era_id_np, dtype=np.int32)).to(self.device)
            unique_eras, era_indices = torch.unique(era_id_gpu, return_inverse=True)

            is_prebinned, has_missing = self._is_prebinned_int(X_np)
            if is_prebinned:
                self._log("Detected pre-binned integer input — skipping quantile binning.")
                self.has_missing_ = has_missing
                bin_indices = torch.as_tensor(
                    np.ascontiguousarray(X_np, dtype=np.int8), device=self.device
                )
                bin_edges = torch.arange(1, self.num_bins, dtype=torch.float32, device=self.device).repeat(
                    self.num_features, 1
                )
                return bin_indices, era_indices, bin_edges, unique_eras, Y_gpu

            self._log("quantile binning.")
            bin_edges, has_missing = self._compute_bin_edges_gpu(X_np)
            self.has_missing_ = has_missing
            bin_indices = torch.empty(
                (self.num_samples, self.num_features), dtype=torch.int8, device=self.device
            )
            f_chunk = 256
            for f0 in range(0, self.num_features, f_chunk):
                f1 = min(f0 + f_chunk, self.num_features)
                X_blk = torch.from_numpy(
                    np.ascontiguousarray(X_np[:, f0:f1], dtype=np.float32)
                ).to(self.device)
                out_blk = torch.empty((self.num_samples, f1 - f0), dtype=torch.int8, device=self.device)
                node_kernel.bin_2d(X_blk, bin_edges[f0:f1].contiguous(), out_blk, False)
                bin_indices[:, f0:f1] = out_blk
                del X_blk, out_blk
            return bin_indices, era_indices, bin_edges, unique_eras, Y_gpu

    def _can_try_prebinned(self, X_np):
        if self.prebinned is False:
            return False
        if self.prebinned is True:
            return np.issubdtype(X_np.dtype, np.integer)
        if not np.issubdtype(X_np.dtype, np.integer):
            return False
        if X_np.size == 0:
            return True
        head = X_np[: min(256, X_np.shape[0])]
        return int(head.min()) >= -1 and int(head.max()) < self.num_bins

    def _upload_prebinned_fm(self, X_np, bin_indices_fm, pack4, detect_missing):
        """Upload integer chunks into [F, N] int8 or [F, (N+1)//2] 4-bit packed GPU buffer.
        Returns (ok, has_missing)."""
        num_samples, num_features = X_np.shape
        flags_gpu = torch.zeros(2, dtype=torch.int32, device=self.device)
        empty_flags = torch.empty(0, dtype=torch.int32, device=self.device)
        r_chunk = max(2, ((64 * 1024 * 1024) // max(1, num_features)) & ~1)
        is_i8 = (X_np.dtype == np.int8)
        need_bounds = (self.prebinned != True)

        for n0 in range(0, num_samples, r_chunk):
            n1 = min(n0 + r_chunk, num_samples)
            chunk_np = X_np[n0:n1]
            if not is_i8:
                raw_blk = torch.from_numpy(np.ascontiguousarray(chunk_np)).to(self.device)
                if need_bounds and bool(((raw_blk < -1) | (raw_blk >= self.num_bins)).any()):
                    del raw_blk
                    return False, False
                if detect_missing and bool((raw_blk < 0).any()):
                    flags_gpu[1] = 1
                blk = raw_blk.to(torch.int8)
                del raw_blk
            else:
                blk = torch.from_numpy(np.ascontiguousarray(chunk_np, dtype=np.int8)).to(self.device)

            if pack4:
                f_arg = flags_gpu if (is_i8 and (need_bounds or detect_missing)) else empty_flags
                node_kernel.pack_int8_to_4bit_fm(blk, bin_indices_fm, f_arg, n0 // 2, self.num_bins)
            else:
                if is_i8 and need_bounds and bool(((blk < -1) | (blk >= self.num_bins)).any()):
                    del blk
                    return False, False
                if is_i8 and detect_missing and bool((blk < 0).any()):
                    flags_gpu[1] = 1
                bin_indices_fm[:, n0:n1].copy_(blk.t())
            del blk

        flag_vals = flags_gpu.tolist()
        if flag_vals[0] != 0 and need_bounds:
            return False, False
        return True, bool(flag_vals[1])

    def _preprocess_gpu_data_fm(self, X_np, Y_np, era_id_np):
        """Chunked feature-major [F, N] or 4-bit packed [F, (N+1)//2] preprocessing."""
        with torch.no_grad():
            self.num_samples, self.num_features = X_np.shape
            Y_gpu = torch.from_numpy(np.ascontiguousarray(Y_np, dtype=np.float32)).to(self.device)
            era_id_gpu = torch.from_numpy(np.ascontiguousarray(era_id_np, dtype=np.int32)).to(self.device)
            unique_eras, era_indices = torch.unique(era_id_gpu, return_inverse=True)

            pack4 = self._use_pack4(self.num_samples, self.num_features)
            self._packed_fm = pack4
            stride_n = ((self.num_samples + 1) // 2) if pack4 else self.num_samples
            bin_dtype = torch.uint8 if pack4 else torch.int8
            bin_indices_fm = torch.empty(
                (self.num_features, stride_n), dtype=bin_dtype, device=self.device
            )

            if self._can_try_prebinned(X_np):
                ok, has_missing = self._upload_prebinned_fm(
                    X_np, bin_indices_fm, pack4=pack4, detect_missing=True
                )
                if ok:
                    self._log("Detected pre-binned integer input — skipping quantile binning.")
                    self.has_missing_ = has_missing
                    bin_edges = torch.arange(1, self.num_bins, dtype=torch.float32, device=self.device).repeat(
                        self.num_features, 1
                    )
                    return bin_indices_fm, era_indices, bin_edges, unique_eras, Y_gpu

            self._log("quantile binning.")
            bin_edges, has_missing = self._compute_bin_edges_gpu(X_np)
            self.has_missing_ = has_missing
            edges_contig = bin_edges.contiguous()
            r_chunk = max(2, ((32 * 1024 * 1024) // max(1, self.num_features)) & ~1)
            for n0 in range(0, self.num_samples, r_chunk):
                n1 = min(n0 + r_chunk, self.num_samples)
                X_blk = torch.from_numpy(
                    np.ascontiguousarray(X_np[n0:n1], dtype=np.float32)
                ).to(self.device)
                if pack4:
                    node_kernel.bin_2d_pack4_fm(X_blk, edges_contig, bin_indices_fm, n0 // 2)
                else:
                    out_blk = torch.empty((self.num_features, n1 - n0), dtype=torch.int8, device=self.device)
                    node_kernel.bin_2d(X_blk, edges_contig, out_blk, True)
                    bin_indices_fm[:, n0:n1].copy_(out_blk)
                    del out_blk
                del X_blk
            return bin_indices_fm, era_indices, bin_edges, unique_eras, Y_gpu

    def bin_data_with_existing_edges(self, X_np):
        """Row-major [N, F] binning using stored bin_edges."""
        num_samples, num_features = X_np.shape
        bin_indices = torch.empty((num_samples, num_features), dtype=torch.int8, device=self.device)
        f_chunk = 256
        with torch.no_grad():
            for f0 in range(0, num_features, f_chunk):
                f1 = min(f0 + f_chunk, num_features)
                X_blk = torch.from_numpy(
                    np.ascontiguousarray(X_np[:, f0:f1], dtype=np.float32)
                ).to(self.device)
                out_blk = torch.empty((num_samples, f1 - f0), dtype=torch.int8, device=self.device)
                node_kernel.bin_2d(X_blk, self.bin_edges[f0:f1].contiguous(), out_blk, False)
                bin_indices[:, f0:f1] = out_blk
                del X_blk, out_blk
        return bin_indices

    def bin_inference_data(self, X_np):
        """Row-major [N, F] inference binning retained for API/test compatibility."""
        is_prebinned, _ = self._is_prebinned_int(X_np)
        if is_prebinned and X_np.shape[1] == self.num_features:
            self._log("Detected pre-binned input at predict-time — skipping binning.")
            return torch.as_tensor(np.ascontiguousarray(X_np, dtype=np.int8), device=self.device)
        return self.bin_data_with_existing_edges(X_np)

    def _bin_inference_data_fm(self, X_np, pack4=False):
        """Feature-major [F, N] or 4-bit packed [F, (N+1)//2] chunked inference binning."""
        num_samples, num_features = X_np.shape
        if num_features != self.num_features:
            raise ValueError(
                f"X has {num_features} features, expected {self.num_features}."
            )
        stride_n = ((num_samples + 1) // 2) if pack4 else num_samples
        bin_indices_fm = torch.empty(
            (num_features, stride_n),
            dtype=torch.uint8 if pack4 else torch.int8,
            device=self.device,
        )

        with torch.no_grad():
            if self._can_try_prebinned(X_np):
                ok, _ = self._upload_prebinned_fm(
                    X_np, bin_indices_fm, pack4=pack4, detect_missing=False
                )
                if ok:
                    self._log("Detected pre-binned input at predict-time — skipping binning.")
                    return bin_indices_fm

            r_chunk = max(2, ((32 * 1024 * 1024) // max(1, num_features)) & ~1)
            edges_contig = self.bin_edges.contiguous()
            for n0 in range(0, num_samples, r_chunk):
                n1 = min(n0 + r_chunk, num_samples)
                X_blk = torch.from_numpy(
                    np.ascontiguousarray(X_np[n0:n1], dtype=np.float32)
                ).to(self.device)
                if pack4:
                    node_kernel.bin_2d_pack4_fm(X_blk, edges_contig, bin_indices_fm, n0 // 2)
                else:
                    out_blk = torch.empty((num_features, n1 - n0), dtype=torch.int8, device=self.device)
                    node_kernel.bin_2d(X_blk, edges_contig, out_blk, True)
                    bin_indices_fm[:, n0:n1].copy_(out_blk)
                    del out_blk
                del X_blk
        return bin_indices_fm

    def compute_histograms(self, sample_indices, feature_indices):
        """Legacy [N, F] histogram builder."""
        grad_hist = torch.zeros(
            (self.num_eras, len(feature_indices), self.num_bins),
            device=self.device,
            dtype=torch.float32,
        )
        hess_hist = torch.zeros(
            (self.num_eras, len(feature_indices), self.num_bins),
            device=self.device,
            dtype=torch.float32,
        )
        node_kernel.compute_histogram3(
            self.bin_indices,
            self.residual,
            sample_indices,
            feature_indices,
            self.era_indices,
            grad_hist,
            hess_hist,
            self.num_bins,
            self.threads_per_block,
            self.rows_per_thread,
        )
        return grad_hist, hess_hist

    def _compute_histograms_fm(
        self,
        sample_indices,
        feature_indices,
        grad_hist=None,
        hess_hist=None,
        miss_grad=None,
        miss_hess=None,
    ):
        """Fast feature-major weighted histogram builder using pre-allocated buffers."""
        E = self._num_hist_eras
        k = len(feature_indices)
        if grad_hist is None:
            grad_hist = torch.zeros((E, k, self.num_bins), device=self.device, dtype=torch.float32)
        else:
            grad_hist.zero_()
        if hess_hist is None:
            hess_hist = torch.zeros_like(grad_hist)
        else:
            hess_hist.zero_()

        if self.has_missing_:
            if miss_grad is None:
                miss_grad = torch.zeros((E, k), device=self.device, dtype=torch.float32)
            else:
                miss_grad.zero_()
            if miss_hess is None:
                miss_hess = torch.zeros_like(miss_grad)
            else:
                miss_hess.zero_()
        else:
            miss_grad = self._empty_float
            miss_hess = self._empty_float

        self._last_miss_g = miss_grad
        self._last_miss_h = miss_hess

        eff_tb = max(128, self.threads_per_block) if E == 1 else self.threads_per_block
        eff_rpt = max(16, self.rows_per_thread) if E == 1 else self.rows_per_thread
        gh_packed = getattr(self, "gh_packed", None)
        if gh_packed is None:
            if self.hessians.numel() > 0:
                gh_packed = torch.stack((self.residual, self.hessians), dim=1).contiguous()
            else:
                gh_packed = self._empty_float

        node_kernel.compute_histogram_fm_v2(
            self.bin_indices_fm,
            self.residual,
            gh_packed,
            sample_indices,
            feature_indices,
            self.era_indices,
            grad_hist,
            hess_hist,
            miss_grad,
            miss_hess,
            self.num_bins,
            bool(getattr(self, "_packed_fm", False)),
            bool(self.has_missing_),
            eff_tb,
            eff_rpt,
        )

        if E == 1:
            return grad_hist, hess_hist, None, None
        total_G = grad_hist[:, 0, :].sum(dim=1)
        total_H = hess_hist[:, 0, :].sum(dim=1)
        if self.has_missing_:
            total_G = total_G + miss_grad[:, 0]
            total_H = total_H + miss_hess[:, 0]
        return grad_hist, hess_hist, total_G, total_H

    def _compute_node_totals(self, sample_indices):
        if self._num_hist_eras == 1:
            total_G = self.residual[sample_indices].sum().unsqueeze(0)
            if self.hessians.numel() > 0:
                total_H = self.hessians[sample_indices].sum().unsqueeze(0)
            else:
                total_H = torch.tensor(
                    [float(sample_indices.numel())], dtype=torch.float32, device=self.device
                )
            return total_G, total_H

        eras = self.era_indices[sample_indices]
        total_G = torch.zeros(self._num_hist_eras, dtype=torch.float32, device=self.device)
        total_G.scatter_add_(0, eras, self.residual[sample_indices])
        total_H = torch.zeros(self._num_hist_eras, dtype=torch.float32, device=self.device)
        if self.hessians.numel() > 0:
            total_H.scatter_add_(0, eras, self.hessians[sample_indices])
        else:
            total_H.scatter_add_(
                0,
                eras,
                torch.ones(sample_indices.numel(), dtype=torch.float32, device=self.device),
            )
        return total_G, total_H

    def _allocate_tree_buffers(self, k_features):
        E = self._num_hist_eras
        B = self.num_bins
        M = (B - 1) * (2 if self.has_missing_ else 1)
        self._out_best_pooled = torch.empty(10, device=self.device, dtype=torch.float32)
        n_sub = max(1, int(self.subsample * self.num_samples)) if self.subsample < 1.0 else self.num_samples
        self._mask_scratch = torch.empty(n_sub, device=self.device, dtype=torch.bool)
        self._last_miss_g = self._empty_float
        self._last_miss_h = self._empty_float

        if E > 1:
            self._cand_gain = torch.empty((E, k_features, M), device=self.device, dtype=torch.float32)
            self._cand_dir = torch.empty((E, k_features, M), device=self.device, dtype=torch.float32)
            self._out_score = torch.empty((k_features, M), device=self.device, dtype=torch.float32)
            self._out_agr = torch.empty((k_features, M), device=self.device, dtype=torch.float32)
            self._out_mean_gain = torch.empty((k_features, M), device=self.device, dtype=torch.float32)
        else:
            self._cand_gain = None
            self._cand_dir = None
            self._out_score = None
            self._out_agr = None
            self._out_mean_gain = None

        if self.max_leaves is None:
            self._hist_pool_g = [
                torch.empty((E, k_features, B), device=self.device, dtype=torch.float32)
                for _ in range(self.max_depth + 1)
            ]
            self._hist_pool_h = [
                torch.empty((E, k_features, B), device=self.device, dtype=torch.float32)
                for _ in range(self.max_depth + 1)
            ]
            if self.has_missing_:
                self._miss_pool_g = [
                    torch.empty((E, k_features), device=self.device, dtype=torch.float32)
                    for _ in range(self.max_depth + 1)
                ]
                self._miss_pool_h = [
                    torch.empty((E, k_features), device=self.device, dtype=torch.float32)
                    for _ in range(self.max_depth + 1)
                ]
            else:
                self._miss_pool_g = [self._empty_float] * (self.max_depth + 1)
                self._miss_pool_h = [self._empty_float] * (self.max_depth + 1)
            self._lw_pool_g = None
            self._lw_pool_h = None
            self._lw_miss_g = None
            self._lw_miss_h = None
        else:
            n_slots = self.max_leaves + 1
            self._hist_pool_g = None
            self._hist_pool_h = None
            self._miss_pool_g = None
            self._miss_pool_h = None
            self._lw_pool_g = [
                torch.empty((E, k_features, B), device=self.device, dtype=torch.float32)
                for _ in range(n_slots)
            ]
            self._lw_pool_h = [
                torch.empty((E, k_features, B), device=self.device, dtype=torch.float32)
                for _ in range(n_slots)
            ]
            if self.has_missing_:
                self._lw_miss_g = [
                    torch.empty((E, k_features), device=self.device, dtype=torch.float32)
                    for _ in range(n_slots)
                ]
                self._lw_miss_h = [
                    torch.empty((E, k_features), device=self.device, dtype=torch.float32)
                    for _ in range(n_slots)
                ]
            else:
                self._lw_miss_g = [self._empty_float] * n_slots
                self._lw_miss_h = [self._empty_float] * n_slots

    def _effective_min_child_weight(self):
        if self.min_child_weight is not None:
            return float(self.min_child_weight)
        return 1.0 if self.objective in ("binary", "multiclass") else 20.0

    def find_best_split(self, gradient_histogram, hessian_histogram):
        """Legacy split finder on existing [E, F, B] buffers."""
        node_kernel.compute_split(
            gradient_histogram,
            hessian_histogram,
            self.min_split_gain,
            self._effective_min_child_weight(),
            self.L2_reg,
            self.per_era_gain,
            self.per_era_direction,
            self.threads_per_block,
        )
        if self.num_eras == 1:
            era_splitting_criterion = self.per_era_gain[0, :, :].clone()
            dir_score_mask = era_splitting_criterion > self.min_split_gain
        else:
            directional_agreement = self.per_era_direction.mean(dim=0).abs()
            era_splitting_criterion = self.per_era_gain.mean(dim=0)
            dir_score_mask = (directional_agreement == directional_agreement.max()) & (
                era_splitting_criterion > self.min_split_gain
            )
        if not dir_score_mask.any():
            return -1, -1
        era_splitting_criterion[dir_score_mask == 0] = float("-inf")
        best_idx = torch.argmax(era_splitting_criterion)
        split_bins = self.num_bins - 1
        return (best_idx // split_bins).item(), (best_idx % split_bins).item()

    def _find_best_split_fm(
        self,
        grad_hist,
        hess_hist,
        total_G,
        total_H,
        level_mask=None,
        miss_g=None,
        miss_h=None,
        parent_val=0.0,
    ):
        """Fused CUDA split finder returning (local_feat, best_bin, default_left, best_cand, best_score)."""
        if self._num_hist_eras == 1:
            mg = self._last_miss_g if miss_g is None else miss_g
            mh = self._last_miss_h if miss_h is None else miss_h
            mask_t = self._empty_bool if level_mask is None else level_mask
            node_kernel.find_best_split_pooled(
                grad_hist,
                hess_hist,
                mg if self.has_missing_ else self._empty_float,
                mh if self.has_missing_ else self._empty_float,
                mask_t,
                float(self.min_split_gain),
                self._effective_min_child_weight(),
                float(self.L1_reg),
                float(self.L2_reg),
                float(self.path_smooth),
                float(parent_val),
                bool(self.has_missing_),
                self._out_best_pooled,
            )
            b = self._out_best_pooled.tolist()
            self._last_pooled_best = b
            local_feat = int(b[0])
            if local_feat < 0:
                return -1, -1, True, -1, float("-inf")
            best_bin = int(b[1])
            default_left = b[2] > 0.5
            best_score = b[3]
            split_bins = self.num_bins - 1
            best_cand = best_bin if default_left else (split_bins + best_bin)
            return local_feat, best_bin, default_left, best_cand, best_score

        era_mode = _ERA_MODE_MAP[self.era_split_mode]
        node_kernel.find_best_split_fused(
            grad_hist,
            hess_hist,
            total_G if self.has_missing_ else self._empty_float,
            total_H if self.has_missing_ else self._empty_float,
            self._empty_float,
            float(self.min_split_gain),
            self._effective_min_child_weight(),
            float(self.L1_reg),
            float(self.L2_reg),
            bool(self.has_missing_),
            era_mode,
            float(self.era_agreement_threshold),
            float(self.era_agreement_power),
            self._cand_gain,
            self._cand_dir,
            self._out_score,
            self._out_agr,
            self._out_mean_gain,
            max(128, self.threads_per_block),
        )

        scores = self._out_score
        if level_mask is not None:
            scores = scores.clone()
            scores[~level_mask, :] = float("-inf")

        if era_mode == 0:
            # Backward-compatible DES max-agreement selection
            agr = self._out_agr
            if level_mask is not None:
                agr = agr.clone()
                agr[~level_mask, :] = -1.0
            max_agr = agr.max()
            valid = (agr == max_agr) & (scores > float("-inf"))
            if not valid.any():
                return -1, -1, True, -1, float("-inf")
            masked_scores = torch.where(valid, scores, float("-inf"))
            best_idx = torch.argmax(masked_scores)
            best_score = masked_scores.view(-1)[best_idx].item()
        else:
            best_idx = torch.argmax(scores)
            best_score = scores.view(-1)[best_idx].item()
            if best_score == float("-inf"):
                return -1, -1, True, -1, float("-inf")

        M = scores.size(1)
        split_bins = self.num_bins - 1
        idx_int = best_idx.item()
        local_feat = idx_int // M
        best_cand = idx_int % M
        default_left = best_cand < split_bins
        best_bin = best_cand if default_left else (best_cand - split_bins)
        return local_feat, best_bin, default_left, best_cand, best_score

    def _compute_leaf_value_scalar(self, G_sum, H_sum, parent_val=0.0):
        s = float(self.path_smooth)
        denom = float(H_sum) + self.L2_reg + s
        if denom <= 0.0:
            return 0.0
        g_eff = float(G_sum) + s * float(parent_val)
        if self.L1_reg > 0.0:
            abs_g = abs(g_eff)
            if abs_g <= self.L1_reg:
                return 0.0
            val = (abs_g - self.L1_reg) / denom
            return val if g_eff >= 0.0 else -val
        return g_eff / denom

    def _compute_leaf_value(self, total_G, total_H, parent_val=0.0):
        if isinstance(total_G, (int, float)):
            return self._compute_leaf_value_scalar(total_G, total_H, parent_val=parent_val)
        return self._compute_leaf_value_scalar(total_G.sum().item(), total_H.sum().item(), parent_val=parent_val)

    def _assign_leaf(self, node_indices, total_G, total_H, class_k=None, parent_val=0.0):
        leaf_value = self._compute_leaf_value(total_G, total_H, parent_val=parent_val)
        if class_k is not None:
            self.gradients[node_indices, class_k] += self.learning_rate * leaf_value
        else:
            self.gradients[node_indices] += self.learning_rate * leaf_value
        return {"leaf_value": float(leaf_value), "samples": int(node_indices.numel())}

    def _partition_indices(self, node_indices, global_feature_idx, best_bin, default_left):
        n_node = node_indices.numel()
        split_mask = self._mask_scratch[:n_node]
        node_kernel.extract_split_mask(
            self.bin_indices_fm,
            node_indices,
            int(global_feature_idx),
            int(best_bin),
            bool(default_left),
            bool(getattr(self, "_packed_fm", False)),
            split_mask,
        )
        left_indices = node_indices[split_mask]
        right_indices = node_indices[~split_mask]
        return left_indices, right_indices

    def _sample_feat_mask(self, k_tree, level_mask=None):
        mask = level_mask
        if self.colsample_bynode < 1.0:
            k_node = max(1, int(self.colsample_bynode * k_tree))
            chosen = torch.randperm(
                k_tree, generator=self._torch_rng, device=self.device
            )[:k_node]
            node_mask = torch.zeros(k_tree, dtype=torch.bool, device=self.device)
            node_mask[chosen] = True
            mask = node_mask if mask is None else (mask & node_mask)
        return mask

    def _sample_level_mask(self, k_tree):
        level_mask = None
        if self.colsample_bylevel < 1.0:
            k_lvl = max(1, int(self.colsample_bylevel * k_tree))
            chosen = torch.randperm(
                k_tree, generator=self._torch_rng, device=self.device
            )[:k_lvl]
            level_mask = torch.zeros(k_tree, dtype=torch.bool, device=self.device)
            level_mask[chosen] = True
        return self._sample_feat_mask(k_tree, level_mask=level_mask)

    def grow_tree(
        self, grad_hist, hess_hist, total_G, total_H, node_indices, depth,
        class_k=None, miss_g=None, miss_h=None, parent_val=0.0,
    ):
        if depth == self.max_depth:
            return self._assign_leaf(node_indices, total_G, total_H, class_k, parent_val=parent_val)

        level_mask = self._sample_level_mask(self.feat_indices_tree.numel())
        local_feature, best_bin, default_left, best_cand, best_score = self._find_best_split_fm(
            grad_hist, hess_hist, total_G, total_H, level_mask=level_mask,
            miss_g=miss_g, miss_h=miss_h, parent_val=parent_val,
        )

        pooled = self._num_hist_eras == 1
        if local_feature == -1:
            if pooled and total_G is None:
                total_G = self._last_pooled_best[8]
                total_H = self._last_pooled_best[9]
            return self._assign_leaf(node_indices, total_G, total_H, class_k, parent_val=parent_val)

        if pooled:
            b = self._last_pooled_best
            g_l_tot, h_l_tot, g_r_tot, h_r_tot = b[4], b[5], b[6], b[7]
            if self.path_smooth > 0.0:
                if total_G is None:
                    cur_val = self._compute_leaf_value_scalar(b[8], b[9], parent_val=parent_val)
                else:
                    cur_val = self._compute_leaf_value_scalar(total_G, total_H, parent_val=parent_val)
            else:
                cur_val = 0.0
            global_feature_idx = self._feat_tree_list[local_feature]
            self._pooled_feat_imp[global_feature_idx] += best_score
        else:
            global_feature_idx = self.feat_indices_tree[local_feature].item()
            per_era_gains = self._cand_gain[:, local_feature, best_cand]
            self._per_era_feature_importance_gpu[:, global_feature_idx] += per_era_gains
            cur_val = self._compute_leaf_value(total_G, total_H, parent_val=parent_val) if self.path_smooth > 0.0 else 0.0

        left_indices, right_indices = self._partition_indices(
            node_indices, global_feature_idx, best_bin, default_left
        )
        left_size = left_indices.numel()
        right_size = right_indices.numel()

        new_depth = depth + 1
        buf_g = self._hist_pool_g[new_depth]
        buf_h = self._hist_pool_h[new_depth]
        buf_mg = self._miss_pool_g[new_depth]
        buf_mh = self._miss_pool_h[new_depth]

        if left_size <= right_size:
            grad_hist_left, hess_hist_left, tot_G_left, tot_H_left = self._compute_histograms_fm(
                left_indices, self.feat_indices_tree, buf_g, buf_h, buf_mg, buf_mh
            )
            grad_hist.sub_(grad_hist_left)
            hess_hist.sub_(hess_hist_left)
            if self.has_missing_:
                miss_g.sub_(buf_mg)
                miss_h.sub_(buf_mh)
            grad_hist_right = grad_hist
            hess_hist_right = hess_hist
            mg_right, mh_right = miss_g, miss_h
            if pooled:
                tot_G_left, tot_H_left = g_l_tot, h_l_tot
                tot_G_right, tot_H_right = g_r_tot, h_r_tot
            else:
                tot_G_right = total_G - tot_G_left
                tot_H_right = total_H - tot_H_left

            right_child = self.grow_tree(
                grad_hist_right, hess_hist_right, tot_G_right, tot_H_right,
                right_indices, new_depth, class_k, miss_g=mg_right, miss_h=mh_right, parent_val=cur_val,
            )
            left_child = self.grow_tree(
                grad_hist_left, hess_hist_left, tot_G_left, tot_H_left,
                left_indices, new_depth, class_k, miss_g=buf_mg, miss_h=buf_mh, parent_val=cur_val,
            )
        else:
            grad_hist_right, hess_hist_right, tot_G_right, tot_H_right = self._compute_histograms_fm(
                right_indices, self.feat_indices_tree, buf_g, buf_h, buf_mg, buf_mh
            )
            grad_hist.sub_(grad_hist_right)
            hess_hist.sub_(hess_hist_right)
            if self.has_missing_:
                miss_g.sub_(buf_mg)
                miss_h.sub_(buf_mh)
            grad_hist_left = grad_hist
            hess_hist_left = hess_hist
            mg_left, mh_left = miss_g, miss_h
            if pooled:
                tot_G_left, tot_H_left = g_l_tot, h_l_tot
                tot_G_right, tot_H_right = g_r_tot, h_r_tot
            else:
                tot_G_left = total_G - tot_G_right
                tot_H_left = total_H - tot_H_right

            left_child = self.grow_tree(
                grad_hist_left, hess_hist_left, tot_G_left, tot_H_left,
                left_indices, new_depth, class_k, miss_g=mg_left, miss_h=mh_left, parent_val=cur_val,
            )
            right_child = self.grow_tree(
                grad_hist_right, hess_hist_right, tot_G_right, tot_H_right,
                right_indices, new_depth, class_k, miss_g=buf_mg, miss_h=buf_mh, parent_val=cur_val,
            )

        node = {
            "feature": int(global_feature_idx),
            "bin": int(best_bin),
            "left": left_child,
            "right": right_child,
        }
        if self.has_missing_:
            node["default_left"] = bool(default_left)
        return node

    def grow_tree_leafwise(
        self, grad_hist, hess_hist, total_G, total_H, node_indices,
        class_k=None, miss_g=None, miss_h=None,
    ):
        """Best-first leaf-wise tree growth up to max_leaves using pre-allocated histogram pool."""
        root_ref = {}
        heap = []
        uid = 0
        pooled = self._num_hist_eras == 1
        free_slots = list(range(1, len(self._lw_pool_g)))

        def evaluate_and_push(node_ref, n_idx, slot, t_g, t_h, depth, parent_val=0.0):
            nonlocal uid
            if depth >= self.max_depth:
                node_ref.update(self._assign_leaf(n_idx, t_g, t_h, class_k, parent_val=parent_val))
                free_slots.append(slot)
                return 0
            g_h = self._lw_pool_g[slot]
            h_h = self._lw_pool_h[slot]
            mg = self._lw_miss_g[slot]
            mh = self._lw_miss_h[slot]
            level_mask = self._sample_level_mask(self.feat_indices_tree.numel())
            lf, bb, dl, bc, best_score = self._find_best_split_fm(
                g_h, h_h, t_g, t_h, level_mask=level_mask,
                miss_g=mg, miss_h=mh, parent_val=parent_val,
            )
            if lf == -1:
                if pooled and t_g is None:
                    t_g = self._last_pooled_best[8]
                    t_h = self._last_pooled_best[9]
                node_ref.update(self._assign_leaf(n_idx, t_g, t_h, class_k, parent_val=parent_val))
                free_slots.append(slot)
                return 0
            if pooled:
                b = self._last_pooled_best
                extra = (b[4], b[5], b[6], b[7], t_g if t_g is not None else b[8], t_h if t_h is not None else b[9], best_score, parent_val)
            else:
                per_era_gains = self._cand_gain[:, lf, bc].clone()
                extra = (per_era_gains, parent_val)
            uid += 1
            heapq.heappush(
                heap,
                (-best_score, uid, node_ref, n_idx, slot, t_g, t_h, lf, bb, dl, extra, depth),
            )
            return 1

        active_leaves = 1
        evaluate_and_push(root_ref, node_indices, 0, total_G, total_H, 0, 0.0)

        while heap and active_leaves < self.max_leaves:
            (
                _, _, node_ref, n_idx, s_p, t_g, t_h,
                lf, bb, dl, extra, depth
            ) = heapq.heappop(heap)

            if pooled:
                gl, hl, gr, hr, tp_g, tp_h, best_score, parent_val = extra
                global_feature_idx = self._feat_tree_list[lf]
                self._pooled_feat_imp[global_feature_idx] += best_score
                cur_val = self._compute_leaf_value_scalar(tp_g, tp_h, parent_val=parent_val) if self.path_smooth > 0.0 else 0.0
            else:
                per_era_gains, parent_val = extra
                global_feature_idx = self.feat_indices_tree[lf].item()
                self._per_era_feature_importance_gpu[:, global_feature_idx] += per_era_gains
                cur_val = self._compute_leaf_value(t_g, t_h, parent_val=parent_val) if self.path_smooth > 0.0 else 0.0

            left_idx, right_idx = self._partition_indices(n_idx, global_feature_idx, bb, dl)
            left_ref, right_ref = {}, {}
            node_ref["feature"] = int(global_feature_idx)
            node_ref["bin"] = int(bb)
            if self.has_missing_:
                node_ref["default_left"] = bool(dl)
            node_ref["left"] = left_ref
            node_ref["right"] = right_ref
            active_leaves += 1

            g_p = self._lw_pool_g[s_p]
            h_p = self._lw_pool_h[s_p]
            mg_p = self._lw_miss_g[s_p]
            mh_p = self._lw_miss_h[s_p]

            s_small = free_slots.pop()
            g_s = self._lw_pool_g[s_small]
            h_s = self._lw_pool_h[s_small]
            mg_s = self._lw_miss_g[s_small]
            mh_s = self._lw_miss_h[s_small]

            if left_idx.numel() <= right_idx.numel():
                _, _, tg_l, th_l = self._compute_histograms_fm(
                    left_idx, self.feat_indices_tree, g_s, h_s, mg_s, mh_s
                )
                g_p.sub_(g_s)
                h_p.sub_(h_s)
                if self.has_missing_:
                    mg_p.sub_(mg_s)
                    mh_p.sub_(mh_s)
                s_left, s_right = s_small, s_p
                if pooled:
                    tg_l, th_l, tg_r, th_r = gl, hl, gr, hr
                else:
                    tg_r, th_r = t_g - tg_l, t_h - th_l
            else:
                _, _, tg_r, th_r = self._compute_histograms_fm(
                    right_idx, self.feat_indices_tree, g_s, h_s, mg_s, mh_s
                )
                g_p.sub_(g_s)
                h_p.sub_(h_s)
                if self.has_missing_:
                    mg_p.sub_(mg_s)
                    mh_p.sub_(mh_s)
                s_left, s_right = s_p, s_small
                if pooled:
                    tg_l, th_l, tg_r, th_r = gl, hl, gr, hr
                else:
                    tg_l, th_l = t_g - tg_r, t_h - th_r

            new_depth = depth + 1
            if active_leaves < self.max_leaves:
                evaluate_and_push(left_ref, left_idx, s_left, tg_l, th_l, new_depth, cur_val)
                evaluate_and_push(right_ref, right_idx, s_right, tg_r, th_r, new_depth, cur_val)
            else:
                left_ref.update(self._assign_leaf(left_idx, tg_l, th_l, class_k, parent_val=cur_val))
                right_ref.update(self._assign_leaf(right_idx, tg_r, th_r, class_k, parent_val=cur_val))
                free_slots.append(s_left)
                free_slots.append(s_right)

        while heap:
            _, _, node_ref, n_idx, slot, t_g, t_h, *rest = heapq.heappop(heap)
            extra = rest[3]
            if pooled:
                t_g, t_h, _, parent_val = extra[4], extra[5], extra[6], extra[7]
            else:
                parent_val = extra[1]
            node_ref.update(self._assign_leaf(n_idx, t_g, t_h, class_k, parent_val=parent_val))
            free_slots.append(slot)

        return root_ref

    def get_eval_metric(self, y_true, y_pred):
        if self.eval_metric == "mse":
            return ((y_true - y_pred) ** 2).mean().item()
        elif self.eval_metric == "expectile":
            return expectile_loss_torch(y_true, y_pred, alpha=self.expectile_alpha).item()
        elif self.eval_metric == "huber":
            return pseudo_huber_loss_torch(y_true, y_pred, delta=self.huber_delta).item()
        elif self.eval_metric == "corr":
            if y_pred.std() < 1e-12 or y_true.std() < 1e-12:
                return 1.0
            return 1.0 - torch.corrcoef(torch.vstack([y_true, y_pred]))[0, 1].item()
        elif self.eval_metric == "rmsle":
            return rmsle_torch(y_true, y_pred).item()
        else:
            raise ValueError(f"Invalid eval_metric: {self.eval_metric}.")

    def _compute_train_regression_loss(self):
        if self.objective == "expectile":
            return expectile_loss_torch(self.Y_gpu, self.gradients, alpha=self.expectile_alpha).item()
        elif self.objective == "huber":
            return pseudo_huber_loss_torch(self.Y_gpu, self.gradients, delta=self.huber_delta).item()
        return ((self.Y_gpu - self.gradients) ** 2).mean().item()

    def compute_eval(self, i):
        if self.eval_every_n_trees is None:
            return
        if i % self.eval_every_n_trees == 0:
            train_loss = self._compute_train_regression_loss()
            self.training_loss.append(train_loss)
            eval_preds = self._predict_binned_fm(
                self.bin_indices_eval_fm,
                num_samples=self._eval_num_samples,
                is_packed=self._eval_packed_fm,
            )
            eval_loss = self.get_eval_metric(self.Y_gpu_eval, eval_preds)
            self.eval_loss.append(eval_loss)

            if np.isfinite(eval_loss) and eval_loss < self.best_eval_loss_:
                self.best_eval_loss_ = eval_loss
                self.best_iteration_ = i + 1
                self._evals_without_improvement = 0
            else:
                self._evals_without_improvement += 1
                if (
                    self.early_stopping_rounds is not None
                    and self._evals_without_improvement >= self.early_stopping_rounds
                ):
                    self.stop = True

            self._log(
                f"🌲 Tree {i+1}/{self.n_estimators} | Train {self.objective}: {train_loss:.6f} | Eval {self.eval_metric}: {eval_loss:.6f}"
            )

    def grow_forest(self):
        if not hasattr(self, "training_loss") or not self.warm_start or not self._is_fitted:
            self.training_loss = []
            self.eval_loss = []
            self.per_era_feature_importance_ = np.zeros(
                (self._num_hist_eras, self.num_features), dtype=np.float32
            )
            self.best_iteration_ = 0
            self.best_eval_loss_ = float("inf")

        pooled = self._num_hist_eras == 1
        if pooled:
            self._pooled_feat_imp = self.per_era_feature_importance_[0].copy()
        else:
            self._per_era_feature_importance_gpu = torch.as_tensor(
                self.per_era_feature_importance_, device=self.device
            )
        self.stop = False
        self._evals_without_improvement = 0

        if self.colsample_bytree < 1.0:
            k = max(1, int(self.colsample_bytree * self.num_features))
        else:
            self.feat_indices_tree = self.feature_indices
            self._feat_tree_list = list(range(self.num_features))
            k = self.num_features

        self._allocate_tree_buffers(k)

        start_iter = self._trees_trained if self.warm_start and self._is_fitted else 0
        if len(self.forest) < self.n_estimators:
            self.forest.extend([{} for _ in range(self.n_estimators - len(self.forest))])

        for i in range(start_iter, self.n_estimators):
            self.residual, self.hessians = self._compute_regression_grad_hess()
            if self._gh_packed_buf is not None and self.hessians.numel() > 0:
                torch.stack((self.residual, self.hessians), dim=1, out=self._gh_packed_buf)
                self.gh_packed = self._gh_packed_buf
            else:
                self.gh_packed = self._empty_float

            if self.colsample_bytree < 1.0:
                self.feat_indices_tree = torch.randperm(
                    self.num_features, generator=self._torch_rng, device=self.device, dtype=torch.int32
                )[:k]
                if pooled:
                    self._feat_tree_list = self.feat_indices_tree.tolist()

            if self.subsample < 1.0:
                n_sub = max(1, int(self.subsample * self.num_samples))
                tree_samples = torch.sort(
                    torch.randperm(
                        self.num_samples, generator=self._torch_rng, device=self.device, dtype=torch.int32
                    )[:n_sub]
                ).values
            else:
                tree_samples = self.root_node_indices

            if self.max_leaves is None:
                g0, h0 = self._hist_pool_g[0], self._hist_pool_h[0]
                mg0, mh0 = self._miss_pool_g[0], self._miss_pool_h[0]
            else:
                g0, h0 = self._lw_pool_g[0], self._lw_pool_h[0]
                mg0, mh0 = self._lw_miss_g[0], self._lw_miss_h[0]

            root_g, root_h, root_tot_g, root_tot_h = self._compute_histograms_fm(
                tree_samples, self.feat_indices_tree, g0, h0, mg0, mh0
            )

            if self.max_leaves is None:
                tree = self.grow_tree(
                    root_g, root_h, root_tot_g, root_tot_h, tree_samples, 0,
                    miss_g=mg0, miss_h=mh0, parent_val=0.0,
                )
            else:
                tree = self.grow_tree_leafwise(
                    root_g, root_h, root_tot_g, root_tot_h, tree_samples,
                    miss_g=mg0, miss_h=mh0,
                )

            self.forest[i] = tree
            self._tree_tensors = None
            self._trees_trained = i + 1

            self.compute_eval(i)
            if self.stop:
                break

        if self.stop and self.restore_best_iteration and self.best_iteration_ > 0:
            self._trees_trained = self.best_iteration_
        self.forest = self.forest[: self._trees_trained]
        self._tree_tensors = None

        if pooled:
            self.per_era_feature_importance_[0] = self._pooled_feat_imp
            del self._pooled_feat_imp
        else:
            self.per_era_feature_importance_ = self._per_era_feature_importance_gpu.cpu().numpy()
            del self._per_era_feature_importance_gpu
        del self._cand_gain, self._cand_dir, self._out_score, self._out_agr, self._out_mean_gain
        del self._hist_pool_g, self._hist_pool_h, self._miss_pool_g, self._miss_pool_h
        del self._lw_pool_g, self._lw_pool_h, self._lw_miss_g, self._lw_miss_h
        del self._out_best_pooled, self._mask_scratch
        self.feature_importance_ = self.per_era_feature_importance_.sum(axis=0)
        self._is_fitted = True

        self._log(f"Finished training forest. Total trees: {self._trees_trained}")

    def grow_forest_multiclass(self):
        if not hasattr(self, "training_loss") or not self.warm_start or not self._is_fitted:
            self.training_loss = []
            self.eval_loss = []
            self.per_era_feature_importance_ = np.zeros(
                (self._num_hist_eras, self.num_features), dtype=np.float32
            )
            self.forest = []
            self.best_iteration_ = 0
            self.best_eval_loss_ = float("inf")

        pooled = self._num_hist_eras == 1
        if pooled:
            self._pooled_feat_imp = self.per_era_feature_importance_[0].copy()
        else:
            self._per_era_feature_importance_gpu = torch.as_tensor(
                self.per_era_feature_importance_, device=self.device
            )
        self.stop = False
        self._evals_without_improvement = 0

        if self.colsample_bytree < 1.0:
            k = max(1, int(self.colsample_bytree * self.num_features))
        else:
            self.feat_indices_tree = self.feature_indices
            self._feat_tree_list = list(range(self.num_features))
            k = self.num_features

        self._allocate_tree_buffers(k)
        start_iter = self._trees_trained if self.warm_start and self._is_fitted else 0

        for i in range(start_iter, self.n_estimators):
            grads, hess = self._compute_softmax_gradients_hessians(self.Y_gpu)
            if self._base_sample_weight is not None:
                grads = grads * self._base_sample_weight.unsqueeze(1)
                hess = hess * self._base_sample_weight.unsqueeze(1)

            if self.era_dro_gamma > 0.0 and self.num_eras > 1:
                probs = softmax(self.gradients, dim=1)
                p_true = torch.clamp(
                    probs[torch.arange(self.num_samples, device=self.device), self.Y_gpu.long()],
                    1e-15,
                    1.0,
                )
                loss_i = -torch.log(p_true)
                era_w = self._compute_era_dro_weights(loss_i)
                sw = era_w[self.era_indices].unsqueeze(1)
                grads = grads * sw
                hess = hess * sw

            trees_k = []
            if self.colsample_bytree < 1.0:
                self.feat_indices_tree = torch.randperm(
                    self.num_features, generator=self._torch_rng, device=self.device, dtype=torch.int32
                )[:k]
                if pooled:
                    self._feat_tree_list = self.feat_indices_tree.tolist()

            if self.subsample < 1.0:
                n_sub = max(1, int(self.subsample * self.num_samples))
                tree_samples = torch.sort(
                    torch.randperm(
                        self.num_samples, generator=self._torch_rng, device=self.device, dtype=torch.int32
                    )[:n_sub]
                ).values
            else:
                tree_samples = self.root_node_indices

            for class_k in range(self.num_classes):
                self.residual = -grads[:, class_k].contiguous()
                self.hessians = hess[:, class_k].contiguous()
                if self._gh_packed_buf is not None:
                    torch.stack((self.residual, self.hessians), dim=1, out=self._gh_packed_buf)
                    self.gh_packed = self._gh_packed_buf
                else:
                    self.gh_packed = self._empty_float

                if self.max_leaves is None:
                    g0, h0 = self._hist_pool_g[0], self._hist_pool_h[0]
                    mg0, mh0 = self._miss_pool_g[0], self._miss_pool_h[0]
                else:
                    g0, h0 = self._lw_pool_g[0], self._lw_pool_h[0]
                    mg0, mh0 = self._lw_miss_g[0], self._lw_miss_h[0]

                root_g, root_h, root_tot_g, root_tot_h = self._compute_histograms_fm(
                    tree_samples, self.feat_indices_tree, g0, h0, mg0, mh0
                )

                if self.max_leaves is None:
                    tree_k = self.grow_tree(
                        root_g, root_h, root_tot_g, root_tot_h, tree_samples, 0,
                        class_k=class_k, miss_g=mg0, miss_h=mh0, parent_val=0.0,
                    )
                else:
                    tree_k = self.grow_tree_leafwise(
                        root_g, root_h, root_tot_g, root_tot_h, tree_samples,
                        class_k=class_k, miss_g=mg0, miss_h=mh0,
                    )
                trees_k.append(tree_k)

            self.forest.append(trees_k)
            self._tree_tensors = None
            self._trees_trained = i + 1

            self.compute_eval_multiclass(i)
            if self.stop:
                break

        if self.stop and self.restore_best_iteration and self.best_iteration_ > 0:
            self._trees_trained = self.best_iteration_
            self.forest = self.forest[: self._trees_trained]
            self._tree_tensors = None

        if pooled:
            self.per_era_feature_importance_[0] = self._pooled_feat_imp
            del self._pooled_feat_imp
        else:
            self.per_era_feature_importance_ = self._per_era_feature_importance_gpu.cpu().numpy()
            del self._per_era_feature_importance_gpu
        del self._cand_gain, self._cand_dir, self._out_score, self._out_agr, self._out_mean_gain
        del self._hist_pool_g, self._hist_pool_h, self._miss_pool_g, self._miss_pool_h
        del self._lw_pool_g, self._lw_pool_h, self._lw_miss_g, self._lw_miss_h
        del self._out_best_pooled, self._mask_scratch
        self.feature_importance_ = self.per_era_feature_importance_.sum(axis=0)
        self._is_fitted = True

        self._log(
            f"Finished training multiclass forest. Total rounds: {self._trees_trained} ({self._trees_trained * self.num_classes} trees)"
        )

    def compute_eval_multiclass(self, i):
        if self.eval_every_n_trees is None:
            return
        if i % self.eval_every_n_trees == 0:
            probs_train = softmax(self.gradients, dim=1)
            train_loss = log_loss_torch(self.Y_gpu, probs_train).item()
            self.training_loss.append(train_loss)

            eval_probs = self._predict_proba_binned_fm(
                self.bin_indices_eval_fm,
                num_samples=self._eval_num_samples,
                is_packed=self._eval_packed_fm,
            )
            if self.eval_metric == "logloss":
                eval_loss = log_loss_torch(self.Y_gpu_eval, eval_probs).item()
            elif self.eval_metric == "accuracy":
                eval_preds = torch.argmax(eval_probs, dim=1)
                eval_loss = 1.0 - accuracy_torch(self.Y_gpu_eval, eval_preds).item()
            else:
                eval_loss = log_loss_torch(self.Y_gpu_eval, eval_probs).item()

            self.eval_loss.append(eval_loss)
            if np.isfinite(eval_loss) and eval_loss < self.best_eval_loss_:
                self.best_eval_loss_ = eval_loss
                self.best_iteration_ = i + 1
                self._evals_without_improvement = 0
            else:
                self._evals_without_improvement += 1
                if (
                    self.early_stopping_rounds is not None
                    and self._evals_without_improvement >= self.early_stopping_rounds
                ):
                    self.stop = True

            self._log(
                f"🌲 Round {i+1}/{self.n_estimators} ({self.num_classes} trees) | "
                f"Train LogLoss: {train_loss:.6f} | Eval {self.eval_metric}: {eval_loss:.6f}"
            )

    def _get_tree_tensors(self):
        if self._tree_tensors is None:
            max_nodes = 2 ** (self.max_depth + 1)
            if self.max_leaves is not None:
                max_nodes = min(max_nodes, 2 * self.max_leaves)
            if self.objective not in ("multiclass", "binary"):
                trees = [self.flatten_tree(tree, max_nodes) for tree in self.forest if tree]
                if trees:
                    self._tree_tensors = torch.stack(trees).to(self.device).contiguous()
                else:
                    self._tree_tensors = torch.empty((0, max_nodes, 7), dtype=torch.float32, device=self.device)
            else:
                # Stack per class: [K, T, max_nodes, 7]
                non_empty = [r for r in self.forest if r]
                if non_empty:
                    per_class = []
                    for k in range(self.num_classes):
                        per_class.append(
                            torch.stack([self.flatten_tree(r[k], max_nodes) for r in non_empty]).to(self.device)
                        )
                    self._tree_tensors = per_class
                else:
                    self._tree_tensors = [
                        torch.empty((0, max_nodes, 7), dtype=torch.float32, device=self.device)
                        for _ in range(self.num_classes)
                    ]
        return self._tree_tensors

    def predict_binned(self, bin_indices):
        """Predict from row-major [N, F] bin_indices."""
        num_samples = bin_indices.size(0)
        tree_tensor = self._get_tree_tensors()[:, :, :6].contiguous()
        out = torch.full((num_samples,), self.base_prediction, dtype=torch.float32, device=self.device)
        node_kernel.predict_forest(bin_indices.contiguous(), tree_tensor, float(self.learning_rate), out)
        return out

    def _predict_binned_fm(self, bin_indices_fm, num_samples=None, is_packed=False):
        if num_samples is None:
            num_samples = bin_indices_fm.size(1)
        tree_tensor = self._get_tree_tensors()
        out = torch.full((num_samples,), self.base_prediction, dtype=torch.float32, device=self.device)
        if is_packed:
            node_kernel.predict_forest_fm_packed(
                bin_indices_fm.contiguous(), int(num_samples), tree_tensor, float(self.learning_rate), 0, 1, out
            )
        else:
            node_kernel.predict_forest_fm(
                bin_indices_fm.contiguous(), tree_tensor, float(self.learning_rate), 0, 1, out
            )
        return out

    def _predict_raw_multiclass_fm(self, bin_indices_fm, num_samples=None, is_packed=False):
        if num_samples is None:
            num_samples = bin_indices_fm.size(1)
        F = torch.zeros((num_samples, self.num_classes), device=self.device, dtype=torch.float32)
        priors = self.class_priors_ or [0.0] * self.num_classes
        for k in range(self.num_classes):
            if priors[k] != 0.0:
                F[:, k] = priors[k]

        per_class_tensors = self._get_tree_tensors()
        bin_c = bin_indices_fm.contiguous()
        for k, tree_tensor in enumerate(per_class_tensors):
            if tree_tensor.size(0) > 0:
                if is_packed:
                    node_kernel.predict_forest_fm_packed(
                        bin_c,
                        int(num_samples),
                        tree_tensor.contiguous(),
                        float(self.learning_rate),
                        k,
                        self.num_classes,
                        F,
                    )
                else:
                    node_kernel.predict_forest_fm(
                        bin_c,
                        tree_tensor.contiguous(),
                        float(self.learning_rate),
                        k,
                        self.num_classes,
                        F,
                    )
        return F

    def _predict_proba_binned_fm(self, bin_indices_fm, num_samples=None, is_packed=False):
        return softmax(
            self._predict_raw_multiclass_fm(bin_indices_fm, num_samples=num_samples, is_packed=is_packed),
            dim=1,
        )

    def predict_proba_binned(self, bin_indices):
        """Predict probabilities from row-major [N, F] bin_indices."""
        num_samples = bin_indices.size(0)
        F = torch.zeros((num_samples, self.num_classes), device=self.device, dtype=torch.float32)
        priors = self.class_priors_ or [0.0] * self.num_classes
        for k in range(self.num_classes):
            if priors[k] != 0.0:
                F[:, k] = priors[k]

        per_class_tensors = self._get_tree_tensors()
        bin_rm = bin_indices.contiguous()
        for k, tree_tensor in enumerate(per_class_tensors):
            if tree_tensor.size(0) > 0:
                class_preds = torch.zeros(num_samples, device=self.device, dtype=torch.float32)
                node_kernel.predict_forest(
                    bin_rm, tree_tensor[:, :, :6].contiguous(), float(self.learning_rate), class_preds
                )
                F[:, k] += class_preds
        return softmax(F, dim=1)

    def predict(self, X_np):
        if self.objective in ("multiclass", "binary"):
            probs = self.predict_proba(X_np)
            class_indices = np.argmax(probs, axis=1)
            return self.label_encoder.inverse_transform(class_indices)
        n_samples = X_np.shape[0]
        chunk_rows = 262_144
        if n_samples <= chunk_rows:
            use_p4 = self._use_pack4(n_samples, self.num_features)
            bin_indices_fm = self._bin_inference_data_fm(X_np, pack4=use_p4)
            preds = self._predict_binned_fm(
                bin_indices_fm, num_samples=n_samples, is_packed=use_p4
            ).cpu().numpy()
            del bin_indices_fm
            return preds

        out_np = np.empty(n_samples, dtype=np.float32)
        for n0 in range(0, n_samples, chunk_rows):
            n1 = min(n0 + chunk_rows, n_samples)
            cn = n1 - n0
            use_p4 = self._use_pack4(cn, self.num_features)
            bin_fm = self._bin_inference_data_fm(X_np[n0:n1], pack4=use_p4)
            out_np[n0:n1] = self._predict_binned_fm(
                bin_fm, num_samples=cn, is_packed=use_p4
            ).cpu().numpy()
            del bin_fm
        return out_np

    def predict_proba(self, X_np):
        if self.objective not in ["multiclass", "binary"]:
            raise ValueError("predict_proba only available for classification objectives")
        n_samples = X_np.shape[0]
        chunk_rows = 262_144
        if n_samples <= chunk_rows:
            use_p4 = self._use_pack4(n_samples, self.num_features)
            bin_indices_fm = self._bin_inference_data_fm(X_np, pack4=use_p4)
            probs = self._predict_proba_binned_fm(
                bin_indices_fm, num_samples=n_samples, is_packed=use_p4
            ).cpu().numpy()
            del bin_indices_fm
            return probs

        out_np = np.empty((n_samples, self.num_classes), dtype=np.float32)
        for n0 in range(0, n_samples, chunk_rows):
            n1 = min(n0 + chunk_rows, n_samples)
            cn = n1 - n0
            use_p4 = self._use_pack4(cn, self.num_features)
            bin_fm = self._bin_inference_data_fm(X_np[n0:n1], pack4=use_p4)
            out_np[n0:n1] = self._predict_proba_binned_fm(
                bin_fm, num_samples=cn, is_packed=use_p4
            ).cpu().numpy()
            del bin_fm
        return out_np

    def get_feature_importance(self, importance_type="gain", normalize=True):
        if self.feature_importance_ is None:
            raise ValueError("Model has not been fitted yet.")
        if importance_type != "gain":
            raise ValueError(f"importance_type '{importance_type}' not supported. Use 'gain'.")
        importance = self.feature_importance_.copy()
        if normalize and importance.sum() > 0:
            importance = importance / importance.sum()
        return importance

    def get_per_era_feature_importance(self, normalize=True):
        if self.per_era_feature_importance_ is None:
            raise ValueError("Model has not been fitted yet.")
        importance = self.per_era_feature_importance_.copy()
        if normalize:
            for era_idx in range(importance.shape[0]):
                era_sum = importance[era_idx].sum()
                if era_sum > 0:
                    importance[era_idx] /= era_sum
        return importance

    def save_model(self, path):
        if not self._is_fitted:
            raise ValueError("Cannot save unfitted model. Call fit() first.")

        state = {
            "objective": self.objective,
            "num_bins": self.num_bins,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "min_child_weight": self.min_child_weight,
            "min_split_gain": self.min_split_gain,
            "L2_reg": self.L2_reg,
            "L1_reg": self.L1_reg,
            "colsample_bytree": self.colsample_bytree,
            "colsample_bylevel": self.colsample_bylevel,
            "colsample_bynode": self.colsample_bynode,
            "path_smooth": self.path_smooth,
            "pack_bits": self.pack_bits,
            "prebinned": self.prebinned,
            "subsample": self.subsample,
            "max_leaves": self.max_leaves,
            "expectile_alpha": self.expectile_alpha,
            "huber_delta": self.huber_delta,
            "era_split_mode": self.era_split_mode,
            "era_agreement_threshold": self.era_agreement_threshold,
            "era_agreement_power": self.era_agreement_power,
            "era_dro_gamma": self.era_dro_gamma,
            "era_dro_clip_min": self.era_dro_clip_min,
            "era_dro_clip_max": self.era_dro_clip_max,
            "restore_best_iteration": self.restore_best_iteration,
            "verbose": self.verbose,
            "random_state": self.random_state,
            "warm_start": self.warm_start,
            "forest": self.forest,
            "bin_edges": self.bin_edges,
            "base_prediction": self.base_prediction,
            "num_features": self.num_features,
            "num_classes": self.num_classes,
            "classes_": self.classes_,
            "class_priors_": self.class_priors_,
            "has_missing_": self.has_missing_,
            "label_encoder": self.label_encoder,
            "feature_importance_": self.feature_importance_,
            "per_era_feature_importance_": self.per_era_feature_importance_,
            "_is_fitted": self._is_fitted,
            "_trees_trained": self._trees_trained,
            "best_iteration_": self.best_iteration_,
            "best_eval_loss_": self.best_eval_loss_,
            "training_loss": self.training_loss if hasattr(self, "training_loss") else [],
            "eval_loss": self.eval_loss if hasattr(self, "eval_loss") else [],
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)
        self._log(f"Model saved to {path}")

    def load_model(self, path):
        with open(path, "rb") as f:
            state = pickle.load(f)

        self.objective = state["objective"]
        self.num_bins = state["num_bins"]
        self.max_depth = state["max_depth"]
        self.learning_rate = state["learning_rate"]
        self.n_estimators = state["n_estimators"]
        self.min_child_weight = state["min_child_weight"]
        self.min_split_gain = state["min_split_gain"]
        self.L2_reg = state["L2_reg"]
        self.L1_reg = state.get("L1_reg", 0.0)
        self.colsample_bytree = state["colsample_bytree"]
        self.colsample_bylevel = state.get("colsample_bylevel", 1.0)
        self.colsample_bynode = state.get("colsample_bynode", 1.0)
        self.path_smooth = state.get("path_smooth", 0.0)
        self.pack_bits = state.get("pack_bits", "auto")
        self.prebinned = state.get("prebinned", "auto")
        self.subsample = state.get("subsample", 1.0)
        self.max_leaves = state.get("max_leaves", None)
        self.expectile_alpha = state.get("expectile_alpha", 0.5)
        self.huber_delta = state.get("huber_delta", 1.0)
        self.era_split_mode = state.get("era_split_mode", "max_agreement")
        self.era_agreement_threshold = state.get("era_agreement_threshold", 0.75)
        self.era_agreement_power = state.get("era_agreement_power", 2.0)
        self.era_dro_gamma = state.get("era_dro_gamma", 0.0)
        self.era_dro_clip_min = state.get("era_dro_clip_min", 0.2)
        self.era_dro_clip_max = state.get("era_dro_clip_max", 5.0)
        self.restore_best_iteration = state.get("restore_best_iteration", True)
        self.verbose = state.get("verbose", 1)
        self.random_state = state["random_state"]
        self.warm_start = state["warm_start"]

        self.forest = state["forest"]
        self._tree_tensors = None
        self.bin_edges = state["bin_edges"]
        self.base_prediction = state["base_prediction"]
        self.num_features = state["num_features"]
        self.num_classes = state["num_classes"]
        self.classes_ = state["classes_"]
        self.class_priors_ = state.get("class_priors_", None)
        self.has_missing_ = state.get("has_missing_", False)
        self.label_encoder = state["label_encoder"]
        self.feature_importance_ = state["feature_importance_"]
        self.per_era_feature_importance_ = state["per_era_feature_importance_"]
        self._is_fitted = state["_is_fitted"]
        self._trees_trained = state["_trees_trained"]
        self.best_iteration_ = state.get("best_iteration_", self._trees_trained)
        self.best_eval_loss_ = state.get("best_eval_loss_", float("inf"))
        self.training_loss = state.get("training_loss", [])
        self.eval_loss = state.get("eval_loss", [])

        self._log(f"Model loaded from {path} ({self._trees_trained} trees)")
        return self

    def flatten_tree(self, tree, max_nodes):
        flat = torch.full((max_nodes, 7), float("nan"), dtype=torch.float32)
        node_counter = [0]
        node_list = []

        def walk(node):
            curr_id = node_counter[0]
            node_counter[0] += 1

            new_node = {"node_id": curr_id}
            if "leaf_value" in node:
                new_node["leaf_value"] = float(node["leaf_value"])
            else:
                new_node["best_feature"] = float(node["feature"])
                new_node["split_bin"] = float(node["bin"])
                new_node["default_left"] = 1.0 if node.get("default_left", True) else 0.0
                new_node["left_id"] = node_counter[0]
                walk(node["left"])
                new_node["right_id"] = node_counter[0]
                walk(node["right"])

            node_list.append(new_node)
            return new_node

        walk(tree)

        for node in node_list:
            i = node["node_id"]
            if "leaf_value" in node:
                flat[i, 4] = 1.0
                flat[i, 5] = node["leaf_value"]
            else:
                flat[i, 0] = node["best_feature"]
                flat[i, 1] = node["split_bin"]
                flat[i, 2] = node["left_id"]
                flat[i, 3] = node["right_id"]
                flat[i, 4] = 0.0
                flat[i, 6] = node["default_left"]

        return flat
