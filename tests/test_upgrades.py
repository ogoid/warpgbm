import os
import pickle

import numpy as np
import pytest
import torch

from warpgbm import WarpGBM
from warpgbm.cuda import node_kernel


def _count_leaves(node):
    if "leaf_value" in node:
        return 1
    return _count_leaves(node["left"]) + _count_leaves(node["right"])


def _has_default_left(node):
    if "leaf_value" in node:
        return False
    return ("default_left" in node) or _has_default_left(node["left"]) or _has_default_left(node["right"])


def test_expectile_and_huber_objectives():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(400, 4)).astype(np.float32)
    y_clean = 2.0 * X[:, 0] - X[:, 1]
    y_skew = y_clean + rng.exponential(scale=1.5, size=400).astype(np.float32)

    m_low = WarpGBM(
        objective="expectile",
        expectile_alpha=0.15,
        n_estimators=15,
        max_depth=3,
        num_bins=10,
        random_state=42,
        verbose=0,
    ).fit(X, y_skew, X_eval=X[:80], y_eval=y_skew[:80], eval_every_n_trees=5, eval_metric="expectile")
    m_high = WarpGBM(
        objective="expectile",
        expectile_alpha=0.85,
        n_estimators=15,
        max_depth=3,
        num_bins=10,
        random_state=42,
        verbose=0,
    ).fit(X, y_skew)

    assert m_low.base_prediction < float(np.mean(y_skew)) < m_high.base_prediction
    assert (m_high.predict(X) > m_low.predict(X)).mean() > 0.9
    assert len(m_low.eval_loss) > 0

    y_out = y_clean.copy()
    y_out[:12] += 150.0
    m_mse = WarpGBM(objective="regression", n_estimators=20, max_depth=3, random_state=42, verbose=0).fit(X, y_out)
    m_hub = WarpGBM(
        objective="huber", huber_delta=1.0, n_estimators=20, max_depth=3, random_state=42, verbose=0
    ).fit(X, y_out, X_eval=X[20:100], y_eval=y_clean[20:100], eval_every_n_trees=5, eval_metric="huber")

    err_mse = np.median(np.abs(m_mse.predict(X[12:]) - y_clean[12:]))
    err_hub = np.median(np.abs(m_hub.predict(X[12:]) - y_clean[12:]))
    assert err_hub < err_mse


def test_regularization_and_sampling_options():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(500, 6)).astype(np.float32)
    y = 3.0 * X[:, 0] + rng.normal(scale=0.3, size=500).astype(np.float32)

    m_base = WarpGBM(
        n_estimators=10, max_depth=3, L2_reg=1.0, L1_reg=0.0, path_smooth=0.0, random_state=7, verbose=0
    ).fit(X, y)
    m_reg = WarpGBM(
        n_estimators=10,
        max_depth=3,
        L2_reg=1.0,
        L1_reg=25.0,
        path_smooth=50.0,
        subsample=0.7,
        colsample_bylevel=0.8,
        colsample_bynode=0.8,
        random_state=7,
        verbose=0,
    ).fit(X, y)
    assert np.std(m_reg.predict(X)) < np.std(m_base.predict(X))

    sw = np.ones(500, dtype=np.float32)
    sw[y > 0] = 50.0
    m_w = WarpGBM(n_estimators=5, max_depth=2, random_state=7, verbose=0).fit(X, y, sample_weight=sw)
    assert m_w.base_prediction > m_base.base_prediction


def test_missing_value_routing_nan_and_prebinned():
    rng = np.random.default_rng(11)
    n = 600
    X = rng.normal(size=(n, 3)).astype(np.float32)
    y = np.where(X[:, 0] > 0, 2.0, -2.0).astype(np.float32)

    miss_idx = rng.choice(n, size=120, replace=False)
    X[miss_idx, 0] = np.nan
    y[miss_idx] = 5.0  # force missing values to separate cleanly

    model = WarpGBM(
        n_estimators=15, max_depth=3, num_bins=8, pack_bits=False, random_state=11, verbose=0
    ).fit(X, y)
    assert model.has_missing_ is True
    assert any(_has_default_left(t) for t in model.forest)
    preds = model.predict(X)
    assert np.corrcoef(preds, y)[0, 1] > 0.9

    # Packed 4-bit path with missing values
    m_pack = WarpGBM(
        n_estimators=15, max_depth=3, num_bins=8, pack_bits=True, random_state=11, verbose=0
    ).fit(X, y)
    np.testing.assert_allclose(m_pack.predict(X), preds, rtol=1e-4, atol=1e-4)

    # Prebinned int8 with -1 missing sentinel
    X_int = rng.integers(0, 5, size=(300, 3), dtype=np.int8)
    y_int = np.where(X_int[:, 0] == -1, 4.0, X_int[:, 0].astype(np.float32))
    X_int[:60, 0] = -1
    y_int[:60] = 6.0
    m_pb = WarpGBM(
        n_estimators=10, max_depth=2, num_bins=5, prebinned=True, pack_bits=True, random_state=11, verbose=0
    ).fit(X_int, y_int)
    assert m_pb.has_missing_ is True
    assert m_pb.predict(X_int[:10]).mean() > m_pb.predict(X_int[60:120]).mean()


def test_pack4_parity_odd_n_and_chunked_predict():
    rng = np.random.default_rng(19)
    X = rng.normal(size=(503, 5)).astype(np.float32)  # odd N tests pack4 padding/tail
    y = (X[:, 0] - 0.5 * X[:, 1]).astype(np.float32)

    common = dict(n_estimators=8, max_depth=3, num_bins=10, random_state=19, verbose=0)
    m_unpacked = WarpGBM(**common, pack_bits=False).fit(X, y)
    m_packed = WarpGBM(**common, pack_bits=True).fit(X, y)

    np.testing.assert_allclose(m_packed.predict(X), m_unpacked.predict(X), rtol=1e-4, atol=1e-4)

    # Chunked prediction boundary (> 262_144 rows) with odd remainder chunk (7 rows)
    X_big = rng.normal(size=(262_151, 5)).astype(np.float32)
    preds_big = m_packed.predict(X_big)
    assert preds_big.shape == (262_151,)
    assert np.all(np.isfinite(preds_big))
    np.testing.assert_allclose(preds_big, m_unpacked.predict(X_big), rtol=1e-4, atol=1e-4)

    # Multiclass chunked predict_proba
    y_cls = (y > 0).astype(np.int32)
    m_cls = WarpGBM(objective="binary", **common, pack_bits=True).fit(X[:, :2], y_cls)
    probs_big = m_cls.predict_proba(X_big[:, :2])
    assert probs_big.shape == (262_151, 2)
    np.testing.assert_allclose(probs_big.sum(axis=1), 1.0, atol=1e-5)


def test_prebinned_auto_fallback_and_disable():
    rng = np.random.default_rng(23)
    X = rng.integers(0, 5, size=(400, 3), dtype=np.int32)
    X[350, 1] = 99  # outside head[:256] check, triggers GPU bounds fallback
    y = X[:, 0].astype(np.float32)

    m_auto = WarpGBM(n_estimators=5, num_bins=5, prebinned="auto", random_state=23, verbose=0).fit(X, y)
    assert m_auto.predict(X).shape == (400,)

    m_off = WarpGBM(n_estimators=5, num_bins=5, prebinned=False, random_state=23, verbose=0).fit(X[:, :2], y)
    assert m_off.predict(X[:, :2]).shape == (400,)


def test_leafwise_growth_and_large_depth_clamp():
    rng = np.random.default_rng(31)
    X = rng.normal(size=(600, 4)).astype(np.float32)
    y = (np.sin(X[:, 0]) + X[:, 1] ** 2).astype(np.float32)
    eras = np.repeat(np.arange(3, dtype=np.int32), 200)

    for era_mode in ("pooled", "max_agreement"):
        m = WarpGBM(
            n_estimators=6,
            max_depth=25,  # would OOM without _get_tree_tensors max_leaves clamp
            max_leaves=7,
            era_split_mode=era_mode,
            random_state=31,
            verbose=0,
        ).fit(X, y, era_id=eras)
        assert all(_count_leaves(t) <= 7 for t in m.forest)
        assert np.corrcoef(m.predict(X), y)[0, 1] > 0.7


def test_era_split_modes_and_era_dro():
    rng = np.random.default_rng(37)
    X = rng.normal(size=(600, 4)).astype(np.float32)
    y = (2.0 * X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.2, size=600)).astype(np.float32)
    eras = np.repeat(np.arange(4, dtype=np.int32), 150)

    for mode in ("max_agreement", "threshold", "soft", "min_gain", "pooled"):
        m = WarpGBM(
            n_estimators=5,
            max_depth=3,
            era_split_mode=mode,
            era_agreement_threshold=0.5,
            era_agreement_power=2.0,
            random_state=37,
            verbose=0,
        ).fit(X, y, era_id=eras)
        assert np.corrcoef(m.predict(X), y)[0, 1] > 0.6

    # Group-DRO regression and multiclass
    m_dro = WarpGBM(
        n_estimators=6, max_depth=3, era_dro_gamma=1.5, random_state=37, verbose=0
    ).fit(X, y, era_id=eras)
    m_nodro = WarpGBM(
        n_estimators=6, max_depth=3, era_dro_gamma=0.0, random_state=37, verbose=0
    ).fit(X, y, era_id=eras)
    assert not np.allclose(m_dro.predict(X), m_nodro.predict(X))

    y_cls = (y > 0).astype(np.int32)
    m_cls_dro = WarpGBM(
        objective="binary", n_estimators=5, max_depth=2, era_dro_gamma=1.0, random_state=37, verbose=0
    ).fit(X, y_cls, era_id=eras)
    assert m_cls_dro.predict_proba(X).shape == (600, 2)


def test_restore_best_iteration_and_legacy_load(tmp_path):
    rng = np.random.default_rng(53)
    X_tr = rng.normal(size=(200, 4)).astype(np.float32)
    y_tr = rng.normal(size=200).astype(np.float32)  # pure noise -> eval stops improving
    X_ev = rng.normal(size=(200, 4)).astype(np.float32)
    y_ev = rng.normal(size=200).astype(np.float32)

    m = WarpGBM(
        n_estimators=40,
        max_depth=4,
        learning_rate=0.5,
        min_child_weight=1.0,
        restore_best_iteration=True,
        random_state=53,
        verbose=0,
    ).fit(X_tr, y_tr, X_eval=X_ev, y_eval=y_ev, eval_every_n_trees=1, early_stopping_rounds=3, eval_metric="mse")
    assert m.best_iteration_ >= 1
    assert len(m.forest) == m._trees_trained == m.best_iteration_
    assert m._trees_trained < 40

    # Backward-compatible load_model with old minimal state dict (without v2.3 keys)
    path = tmp_path / "legacy_model.pkl"
    legacy_state = {
        "objective": "regression",
        "num_bins": m.num_bins,
        "max_depth": m.max_depth,
        "learning_rate": m.learning_rate,
        "n_estimators": m.n_estimators,
        "min_child_weight": m.min_child_weight,
        "min_split_gain": m.min_split_gain,
        "L2_reg": m.L2_reg,
        "colsample_bytree": m.colsample_bytree,
        "random_state": m.random_state,
        "warm_start": False,
        "forest": m.forest,
        "bin_edges": m.bin_edges,
        "base_prediction": m.base_prediction,
        "num_features": m.num_features,
        "num_classes": None,
        "classes_": None,
        "label_encoder": None,
        "feature_importance_": m.feature_importance_,
        "per_era_feature_importance_": m.per_era_feature_importance_,
        "_is_fitted": True,
        "_trees_trained": m._trees_trained,
    }
    with open(path, "wb") as f:
        pickle.dump(legacy_state, f)

    loaded = WarpGBM().load_model(str(path))
    np.testing.assert_allclose(loaded.predict(X_ev), m.predict(X_ev), rtol=1e-5, atol=1e-5)


def test_validation_and_cuda_contracts():
    with pytest.raises(ValueError):
        WarpGBM(max_leaves=True)
    with pytest.raises(ValueError):
        WarpGBM(pack_bits=True, num_bins=16)

    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 2)).astype(np.float32)
    y = rng.normal(size=20).astype(np.float32)
    sw_bad = np.ones(20, dtype=np.float32)
    sw_bad[0] = -1.0
    with pytest.raises(ValueError):
        WarpGBM(verbose=0).fit(X, y, sample_weight=sw_bad)

    # CUDA wrapper tensor validation raises ValueError/RuntimeError instead of printing and continuing
    x_cpu = torch.zeros((8, 2), dtype=torch.float32)
    edges_cpu = torch.zeros((2, 3), dtype=torch.float32)
    out_cpu = torch.zeros((8, 2), dtype=torch.int8)
    with pytest.raises((ValueError, RuntimeError)):
        node_kernel.bin_2d(x_cpu, edges_cpu, out_cpu, False)

    fm_wrong_dtype = torch.zeros((2, 8), dtype=torch.int32, device="cuda")
    res_bad = torch.zeros(8, dtype=torch.float64, device="cuda")
    gh = torch.empty(0, dtype=torch.float32, device="cuda")
    idx = torch.arange(8, dtype=torch.int32, device="cuda")
    fidx = torch.arange(2, dtype=torch.int32, device="cuda")
    era = torch.zeros(8, dtype=torch.int32, device="cuda")
    g_h = torch.zeros((1, 2, 4), dtype=torch.float32, device="cuda")
    h_h = torch.zeros((1, 2, 4), dtype=torch.float32, device="cuda")
    mg = torch.empty(0, dtype=torch.float32, device="cuda")
    with pytest.raises((ValueError, RuntimeError)):
        node_kernel.compute_histogram_fm_v2(
            fm_wrong_dtype, res_bad, gh, idx, fidx, era, g_h, h_h, mg, mg, 4, False, False, 64, 4
        )

    out_best = torch.zeros(5, dtype=torch.float32, device="cuda")  # expects size >= 10
    with pytest.raises((ValueError, RuntimeError)):
        node_kernel.find_best_split_pooled(
            g_h, h_h, mg, mg, torch.empty(0, dtype=torch.bool, device="cuda"),
            0.0, 1.0, 0.0, 1.0, 0.0, 0.0, False, out_best
        )


def test_gpu_partition_stability_and_parity():
    rng = np.random.default_rng(101)
    N = 2503  # spans >2 1024-sample partition blocks + odd tail
    F = 4
    X_int = rng.integers(-1, 8, size=(N, F), dtype=np.int8)
    perm = torch.tensor(rng.permutation(N), dtype=torch.int32, device="cuda")

    bin_rm = torch.as_tensor(X_int, device="cuda")
    bin_fm = bin_rm.t().contiguous()
    bin_p4 = torch.empty((F, (N + 1) // 2), dtype=torch.uint8, device="cuda")
    flags = torch.zeros(2, dtype=torch.int32, device="cuda")
    node_kernel.pack_int8_to_4bit_fm(bin_rm, bin_p4, flags, 0, 8)

    for default_left in (True, False):
        mask = torch.empty(N, dtype=torch.bool, device="cuda")
        node_kernel.extract_split_mask(bin_fm, perm, 2, 3, default_left, False, mask)
        ref_left = perm[mask]
        ref_right = perm[~mask]

        for pack4, data in ((False, bin_fm), (True, bin_p4)):
            out = torch.empty_like(perm)
            ws = torch.empty((N + 1023) // 1024 + 2, dtype=torch.int32, device="cuda")
            n_left = node_kernel.partition_samples_fm(data, perm, out, ws, 2, 3, default_left, pack4)
            assert n_left == ref_left.numel()
            torch.testing.assert_close(out[:n_left], ref_left)
            torch.testing.assert_close(out[n_left:], ref_right)


def test_multiblock_pooled_and_era_select_parity():
    torch.manual_seed(7)
    F, B = 700, 8  # F > 512 triggers multi-block pooled split
    G = torch.randn((1, F, B), dtype=torch.float32, device="cuda") * 2.0
    H = torch.rand((1, F, B), dtype=torch.float32, device="cuda") * 5.0 + 1.0
    mg = torch.randn((1, F), dtype=torch.float32, device="cuda")
    mh = torch.rand((1, F), dtype=torch.float32, device="cuda") * 2.0
    lmask = torch.rand(F, device="cuda") > 0.3
    out_mb = torch.empty(10, dtype=torch.float32, device="cuda")
    node_kernel.find_best_split_pooled(
        G, H, mg, mh, lmask, 0.0, 1.0, 0.5, 1.0, 2.0, 0.2, True, out_mb
    )
    b = out_mb.tolist()
    win_f = int(b[0])
    assert 0 <= win_f < F
    assert lmask[win_f].item() is True
    assert b[3] > 0.0

    # Verify find_best_split_era_select matches find_best_split_fused across era_mode
    E, Fe, Be = 12, 48, 6
    Ge = torch.randn((E, Fe, Be), dtype=torch.float32, device="cuda")
    He = torch.rand((E, Fe, Be), dtype=torch.float32, device="cuda") * 4.0 + 2.0
    tG = Ge.sum(dim=2)[:, 0].contiguous() + 0.5
    tH = He.sum(dim=2)[:, 0].contiguous() + 1.0
    lmask_e = torch.rand(Fe, device="cuda") > 0.25
    M = 2 * (Be - 1)
    cg = torch.empty((E, Fe, M), dtype=torch.float32, device="cuda")
    cd = torch.empty((E, Fe, M), dtype=torch.float32, device="cuda")
    os_t = torch.empty((Fe, M), dtype=torch.float32, device="cuda")
    oa_t = torch.empty((Fe, M), dtype=torch.float32, device="cuda")
    om_t = torch.empty((Fe, M), dtype=torch.float32, device="cuda")
    ob_e = torch.empty(5, dtype=torch.float32, device="cuda")
    og_e = torch.empty(E, dtype=torch.float32, device="cuda")
    empty_f = torch.empty(0, dtype=torch.float32, device="cuda")

    for mode in (0, 2):
        node_kernel.find_best_split_fused(
            Ge, He, tG, tH, empty_f, 0.0, 1.0, 0.0, 1.0, True, mode, 0.5, 2.0,
            cg, cd, os_t, oa_t, om_t, 128
        )
        scores = os_t.clone()
        scores[~lmask_e, :] = float("-inf")
        if mode == 0:
            agr = oa_t.clone()
            agr[~lmask_e, :] = -1.0
            valid = (agr == agr.max()) & (scores > float("-inf"))
            ref_idx = int(torch.argmax(torch.where(valid, scores, float("-inf"))).item())
        else:
            ref_idx = int(torch.argmax(scores).item())

        node_kernel.find_best_split_era_select(
            Ge, He, tG, tH, empty_f, lmask_e, 0.0, 1.0, 0.0, 1.0, True, mode, 0.5, 2.0,
            cg, cd, os_t, oa_t, om_t, ob_e, og_e, 128
        )
        res = ob_e.tolist()
        assert int(res[0]) * M + int(res[3]) == ref_idx
        torch.testing.assert_close(og_e, cg[:, int(res[0]), int(res[3])])


def test_era_sorted_large_era_and_int4_predict_parity():
    rng = np.random.default_rng(88)
    N, F, E = 2000, 16, 250  # 2 * 250 * 8 * 4 = 16000 > 12288 -> era_sorted path
    X = rng.normal(size=(N, F)).astype(np.float32)
    X[:100, 0] = np.nan
    eras = np.repeat(np.arange(E, dtype=np.int32), N // E)
    rng.shuffle(eras)
    y = np.where(np.isnan(X[:, 0]), 3.0, 1.5 * X[:, 0] - 0.5 * X[:, 1]).astype(np.float32)

    m_unp = WarpGBM(
        n_estimators=6, max_depth=3, num_bins=7, era_split_mode="soft", pack_bits=False, random_state=88, verbose=0
    ).fit(X, y, era_id=eras)
    m_p4 = WarpGBM(
        n_estimators=6, max_depth=3, num_bins=7, era_split_mode="soft", pack_bits=True, random_state=88, verbose=0
    ).fit(X, y, era_id=eras)
    assert getattr(m_p4, "_use_era_sorted", False) is True
    np.testing.assert_allclose(m_p4.predict(X), m_unp.predict(X), rtol=1e-4, atol=1e-4)

    # Verify packed int4 tree node predictions match legacy 7-float predict_binned
    X_clean = rng.normal(size=(300, F)).astype(np.float32)
    m_clean = WarpGBM(n_estimators=6, max_depth=3, num_bins=8, pack_bits=False, random_state=88, verbose=0).fit(
        X_clean, (X_clean[:, 0] + X_clean[:, 1]).astype(np.float32)
    )
    b_rm = m_clean.bin_inference_data(X_clean)
    b_fm = m_clean._bin_inference_data_fm(X_clean, pack4=False)
    p_rm = m_clean.predict_binned(b_rm).cpu().numpy()
    p_fm = m_clean._predict_binned_fm(b_fm, is_packed=False).cpu().numpy()
    np.testing.assert_allclose(p_fm, p_rm, rtol=1e-5, atol=1e-5)

