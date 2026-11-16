#include <torch/extension.h>
#include <vector>

// Declare the function from histogram_kernel.cu

void launch_directional_split_kernel(
    const at::Tensor &G, // [E, F, B]
    const at::Tensor &H, // [E, F, B]
    float min_split_gain,
    float min_child_samples,
    float eps,
    at::Tensor &per_era_gain,       // [E, F, B]
    at::Tensor &per_era_direction,  // [E, F, B]
    int threads = 128);

void launch_histogram_kernel_cuda_configurable(
    const at::Tensor &bin_indices,
    const at::Tensor &residuals,
    const at::Tensor &sample_indices,
    const at::Tensor &feature_indices,
    const at::Tensor &era_indices,
    at::Tensor &grad_hist,
    at::Tensor &hess_hist,
    int num_bins,
    int threads_per_block = 256,
    int rows_per_thread = 1);

void launch_bin_column_kernel(
    at::Tensor X,
    at::Tensor bin_edges,
    at::Tensor bin_indices);

void predict_with_forest(
    const at::Tensor &bin_indices, // [N x F], int8
    const at::Tensor &tree_tensor, // [T x max_nodes x 6], float32
    float learning_rate,
    at::Tensor &out // [N], float32
);

void launch_histogram_fm_kernel(
    const at::Tensor &bin_indices_fm,
    const at::Tensor &grads,
    const at::Tensor &hessians,
    const at::Tensor &sample_indices,
    const at::Tensor &feature_indices,
    const at::Tensor &era_indices,
    at::Tensor &grad_hist,
    at::Tensor &hess_hist,
    int num_bins,
    int threads_per_block = 256,
    int rows_per_thread = 4);

void launch_find_best_split_fused(
    const at::Tensor &G,
    const at::Tensor &H,
    const at::Tensor &total_G,
    const at::Tensor &total_H,
    const at::Tensor &era_weights,
    float min_split_gain,
    float min_child_weight,
    float L1_reg,
    float L2_reg,
    bool has_missing,
    int era_mode,
    float agreement_threshold,
    float agreement_power,
    at::Tensor &cand_gain,
    at::Tensor &cand_dir,
    at::Tensor &out_score,
    at::Tensor &out_agreement,
    at::Tensor &out_mean_gain,
    int threads = 128);

void launch_bin_2d_kernel(
    const at::Tensor &X,
    const at::Tensor &bin_edges,
    at::Tensor &bin_indices,
    bool feature_major);

void predict_with_forest_fm(
    const at::Tensor &bin_indices_fm,
    const at::Tensor &tree_tensor,
    float learning_rate,
    int64_t out_col_offset,
    int64_t out_stride,
    at::Tensor &out);

void launch_histogram_fm_v2(
    const at::Tensor &bin_data,
    const at::Tensor &grads,
    const at::Tensor &gh_packed,
    const at::Tensor &sample_indices,
    const at::Tensor &feature_indices,
    const at::Tensor &era_indices,
    at::Tensor &grad_hist,
    at::Tensor &hess_hist,
    at::Tensor &miss_grad,
    at::Tensor &miss_hess,
    int num_bins,
    bool pack4,
    bool has_missing,
    int threads_per_block = 128,
    int rows_per_thread = 16);

void launch_extract_split_mask(
    const at::Tensor &bin_data,
    const at::Tensor &sample_indices,
    int64_t feat_idx,
    int32_t split_bin,
    bool default_left,
    bool pack4,
    at::Tensor &out_mask);

void launch_find_best_split_pooled(
    const at::Tensor &G,
    const at::Tensor &H,
    const at::Tensor &miss_G,
    const at::Tensor &miss_H,
    const at::Tensor &level_mask,
    float min_split_gain,
    float min_child_weight,
    float L1_reg,
    float L2_reg,
    float path_smooth,
    float parent_val,
    bool has_missing,
    at::Tensor &out_best);

void launch_pack_int8_to_4bit_fm(
    const at::Tensor &X_int8,
    at::Tensor &out_packed,
    at::Tensor &flags,
    int64_t p_offset,
    int num_bins);

void launch_bin_2d_pack4_fm(
    const at::Tensor &X,
    const at::Tensor &bin_edges,
    at::Tensor &out_packed,
    int64_t p_offset);

void predict_with_forest_fm_packed(
    const at::Tensor &bin_packed_fm,
    int64_t N,
    const at::Tensor &tree_tensor,
    float learning_rate,
    int64_t out_col_offset,
    int64_t out_stride,
    at::Tensor &out);

// Bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("compute_histogram3", &launch_histogram_kernel_cuda_configurable, "Histogram Feature Shared Mem");
    m.def("compute_split", &launch_directional_split_kernel, "Best Split (CUDA)");
    m.def("custom_cuda_binner", &launch_bin_column_kernel, "Custom CUDA binning kernel");
    m.def("predict_forest", &predict_with_forest, "CUDA Predictions");
    m.def("compute_histogram_fm", &launch_histogram_fm_kernel, "Weighted Feature-Major Histogram (CUDA)");
    m.def("compute_histogram_fm_v2", &launch_histogram_fm_v2, "Fast Privatized Feature-Major Histogram (CUDA)");
    m.def("extract_split_mask", &launch_extract_split_mask, "Extract Split Mask (CUDA)");
    m.def("find_best_split_pooled", &launch_find_best_split_pooled, "Fused Pooled Split+Argmax (CUDA)");
    m.def("pack_int8_to_4bit_fm", &launch_pack_int8_to_4bit_fm, "Pack Int8 to 4-bit FM (CUDA)");
    m.def("bin_2d_pack4_fm", &launch_bin_2d_pack4_fm, "Bin Float32 to 4-bit FM (CUDA)");
    m.def("find_best_split_fused", &launch_find_best_split_fused, "Fused Era Split Finder (CUDA)");
    m.def("bin_2d", &launch_bin_2d_kernel, "2D Batch CUDA Binning Kernel");
    m.def("predict_forest_fm", &predict_with_forest_fm, "Feature-Major CUDA Predictions");
    m.def("predict_forest_fm_packed", &predict_with_forest_fm_packed, "4-bit Packed Feature-Major CUDA Predictions");
}
