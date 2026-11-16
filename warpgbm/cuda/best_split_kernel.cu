#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ops/empty.h>
#include "cuda_utils.h"

__global__ void directional_split_kernel(
    const float *__restrict__ G,                // [E * F * B]
    const float *__restrict__ H,                // [E * F * B]
    int E, int F, int B,
    float min_split_gain,
    float min_child_samples,
    float eps,
    float *__restrict__ per_era_gain,           // [E * F * (B-1)]
    float *__restrict__ per_era_direction       // [E * F * (B-1)]
)
{
    int64_t f = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; // feature index
    int e = blockIdx.y;                            // era index

    if (f >= F || e >= E) return;

    // Access base offset for this (era, feature)
    int64_t base = static_cast<int64_t>(e) * F * B + static_cast<int64_t>(f) * B;
    int64_t base_gain = static_cast<int64_t>(e) * F * (B - 1) + static_cast<int64_t>(f) * (B - 1);

    float G_total = 0.0f, H_total = 0.0f;
    for (int b = 0; b < B; ++b) {
        G_total += G[base + b];
        H_total += H[base + b];
    }

    float G_L = 0.0f, H_L = 0.0f;
    for (int b = 0; b < B - 1; ++b) {
        G_L += G[base + b];
        H_L += H[base + b];

        float G_R = G_total - G_L;
        float H_R = H_total - H_L;

        float gain = 0.0f;
        float dir = 0.0f;

        if (H_L >= min_child_samples && H_R >= min_child_samples) {
            gain = (G_L * G_L) / (H_L + eps)
                 + (G_R * G_R) / (H_R + eps)
                 - (G_total * G_total) / (H_total + eps);

            float left_val = G_L / (H_L + eps);
            float right_val = G_R / (H_R + eps);
            dir = (left_val > right_val) ? 1.0f : -1.0f;
        }

        per_era_gain[base_gain + b] = gain;
        per_era_direction[base_gain + b] = dir;
    }
}

void launch_directional_split_kernel(
    const at::Tensor &G, // [E, F, B]
    const at::Tensor &H, // [E, F, B]
    float min_split_gain,
    float min_child_samples,
    float eps,
    at::Tensor &per_era_gain,       // [E, F, B]
    at::Tensor &per_era_direction,  // [E, F, B]
    int threads = 128)
{
    const auto device = G.device();
    check_cuda_tensor(G, at::kFloat, device, "G");
    check_cuda_tensor(H, at::kFloat, device, "H");
    check_cuda_tensor(per_era_gain, at::kFloat, device, "per_era_gain");
    check_cuda_tensor(per_era_direction, at::kFloat, device, "per_era_direction");
    TORCH_CHECK(G.dim() == 3 && G.sizes() == H.sizes() && G.size(0) > 0 &&
                G.size(2) >= 2 && G.size(2) <= 127, "Split histograms must have matching [eras, features, bins] shapes");
    TORCH_CHECK(per_era_gain.dim() == 3 && per_era_gain.size(0) == G.size(0) &&
                per_era_gain.size(1) == G.size(1) && per_era_gain.size(2) == G.size(2) - 1 &&
                per_era_direction.sizes() == per_era_gain.sizes(), "Split outputs must have shape [eras, features, bins - 1]");
    TORCH_CHECK(std::isfinite(min_split_gain) && min_split_gain >= 0 &&
                std::isfinite(min_child_samples) && min_child_samples > 0 && std::isfinite(eps) && eps >= 0,
                "Split parameters must be finite and nonnegative, with positive min_child_samples");
    TORCH_CHECK(G.size(0) <= std::numeric_limits<int>::max() && G.size(1) <= std::numeric_limits<int>::max(),
                "Split dimensions exceed int32 capacity");
    const c10::cuda::CUDAGuard guard(device);
    const auto *properties = at::cuda::getDeviceProperties(device.index());
    TORCH_CHECK(threads > 0 && threads <= properties->maxThreadsPerBlock && threads % 32 == 0,
                "threads must be a supported positive multiple of 32");
    int E = G.size(0);
    int F = G.size(1);
    int B = G.size(2);

    if (F == 0) return;
    const int64_t feature_blocks = (static_cast<int64_t>(F) + threads - 1) / threads;
    TORCH_CHECK(feature_blocks <= properties->maxGridSize[0] && E <= properties->maxGridSize[1],
                "Split launch exceeds CUDA grid limits");
    dim3 blocks(feature_blocks, E);
    dim3 thread_dims(threads);
    const auto stream = c10::cuda::getCurrentCUDAStream(device.index());

    directional_split_kernel<<<blocks, thread_dims, 0, stream.stream()>>>(
        G.data_ptr<float>(),
        H.data_ptr<float>(),
        E, F, B,
        min_split_gain,
        min_child_samples,
        eps,
        per_era_gain.data_ptr<float>(),
        per_era_direction.data_ptr<float>());

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__device__ __forceinline__ float regularized_leaf_score(
    float g, float h, float l1_reg, float l2_reg)
{
    if (l1_reg > 0.0f)
    {
        float abs_g = fabsf(g);
        if (abs_g <= l1_reg) return 0.0f;
        float tg = copysignf(abs_g - l1_reg, g);
        return (tg * tg) / (h + l2_reg);
    }
    return (g * g) / (h + l2_reg);
}

__global__ void eval_era_splits_v2_kernel(
    const float *__restrict__ G,         // [E, F, B]
    const float *__restrict__ H,         // [E, F, B]
    const float *__restrict__ total_G,   // [E] or nullptr
    const float *__restrict__ total_H,   // [E] or nullptr
    int E, int F, int B,
    float min_child_weight,
    float L1_reg,
    float L2_reg,
    bool has_missing,
    float *__restrict__ cand_gain,       // [E, F, M]
    float *__restrict__ cand_dir)        // [E, F, M]
{
    int64_t f = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int e = blockIdx.y;
    if (f >= F || e >= E) return;

    const int B_minus1 = B - 1;
    const int M = has_missing ? (2 * B_minus1) : B_minus1;
    const int64_t base = static_cast<int64_t>(e) * F * B + f * B;
    const int64_t base_out = static_cast<int64_t>(e) * F * M + f * M;

    float G_valid = 0.0f, H_valid = 0.0f;
    for (int b = 0; b < B; ++b)
    {
        G_valid += G[base + b];
        H_valid += H[base + b];
    }

    float G_tot = has_missing ? total_G[e] : G_valid;
    float H_tot = has_missing ? total_H[e] : H_valid;
    float G_miss = has_missing ? (G_tot - G_valid) : 0.0f;
    float H_miss = has_missing ? fmaxf(0.0f, H_tot - H_valid) : 0.0f;
    float parent_score = regularized_leaf_score(G_tot, H_tot, L1_reg, L2_reg);

    float G_L_base = 0.0f, H_L_base = 0.0f;
    for (int b = 0; b < B_minus1; ++b)
    {
        G_L_base += G[base + b];
        H_L_base += H[base + b];
        float G_R_base = G_valid - G_L_base;
        float H_R_base = H_valid - H_L_base;

        // Candidate 0: missing goes left (default_left = true)
        {
            float G_L = G_L_base + G_miss;
            float H_L = H_L_base + H_miss;
            float G_R = G_R_base;
            float H_R = H_R_base;
            float gain = 0.0f, dir = 0.0f;
            if (H_L >= min_child_weight && H_R >= min_child_weight)
            {
                gain = regularized_leaf_score(G_L, H_L, L1_reg, L2_reg)
                     + regularized_leaf_score(G_R, H_R, L1_reg, L2_reg)
                     - parent_score;
                float left_val = G_L / (H_L + L2_reg);
                float right_val = G_R / (H_R + L2_reg);
                dir = (left_val > right_val) ? 1.0f : -1.0f;
            }
            cand_gain[base_out + b] = gain;
            cand_dir[base_out + b] = dir;
        }

        // Candidate 1: missing goes right (default_left = false)
        if (has_missing)
        {
            float G_L = G_L_base;
            float H_L = H_L_base;
            float G_R = G_R_base + G_miss;
            float H_R = H_R_base + H_miss;
            float gain = 0.0f, dir = 0.0f;
            if (H_L >= min_child_weight && H_R >= min_child_weight)
            {
                gain = regularized_leaf_score(G_L, H_L, L1_reg, L2_reg)
                     + regularized_leaf_score(G_R, H_R, L1_reg, L2_reg)
                     - parent_score;
                float left_val = G_L / (H_L + L2_reg);
                float right_val = G_R / (H_R + L2_reg);
                dir = (left_val > right_val) ? 1.0f : -1.0f;
            }
            cand_gain[base_out + B_minus1 + b] = gain;
            cand_dir[base_out + B_minus1 + b] = dir;
        }
    }
}

__global__ void reduce_era_splits_kernel(
    const float *__restrict__ cand_gain,    // [E, FM]
    const float *__restrict__ cand_dir,     // [E, FM]
    const float *__restrict__ era_weights,  // [E] or nullptr
    const bool *__restrict__ level_mask,    // [F] or nullptr
    int E,
    int FM,
    int M,
    float min_split_gain,
    int era_mode,
    float agreement_threshold,
    float agreement_power,
    float *__restrict__ out_score,          // [FM]
    float *__restrict__ out_agreement,      // [FM]
    float *__restrict__ out_mean_gain)      // [FM]
{
    const int64_t fc = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (fc >= FM) return;

    if (level_mask != nullptr && !level_mask[fc / M])
    {
        out_mean_gain[fc] = -INFINITY;
        out_agreement[fc] = -1.0f;
        out_score[fc] = -INFINITY;
        return;
    }

    float sum_gain = 0.0f;
    float sum_w = 0.0f;
    float dir_sum = 0.0f;
    int active_eras = 0;
    float min_era_gain = 1e30f;

    for (int e = 0; e < E; ++e)
    {
        const int64_t idx = static_cast<int64_t>(e) * FM + fc;
        const float g = cand_gain[idx];
        const float d = cand_dir[idx];
        const float w = (era_weights != nullptr) ? era_weights[e] : 1.0f;
        if (d != 0.0f)
        {
            active_eras += 1;
            dir_sum += d;
            min_era_gain = fminf(min_era_gain, g);
        }
        else
        {
            min_era_gain = fminf(min_era_gain, 0.0f);
        }
        sum_gain += w * g;
        sum_w += w;
    }

    const float mean_gain = (sum_w > 0.0f) ? (sum_gain / sum_w) : 0.0f;
    const float agreement = fabsf(dir_sum) / static_cast<float>(E);
    out_mean_gain[fc] = mean_gain;
    out_agreement[fc] = agreement;

    float score = -INFINITY;
    if (active_eras > 0)
    {
        if (era_mode == 0)
        {
            if (mean_gain > min_split_gain) score = mean_gain;
        }
        else if (era_mode == 1)
        {
            if (mean_gain > min_split_gain && agreement >= agreement_threshold) score = mean_gain;
        }
        else if (era_mode == 2)
        {
            if (mean_gain > min_split_gain) score = mean_gain * powf(agreement, agreement_power);
        }
        else if (era_mode == 3)
        {
            if (min_era_gain > min_split_gain) score = min_era_gain;
        }
        else
        {
            if (mean_gain > min_split_gain) score = mean_gain;
        }
    }
    out_score[fc] = score;
}

__device__ __forceinline__ bool is_better_era_cand(
    float a1, float s1, int fc1,
    float a2, float s2, int fc2,
    int era_mode)
{
    if (s1 == -INFINITY) return false;
    if (s2 == -INFINITY) return true;
    if (era_mode == 0)
    {
        if (a1 != a2) return a1 > a2;
    }
    if (s1 != s2) return s1 > s2;
    return fc1 < fc2;
}

__global__ void argmax_era_splits_stage1_kernel(
    const float *__restrict__ out_score,
    const float *__restrict__ out_agreement,
    int FM,
    int era_mode,
    float *__restrict__ stage_score,
    float *__restrict__ stage_agr,
    int32_t *__restrict__ stage_fc)
{
    const int tid = threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    float best_s = -INFINITY;
    float best_a = -1.0f;
    int32_t best_fc = -1;

    for (int64_t fc = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid; fc < FM; fc += stride)
    {
        const float s = out_score[fc];
        if (s > -INFINITY)
        {
            const float a = out_agreement[fc];
            if (best_fc < 0 || is_better_era_cand(a, s, static_cast<int>(fc), best_a, best_s, best_fc, era_mode))
            {
                best_s = s;
                best_a = a;
                best_fc = static_cast<int32_t>(fc);
            }
        }
    }

    const int lane = tid & 31;
    const int warp = tid >> 5;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        const float os = __shfl_down_sync(0xFFFFFFFF, best_s, offset);
        const float oa = __shfl_down_sync(0xFFFFFFFF, best_a, offset);
        const int32_t ofc = __shfl_down_sync(0xFFFFFFFF, best_fc, offset);
        if (ofc >= 0 && (best_fc < 0 || is_better_era_cand(oa, os, ofc, best_a, best_s, best_fc, era_mode)))
        {
            best_s = os;
            best_a = oa;
            best_fc = ofc;
        }
    }

    __shared__ float sh_s[8];
    __shared__ float sh_a[8];
    __shared__ int32_t sh_fc[8];
    if (lane == 0)
    {
        sh_s[warp] = best_s;
        sh_a[warp] = best_a;
        sh_fc[warp] = best_fc;
    }
    __syncthreads();

    if (warp == 0)
    {
        const int nw = blockDim.x >> 5;
        if (lane < nw)
        {
            best_s = sh_s[lane];
            best_a = sh_a[lane];
            best_fc = sh_fc[lane];
        }
        else
        {
            best_s = -INFINITY;
            best_a = -1.0f;
            best_fc = -1;
        }
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1)
        {
            const float os = __shfl_down_sync(0xFFFFFFFF, best_s, offset);
            const float oa = __shfl_down_sync(0xFFFFFFFF, best_a, offset);
            const int32_t ofc = __shfl_down_sync(0xFFFFFFFF, best_fc, offset);
            if (ofc >= 0 && (best_fc < 0 || is_better_era_cand(oa, os, ofc, best_a, best_s, best_fc, era_mode)))
            {
                best_s = os;
                best_a = oa;
                best_fc = ofc;
            }
        }
        if (lane == 0)
        {
            stage_score[blockIdx.x] = best_s;
            stage_agr[blockIdx.x] = best_a;
            stage_fc[blockIdx.x] = best_fc;
        }
    }
}

__global__ void argmax_era_splits_stage2_kernel(
    const float *__restrict__ stage_score,
    const float *__restrict__ stage_agr,
    const int32_t *__restrict__ stage_fc,
    const float *__restrict__ cand_gain,
    int num_blocks,
    int E,
    int FM,
    int M,
    int split_bins,
    int era_mode,
    float *__restrict__ out_best,       // [5]
    float *__restrict__ out_era_gains)  // [E]
{
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    float best_s = -INFINITY;
    float best_a = -1.0f;
    int32_t best_fc = -1;

    for (int i = tid; i < num_blocks; i += blockDim.x)
    {
        const int32_t fc = stage_fc[i];
        if (fc >= 0)
        {
            const float s = stage_score[i];
            const float a = stage_agr[i];
            if (best_fc < 0 || is_better_era_cand(a, s, fc, best_a, best_s, best_fc, era_mode))
            {
                best_s = s;
                best_a = a;
                best_fc = fc;
            }
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        const float os = __shfl_down_sync(0xFFFFFFFF, best_s, offset);
        const float oa = __shfl_down_sync(0xFFFFFFFF, best_a, offset);
        const int32_t ofc = __shfl_down_sync(0xFFFFFFFF, best_fc, offset);
        if (ofc >= 0 && (best_fc < 0 || is_better_era_cand(oa, os, ofc, best_a, best_s, best_fc, era_mode)))
        {
            best_s = os;
            best_a = oa;
            best_fc = ofc;
        }
    }

    __shared__ float sh_s[8];
    __shared__ float sh_a[8];
    __shared__ int32_t sh_fc[8];
    __shared__ int32_t sh_win_fc;
    if (lane == 0)
    {
        sh_s[warp] = best_s;
        sh_a[warp] = best_a;
        sh_fc[warp] = best_fc;
    }
    __syncthreads();

    if (warp == 0)
    {
        const int nw = blockDim.x >> 5;
        if (lane < nw)
        {
            best_s = sh_s[lane];
            best_a = sh_a[lane];
            best_fc = sh_fc[lane];
        }
        else
        {
            best_s = -INFINITY;
            best_a = -1.0f;
            best_fc = -1;
        }
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1)
        {
            const float os = __shfl_down_sync(0xFFFFFFFF, best_s, offset);
            const float oa = __shfl_down_sync(0xFFFFFFFF, best_a, offset);
            const int32_t ofc = __shfl_down_sync(0xFFFFFFFF, best_fc, offset);
            if (ofc >= 0 && (best_fc < 0 || is_better_era_cand(oa, os, ofc, best_a, best_s, best_fc, era_mode)))
            {
                best_s = os;
                best_a = oa;
                best_fc = ofc;
            }
        }
        if (lane == 0)
        {
            sh_win_fc = best_fc;
            if (best_fc < 0)
            {
                out_best[0] = -1.0f;
                out_best[1] = -1.0f;
                out_best[2] = 1.0f;
                out_best[3] = -1.0f;
                out_best[4] = -INFINITY;
            }
            else
            {
                const int local_feat = best_fc / M;
                const int best_cand = best_fc % M;
                const bool default_left = best_cand < split_bins;
                const int best_bin = default_left ? best_cand : (best_cand - split_bins);
                out_best[0] = static_cast<float>(local_feat);
                out_best[1] = static_cast<float>(best_bin);
                out_best[2] = default_left ? 1.0f : 0.0f;
                out_best[3] = static_cast<float>(best_cand);
                out_best[4] = best_s;
            }
        }
    }
    __syncthreads();

    const int32_t win_fc = sh_win_fc;
    if (win_fc >= 0 && out_era_gains != nullptr)
    {
        for (int e = tid; e < E; e += blockDim.x)
        {
            out_era_gains[e] = cand_gain[static_cast<int64_t>(e) * FM + win_fc];
        }
    }
}

void launch_find_best_split_fused(
    const at::Tensor &G,              // [E, F, B]
    const at::Tensor &H,              // [E, F, B]
    const at::Tensor &total_G,        // [E] or empty
    const at::Tensor &total_H,        // [E] or empty
    const at::Tensor &era_weights,    // [E] or empty
    float min_split_gain,
    float min_child_weight,
    float L1_reg,
    float L2_reg,
    bool has_missing,
    int era_mode,
    float agreement_threshold,
    float agreement_power,
    at::Tensor &cand_gain,            // [E, F, M]
    at::Tensor &cand_dir,             // [E, F, M]
    at::Tensor &out_score,            // [F, M]
    at::Tensor &out_agreement,        // [F, M]
    at::Tensor &out_mean_gain,        // [F, M]
    int threads = 128)
{
    const auto device = G.device();
    check_cuda_tensor(G, at::kFloat, device, "G");
    check_cuda_tensor(H, at::kFloat, device, "H");
    check_cuda_tensor(cand_gain, at::kFloat, device, "cand_gain");
    check_cuda_tensor(cand_dir, at::kFloat, device, "cand_dir");
    check_cuda_tensor(out_score, at::kFloat, device, "out_score");
    check_cuda_tensor(out_agreement, at::kFloat, device, "out_agreement");
    check_cuda_tensor(out_mean_gain, at::kFloat, device, "out_mean_gain");

    const int E = G.size(0);
    const int F = G.size(1);
    const int B = G.size(2);
    const int M = (B - 1) * (has_missing ? 2 : 1);

    TORCH_CHECK(G.dim() == 3 && H.sizes() == G.sizes() && E > 0 && B >= 2 && B <= 127,
                "Fused split histograms must have shape [E, F, B]");
    TORCH_CHECK(cand_gain.sizes() == at::IntArrayRef({E, F, M}) && cand_dir.sizes() == cand_gain.sizes(),
                "cand_gain and cand_dir must have shape [E, F, M]");
    TORCH_CHECK(out_score.sizes() == at::IntArrayRef({F, M}) &&
                out_agreement.sizes() == out_score.sizes() &&
                out_mean_gain.sizes() == out_score.sizes(),
                "Split output tensors must have shape [F, M]");

    const float *tot_g_ptr = nullptr;
    const float *tot_h_ptr = nullptr;
    if (has_missing)
    {
        check_cuda_tensor(total_G, at::kFloat, device, "total_G");
        check_cuda_tensor(total_H, at::kFloat, device, "total_H");
        TORCH_CHECK(total_G.dim() == 1 && total_G.size(0) == E && total_H.sizes() == total_G.sizes(),
                    "total_G and total_H must have shape [E]");
        tot_g_ptr = total_G.data_ptr<float>();
        tot_h_ptr = total_H.data_ptr<float>();
    }

    const float *era_w_ptr = nullptr;
    if (era_weights.numel() > 0)
    {
        check_cuda_tensor(era_weights, at::kFloat, device, "era_weights");
        TORCH_CHECK(era_weights.dim() == 1 && era_weights.size(0) == E, "era_weights must have shape [E]");
        era_w_ptr = era_weights.data_ptr<float>();
    }

    if (F == 0) return;

    const c10::cuda::CUDAGuard guard(device);
    const auto *properties = at::cuda::getDeviceProperties(device.index());
    const auto stream = c10::cuda::getCurrentCUDAStream(device.index());

    const int64_t feature_blocks = (static_cast<int64_t>(F) + threads - 1) / threads;
    TORCH_CHECK(feature_blocks <= properties->maxGridSize[0] && E <= properties->maxGridSize[1],
                "Fused split launch exceeds CUDA grid limits");

    eval_era_splits_v2_kernel<<<dim3(feature_blocks, E), dim3(threads), 0, stream.stream()>>>(
        G.data_ptr<float>(), H.data_ptr<float>(),
        tot_g_ptr, tot_h_ptr,
        E, F, B,
        min_child_weight, L1_reg, L2_reg, has_missing,
        cand_gain.data_ptr<float>(), cand_dir.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int FM = F * M;
    const int red_threads = 128;
    const int red_blocks = (FM + red_threads - 1) / red_threads;
    reduce_era_splits_kernel<<<dim3(red_blocks), dim3(red_threads), 0, stream.stream()>>>(
        cand_gain.data_ptr<float>(), cand_dir.data_ptr<float>(), era_w_ptr, nullptr,
        E, FM, M, min_split_gain, era_mode, agreement_threshold, agreement_power,
        out_score.data_ptr<float>(), out_agreement.data_ptr<float>(), out_mean_gain.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_find_best_split_era_select(
    const at::Tensor &G,              // [E, F, B]
    const at::Tensor &H,              // [E, F, B]
    const at::Tensor &total_G,        // [E] or empty
    const at::Tensor &total_H,        // [E] or empty
    const at::Tensor &era_weights,    // [E] or empty
    const at::Tensor &level_mask,     // [F] bool or empty
    float min_split_gain,
    float min_child_weight,
    float L1_reg,
    float L2_reg,
    bool has_missing,
    int era_mode,
    float agreement_threshold,
    float agreement_power,
    at::Tensor &cand_gain,            // [E, F, M]
    at::Tensor &cand_dir,             // [E, F, M]
    at::Tensor &out_score,            // [F, M]
    at::Tensor &out_agreement,        // [F, M]
    at::Tensor &out_mean_gain,        // [F, M]
    at::Tensor &out_best,             // [5] float32
    at::Tensor &out_era_gains,        // [E] float32
    int threads = 128)
{
    const auto device = G.device();
    check_cuda_tensor(G, at::kFloat, device, "G");
    check_cuda_tensor(H, at::kFloat, device, "H");
    check_cuda_tensor(cand_gain, at::kFloat, device, "cand_gain");
    check_cuda_tensor(cand_dir, at::kFloat, device, "cand_dir");
    check_cuda_tensor(out_score, at::kFloat, device, "out_score");
    check_cuda_tensor(out_agreement, at::kFloat, device, "out_agreement");
    check_cuda_tensor(out_mean_gain, at::kFloat, device, "out_mean_gain");
    check_cuda_tensor(out_best, at::kFloat, device, "out_best");
    check_cuda_tensor(out_era_gains, at::kFloat, device, "out_era_gains");

    const int E = G.size(0);
    const int F = G.size(1);
    const int B = G.size(2);
    const int M = (B - 1) * (has_missing ? 2 : 1);

    TORCH_CHECK(G.dim() == 3 && H.sizes() == G.sizes() && E > 0 && B >= 2 && B <= 127,
                "Era select split histograms must have shape [E, F, B]");
    TORCH_CHECK(cand_gain.sizes() == at::IntArrayRef({E, F, M}) && cand_dir.sizes() == cand_gain.sizes(),
                "cand_gain and cand_dir must have shape [E, F, M]");
    TORCH_CHECK(out_score.sizes() == at::IntArrayRef({F, M}) &&
                out_agreement.sizes() == out_score.sizes() &&
                out_mean_gain.sizes() == out_score.sizes(),
                "Split output tensors must have shape [F, M]");
    TORCH_CHECK(out_best.numel() == 5, "out_best must have 5 float elements");
    TORCH_CHECK(out_era_gains.dim() == 1 && out_era_gains.size(0) == E, "out_era_gains must have shape [E]");

    const float *tot_g_ptr = nullptr;
    const float *tot_h_ptr = nullptr;
    if (has_missing)
    {
        check_cuda_tensor(total_G, at::kFloat, device, "total_G");
        check_cuda_tensor(total_H, at::kFloat, device, "total_H");
        TORCH_CHECK(total_G.dim() == 1 && total_G.size(0) == E && total_H.sizes() == total_G.sizes(),
                    "total_G and total_H must have shape [E]");
        tot_g_ptr = total_G.data_ptr<float>();
        tot_h_ptr = total_H.data_ptr<float>();
    }

    const float *era_w_ptr = nullptr;
    if (era_weights.numel() > 0)
    {
        check_cuda_tensor(era_weights, at::kFloat, device, "era_weights");
        TORCH_CHECK(era_weights.dim() == 1 && era_weights.size(0) == E, "era_weights must have shape [E]");
        era_w_ptr = era_weights.data_ptr<float>();
    }

    const bool *mask_ptr = nullptr;
    if (level_mask.numel() > 0)
    {
        check_cuda_tensor(level_mask, at::kBool, device, "level_mask");
        TORCH_CHECK(level_mask.dim() == 1 && level_mask.size(0) == F, "level_mask must have shape [F]");
        mask_ptr = level_mask.data_ptr<bool>();
    }

    if (F == 0) return;

    const c10::cuda::CUDAGuard guard(device);
    const auto *properties = at::cuda::getDeviceProperties(device.index());
    const auto stream = c10::cuda::getCurrentCUDAStream(device.index());

    const int64_t feature_blocks = (static_cast<int64_t>(F) + threads - 1) / threads;
    TORCH_CHECK(feature_blocks <= properties->maxGridSize[0] && E <= properties->maxGridSize[1],
                "Era select split launch exceeds CUDA grid limits");

    eval_era_splits_v2_kernel<<<dim3(feature_blocks, E), dim3(threads), 0, stream.stream()>>>(
        G.data_ptr<float>(), H.data_ptr<float>(),
        tot_g_ptr, tot_h_ptr,
        E, F, B,
        min_child_weight, L1_reg, L2_reg, has_missing,
        cand_gain.data_ptr<float>(), cand_dir.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int FM = F * M;
    const int red_threads = 128;
    const int red_blocks = (FM + red_threads - 1) / red_threads;
    reduce_era_splits_kernel<<<dim3(red_blocks), dim3(red_threads), 0, stream.stream()>>>(
        cand_gain.data_ptr<float>(), cand_dir.data_ptr<float>(), era_w_ptr, mask_ptr,
        E, FM, M, min_split_gain, era_mode, agreement_threshold, agreement_power,
        out_score.data_ptr<float>(), out_agreement.data_ptr<float>(), out_mean_gain.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int arg_threads = 256;
    const int arg_blocks = std::min<int>(64, std::max<int>(1, (FM + arg_threads - 1) / arg_threads));
    auto stage_s = at::empty({arg_blocks}, out_score.options());
    auto stage_a = at::empty({arg_blocks}, out_score.options());
    auto stage_fc = at::empty({arg_blocks}, out_score.options().dtype(at::kInt));

    argmax_era_splits_stage1_kernel<<<arg_blocks, arg_threads, 0, stream.stream()>>>(
        out_score.data_ptr<float>(), out_agreement.data_ptr<float>(),
        FM, era_mode,
        stage_s.data_ptr<float>(), stage_a.data_ptr<float>(), stage_fc.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    argmax_era_splits_stage2_kernel<<<1, 256, 0, stream.stream()>>>(
        stage_s.data_ptr<float>(), stage_a.data_ptr<float>(), stage_fc.data_ptr<int32_t>(),
        cand_gain.data_ptr<float>(),
        arg_blocks, E, FM, M, B - 1, era_mode,
        out_best.data_ptr<float>(), out_era_gains.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__device__ __forceinline__ float smoothed_leaf_score(
    float g, float h, float l1_reg, float l2_reg, float path_smooth, float parent_val)
{
    if (path_smooth > 0.0f)
    {
        float g_eff = g + path_smooth * parent_val;
        float denom = h + l2_reg + path_smooth;
        if (l1_reg > 0.0f)
        {
            float abs_g = fabsf(g_eff);
            if (abs_g <= l1_reg) return 0.0f;
            float tg = copysignf(abs_g - l1_reg, g_eff);
            return (tg * tg) / denom;
        }
        return (g_eff * g_eff) / denom;
    }
    return regularized_leaf_score(g, h, l1_reg, l2_reg);
}

__global__ void find_best_split_pooled_kernel(
    const float *__restrict__ G,         // [F, B]
    const float *__restrict__ H,         // [F, B]
    const float *__restrict__ miss_G,    // [F] or nullptr
    const float *__restrict__ miss_H,    // [F] or nullptr
    const bool *__restrict__ level_mask, // [F] or nullptr
    int F,
    int B,
    float min_split_gain,
    float min_child_weight,
    float L1_reg,
    float L2_reg,
    float path_smooth,
    float parent_val,
    bool has_missing,
    float *__restrict__ out_best)        // [10]
{
    const int tid = threadIdx.x;
    const int B_minus1 = B - 1;
    const int M = has_missing ? (2 * B_minus1) : B_minus1;

    float best_score = -INFINITY;
    int best_cand = -1;
    int best_f = -1;
    int best_bin = -1;
    bool best_dl = true;
    float best_gl = 0.0f, best_hl = 0.0f, best_gr = 0.0f, best_hr = 0.0f;

    float root_tot_g = 0.0f, root_tot_h = 0.0f;

    const int64_t f_stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int f = static_cast<int>(blockIdx.x) * blockDim.x + tid; f < F; f += static_cast<int>(f_stride))
    {
        const int base = f * B;
        float G_valid = 0.0f, H_valid = 0.0f;
        for (int b = 0; b < B; ++b)
        {
            G_valid += G[base + b];
            H_valid += H[base + b];
        }
        const float Gm = (has_missing && miss_G != nullptr) ? miss_G[f] : 0.0f;
        const float Hm = (has_missing && miss_H != nullptr) ? miss_H[f] : 0.0f;
        const float G_tot = G_valid + Gm;
        const float H_tot = H_valid + Hm;
        if (f == 0)
        {
            root_tot_g = G_tot;
            root_tot_h = H_tot;
        }
        if (level_mask != nullptr && !level_mask[f])
        {
            continue;
        }

        const float parent_score = smoothed_leaf_score(G_tot, H_tot, L1_reg, L2_reg, path_smooth, parent_val);
        float G_L_base = 0.0f, H_L_base = 0.0f;

        for (int b = 0; b < B_minus1; ++b)
        {
            G_L_base += G[base + b];
            H_L_base += H[base + b];
            const float G_R_base = G_valid - G_L_base;
            const float H_R_base = H_valid - H_L_base;

            // Default left
            {
                const float GL = G_L_base + Gm;
                const float HL = H_L_base + Hm;
                const float GR = G_R_base;
                const float HR = H_R_base;
                if (HL >= min_child_weight && HR >= min_child_weight)
                {
                    const float gain = smoothed_leaf_score(GL, HL, L1_reg, L2_reg, path_smooth, parent_val)
                                     + smoothed_leaf_score(GR, HR, L1_reg, L2_reg, path_smooth, parent_val)
                                     - parent_score;
                    if (gain > min_split_gain)
                    {
                        const int cidx = f * M + b;
                        if (gain > best_score || (gain == best_score && (best_cand < 0 || cidx < best_cand)))
                        {
                            best_score = gain;
                            best_cand = cidx;
                            best_f = f;
                            best_bin = b;
                            best_dl = true;
                            best_gl = GL; best_hl = HL;
                            best_gr = GR; best_hr = HR;
                        }
                    }
                }
            }

            // Default right
            if (has_missing)
            {
                const float GL = G_L_base;
                const float HL = H_L_base;
                const float GR = G_R_base + Gm;
                const float HR = H_R_base + Hm;
                if (HL >= min_child_weight && HR >= min_child_weight)
                {
                    const float gain = smoothed_leaf_score(GL, HL, L1_reg, L2_reg, path_smooth, parent_val)
                                     + smoothed_leaf_score(GR, HR, L1_reg, L2_reg, path_smooth, parent_val)
                                     - parent_score;
                    if (gain > min_split_gain)
                    {
                        const int cidx = f * M + B_minus1 + b;
                        if (gain > best_score || (gain == best_score && (best_cand < 0 || cidx < best_cand)))
                        {
                            best_score = gain;
                            best_cand = cidx;
                            best_f = f;
                            best_bin = b;
                            best_dl = false;
                            best_gl = GL; best_hl = HL;
                            best_gr = GR; best_hr = HR;
                        }
                    }
                }
            }
        }
    }

    // Store per-thread winner into shared memory and reduce in 1 warp (blockDim.x <= 256 -> 8 warps)
    __shared__ float sh_score[8];
    __shared__ int sh_cand[8];
    __shared__ int sh_f[8];
    __shared__ int sh_bin[8];
    __shared__ int sh_dl[8];
    __shared__ float sh_gl[8], sh_hl[8], sh_gr[8], sh_hr[8];
    __shared__ float sh_tot_g, sh_tot_h;

    if (tid == 0)
    {
        sh_tot_g = root_tot_g;
        sh_tot_h = root_tot_h;
    }

    const int lane = tid & 31;
    const int warp = tid >> 5;
    int dl_int = best_dl ? 1 : 0;

    for (int offset = 16; offset > 0; offset >>= 1)
    {
        const float o_score = __shfl_down_sync(0xFFFFFFFF, best_score, offset);
        const int o_cand = __shfl_down_sync(0xFFFFFFFF, best_cand, offset);
        const int o_f = __shfl_down_sync(0xFFFFFFFF, best_f, offset);
        const int o_bin = __shfl_down_sync(0xFFFFFFFF, best_bin, offset);
        const int o_dl = __shfl_down_sync(0xFFFFFFFF, dl_int, offset);
        const float o_gl = __shfl_down_sync(0xFFFFFFFF, best_gl, offset);
        const float o_hl = __shfl_down_sync(0xFFFFFFFF, best_hl, offset);
        const float o_gr = __shfl_down_sync(0xFFFFFFFF, best_gr, offset);
        const float o_hr = __shfl_down_sync(0xFFFFFFFF, best_hr, offset);
        if (o_cand >= 0 && (o_score > best_score || (o_score == best_score && (best_cand < 0 || o_cand < best_cand))))
        {
            best_score = o_score; best_cand = o_cand; best_f = o_f; best_bin = o_bin; dl_int = o_dl;
            best_gl = o_gl; best_hl = o_hl; best_gr = o_gr; best_hr = o_hr;
        }
    }

    if (lane == 0)
    {
        sh_score[warp] = best_score;
        sh_cand[warp] = best_cand;
        sh_f[warp] = best_f;
        sh_bin[warp] = best_bin;
        sh_dl[warp] = dl_int;
        sh_gl[warp] = best_gl; sh_hl[warp] = best_hl;
        sh_gr[warp] = best_gr; sh_hr[warp] = best_hr;
    }
    __syncthreads();

    if (warp == 0)
    {
        const int num_warps = blockDim.x >> 5;
        if (lane < num_warps)
        {
            best_score = sh_score[lane];
            best_cand = sh_cand[lane];
            best_f = sh_f[lane];
            best_bin = sh_bin[lane];
            dl_int = sh_dl[lane];
            best_gl = sh_gl[lane]; best_hl = sh_hl[lane];
            best_gr = sh_gr[lane]; best_hr = sh_hr[lane];
        }
        else
        {
            best_score = -INFINITY;
            best_cand = -1; best_f = -1; best_bin = -1; dl_int = 1;
            best_gl = best_hl = best_gr = best_hr = 0.0f;
        }

        for (int offset = 4; offset > 0; offset >>= 1)
        {
            const float o_score = __shfl_down_sync(0xFFFFFFFF, best_score, offset);
            const int o_cand = __shfl_down_sync(0xFFFFFFFF, best_cand, offset);
            const int o_f = __shfl_down_sync(0xFFFFFFFF, best_f, offset);
            const int o_bin = __shfl_down_sync(0xFFFFFFFF, best_bin, offset);
            const int o_dl = __shfl_down_sync(0xFFFFFFFF, dl_int, offset);
            const float o_gl = __shfl_down_sync(0xFFFFFFFF, best_gl, offset);
            const float o_hl = __shfl_down_sync(0xFFFFFFFF, best_hl, offset);
            const float o_gr = __shfl_down_sync(0xFFFFFFFF, best_gr, offset);
            const float o_hr = __shfl_down_sync(0xFFFFFFFF, best_hr, offset);
            if (o_cand >= 0 && (o_score > best_score || (o_score == best_score && (best_cand < 0 || o_cand < best_cand))))
            {
                best_score = o_score; best_cand = o_cand; best_f = o_f; best_bin = o_bin; dl_int = o_dl;
                best_gl = o_gl; best_hl = o_hl; best_gr = o_gr; best_hr = o_hr;
            }
        }

        if (lane == 0)
        {
            float *dst = out_best + static_cast<int64_t>(blockIdx.x) * 10;
            dst[0] = static_cast<float>(best_f);
            dst[1] = static_cast<float>(best_bin);
            dst[2] = static_cast<float>(dl_int);
            dst[3] = best_score;
            dst[4] = best_gl;
            dst[5] = best_hl;
            dst[6] = best_gr;
            dst[7] = best_hr;
            dst[8] = sh_tot_g;
            dst[9] = sh_tot_h;
        }
    }
}

__global__ void reduce_pooled_split_blocks_kernel(
    const float *__restrict__ block_best, // [num_blocks, 10]
    int num_blocks,
    int B_minus1,
    int M,
    float *__restrict__ out_best)         // [10]
{
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    float best_score = -INFINITY;
    int best_cand = -1;
    int best_f = -1, best_bin = -1, dl_int = 1;
    float best_gl = 0.0f, best_hl = 0.0f, best_gr = 0.0f, best_hr = 0.0f;

    if (tid < num_blocks)
    {
        const float *src = block_best + static_cast<int64_t>(tid) * 10;
        const int f = static_cast<int>(src[0]);
        if (f >= 0)
        {
            best_f = f;
            best_bin = static_cast<int>(src[1]);
            dl_int = static_cast<int>(src[2]);
            best_score = src[3];
            best_gl = src[4]; best_hl = src[5];
            best_gr = src[6]; best_hr = src[7];
            best_cand = best_f * M + (dl_int ? best_bin : (B_minus1 + best_bin));
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        const float o_score = __shfl_down_sync(0xFFFFFFFF, best_score, offset);
        const int o_cand = __shfl_down_sync(0xFFFFFFFF, best_cand, offset);
        const int o_f = __shfl_down_sync(0xFFFFFFFF, best_f, offset);
        const int o_bin = __shfl_down_sync(0xFFFFFFFF, best_bin, offset);
        const int o_dl = __shfl_down_sync(0xFFFFFFFF, dl_int, offset);
        const float o_gl = __shfl_down_sync(0xFFFFFFFF, best_gl, offset);
        const float o_hl = __shfl_down_sync(0xFFFFFFFF, best_hl, offset);
        const float o_gr = __shfl_down_sync(0xFFFFFFFF, best_gr, offset);
        const float o_hr = __shfl_down_sync(0xFFFFFFFF, best_hr, offset);
        if (o_cand >= 0 && (o_score > best_score || (o_score == best_score && (best_cand < 0 || o_cand < best_cand))))
        {
            best_score = o_score; best_cand = o_cand; best_f = o_f; best_bin = o_bin; dl_int = o_dl;
            best_gl = o_gl; best_hl = o_hl; best_gr = o_gr; best_hr = o_hr;
        }
    }

    __shared__ float sh_score[4], sh_gl[4], sh_hl[4], sh_gr[4], sh_hr[4];
    __shared__ int sh_cand[4], sh_f[4], sh_bin[4], sh_dl[4];
    if (lane == 0)
    {
        sh_score[warp] = best_score; sh_cand[warp] = best_cand;
        sh_f[warp] = best_f; sh_bin[warp] = best_bin; sh_dl[warp] = dl_int;
        sh_gl[warp] = best_gl; sh_hl[warp] = best_hl;
        sh_gr[warp] = best_gr; sh_hr[warp] = best_hr;
    }
    __syncthreads();

    if (warp == 0)
    {
        if (lane < 4)
        {
            best_score = sh_score[lane]; best_cand = sh_cand[lane];
            best_f = sh_f[lane]; best_bin = sh_bin[lane]; dl_int = sh_dl[lane];
            best_gl = sh_gl[lane]; best_hl = sh_hl[lane];
            best_gr = sh_gr[lane]; best_hr = sh_hr[lane];
        }
        else
        {
            best_score = -INFINITY; best_cand = -1; best_f = -1; best_bin = -1; dl_int = 1;
        }
        for (int offset = 2; offset > 0; offset >>= 1)
        {
            const float o_score = __shfl_down_sync(0xFFFFFFFF, best_score, offset);
            const int o_cand = __shfl_down_sync(0xFFFFFFFF, best_cand, offset);
            const int o_f = __shfl_down_sync(0xFFFFFFFF, best_f, offset);
            const int o_bin = __shfl_down_sync(0xFFFFFFFF, best_bin, offset);
            const int o_dl = __shfl_down_sync(0xFFFFFFFF, dl_int, offset);
            const float o_gl = __shfl_down_sync(0xFFFFFFFF, best_gl, offset);
            const float o_hl = __shfl_down_sync(0xFFFFFFFF, best_hl, offset);
            const float o_gr = __shfl_down_sync(0xFFFFFFFF, best_gr, offset);
            const float o_hr = __shfl_down_sync(0xFFFFFFFF, best_hr, offset);
            if (o_cand >= 0 && (o_score > best_score || (o_score == best_score && (best_cand < 0 || o_cand < best_cand))))
            {
                best_score = o_score; best_cand = o_cand; best_f = o_f; best_bin = o_bin; dl_int = o_dl;
                best_gl = o_gl; best_hl = o_hl; best_gr = o_gr; best_hr = o_hr;
            }
        }
        if (lane == 0)
        {
            out_best[0] = static_cast<float>(best_f);
            out_best[1] = static_cast<float>(best_bin);
            out_best[2] = static_cast<float>(dl_int);
            out_best[3] = best_score;
            out_best[4] = best_gl;
            out_best[5] = best_hl;
            out_best[6] = best_gr;
            out_best[7] = best_hr;
            out_best[8] = block_best[8];
            out_best[9] = block_best[9];
        }
    }
}

void launch_find_best_split_pooled(
    const at::Tensor &G,              // [1, F, B]
    const at::Tensor &H,              // [1, F, B]
    const at::Tensor &miss_G,         // [1, F] or empty
    const at::Tensor &miss_H,         // [1, F] or empty
    const at::Tensor &level_mask,     // [F] bool or empty
    float min_split_gain,
    float min_child_weight,
    float L1_reg,
    float L2_reg,
    float path_smooth,
    float parent_val,
    bool has_missing,
    at::Tensor &out_best)             // [10] float32
{
    const auto device = G.device();
    check_cuda_tensor(G, at::kFloat, device, "G");
    check_cuda_tensor(H, at::kFloat, device, "H");
    check_cuda_tensor(out_best, at::kFloat, device, "out_best");
    TORCH_CHECK(G.dim() == 3 && G.size(0) == 1 && H.sizes() == G.sizes(), "Pooled G/H must have shape [1, F, B]");
    TORCH_CHECK(out_best.numel() == 10, "out_best must have 10 float elements");

    const int F = static_cast<int>(G.size(1));
    const int B = static_cast<int>(G.size(2));
    const float *mg_ptr = nullptr;
    const float *mh_ptr = nullptr;
    if (has_missing && miss_G.numel() > 0)
    {
        check_cuda_tensor(miss_G, at::kFloat, device, "miss_G");
        check_cuda_tensor(miss_H, at::kFloat, device, "miss_H");
        mg_ptr = miss_G.data_ptr<float>();
        mh_ptr = miss_H.data_ptr<float>();
    }
    const bool *mask_ptr = nullptr;
    if (level_mask.numel() > 0)
    {
        check_cuda_tensor(level_mask, at::kBool, device, "level_mask");
        mask_ptr = level_mask.data_ptr<bool>();
    }

    const c10::cuda::CUDAGuard guard(device);
    const auto stream = c10::cuda::getCurrentCUDAStream(device.index());
    if (F <= 512)
    {
        find_best_split_pooled_kernel<<<1, 256, 0, stream.stream()>>>(
            G.data_ptr<float>(), H.data_ptr<float>(), mg_ptr, mh_ptr, mask_ptr,
            F, B, min_split_gain, min_child_weight, L1_reg, L2_reg, path_smooth, parent_val, has_missing,
            out_best.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    else
    {
        const int threads = 128;
        const int num_blocks = std::min<int>(128, (F + threads - 1) / threads);
        auto stage_best = at::empty({num_blocks, 10}, G.options());
        find_best_split_pooled_kernel<<<num_blocks, threads, 0, stream.stream()>>>(
            G.data_ptr<float>(), H.data_ptr<float>(), mg_ptr, mh_ptr, mask_ptr,
            F, B, min_split_gain, min_child_weight, L1_reg, L2_reg, path_smooth, parent_val, has_missing,
            stage_best.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        const int M = (B - 1) * (has_missing ? 2 : 1);
        reduce_pooled_split_blocks_kernel<<<1, 128, 0, stream.stream()>>>(
            stage_best.data_ptr<float>(), num_blocks, B - 1, M, out_best.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}
