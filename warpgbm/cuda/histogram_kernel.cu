#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"

__global__ void histogram_tiled_configurable_kernel(
    const int8_t *__restrict__ bin_indices, // [N, F_master]
    const float *__restrict__ residuals,    // [N]
    const int32_t *__restrict__ sample_indices, // [N]
    const int32_t *__restrict__ feature_indices, // [F]
    const int32_t *__restrict__ era_indices, // [N]
    float *__restrict__ grad_hist,          // [F * B]
    float *__restrict__ hess_hist,          // [F * B]
    int64_t N, int64_t F_master, int64_t F, int64_t B, int64_t num_eras,
    int rows_per_thread)
{
    int hist_feat_idx = blockIdx.x;
    int feat = feature_indices[ hist_feat_idx ]; // 1 block per feature
    int64_t row_start = (static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x) * rows_per_thread;

    extern __shared__ float shmem[];
    float *sh_grad = shmem;                             // [num_eras * B]
    float *sh_hess = &sh_grad[num_eras * B];            // [num_eras * B]

    // Initialize shared memory histograms
    for (int i = threadIdx.x; i < num_eras * B; i += blockDim.x) {
        sh_grad[i] = 0.0f;
        sh_hess[i] = 0.0f;
    }
    
    __syncthreads();

    // Each thread processes multiple rows
    for (int r = 0; r < rows_per_thread; ++r)
    {
        int64_t row = row_start + r;
        if (row < N)
        {
            int sample = sample_indices[row];
            int8_t bin = bin_indices[sample * F_master + feat];
            int32_t era = era_indices[sample];
            if (bin >= 0 && bin < B)
            {
                atomicAdd(&sh_grad[era * B + bin], residuals[sample]);
                atomicAdd(&sh_hess[era * B + bin], 1.0f);
            }
        }
    }
    __syncthreads();

    // One thread per bin writes results back to global memory
    for (int b = threadIdx.x; b < num_eras * B; b += blockDim.x)
    {
        int e = b / B;
        int bin = b % B;
        int64_t idx = e * F * B + hist_feat_idx * B + bin;

        atomicAdd(&grad_hist[idx], sh_grad[b]);
        atomicAdd(&hess_hist[idx], sh_hess[b]);
    }

}

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
    int rows_per_thread = 1)
{
    const auto device = bin_indices.device();
    check_cuda_tensor(bin_indices, at::kChar, device, "bin_indices");
    check_cuda_tensor(residuals, at::kFloat, device, "residuals");
    check_cuda_tensor(sample_indices, at::kInt, device, "sample_indices");
    check_cuda_tensor(feature_indices, at::kInt, device, "feature_indices");
    check_cuda_tensor(era_indices, at::kInt, device, "era_indices");
    check_cuda_tensor(grad_hist, at::kFloat, device, "grad_hist");
    check_cuda_tensor(hess_hist, at::kFloat, device, "hess_hist");
    TORCH_CHECK(bin_indices.dim() == 2 && residuals.dim() == 1 &&
                sample_indices.dim() == 1 && feature_indices.dim() == 1 && era_indices.dim() == 1,
                "Histogram inputs have incorrect dimensions");
    TORCH_CHECK(residuals.size(0) == bin_indices.size(0) && era_indices.size(0) == bin_indices.size(0),
                "residuals and era_indices must match the row count");
    TORCH_CHECK(bin_indices.size(0) <= std::numeric_limits<int32_t>::max() &&
                bin_indices.size(1) <= std::numeric_limits<int32_t>::max(),
                "Histogram dimensions exceed int32 index capacity");
    TORCH_CHECK(num_bins >= 2 && num_bins <= 127, "num_bins must be in [2, 127]");
    TORCH_CHECK(grad_hist.dim() == 3 && grad_hist.size(0) > 0 &&
                grad_hist.size(1) == feature_indices.size(0) && grad_hist.size(2) == num_bins &&
                hess_hist.sizes() == grad_hist.sizes(), "Histogram output shapes must be [eras, features, bins]");
    const c10::cuda::CUDAGuard guard(device);
    const auto *properties = at::cuda::getDeviceProperties(device.index());
    TORCH_CHECK(threads_per_block > 0 && threads_per_block <= properties->maxThreadsPerBlock &&
                threads_per_block % 32 == 0, "threads_per_block must be a supported positive multiple of 32");
    TORCH_CHECK(rows_per_thread >= 1 && rows_per_thread <= 16, "rows_per_thread must be in [1, 16]");

    int64_t N = sample_indices.size(0);
    int64_t F = feature_indices.size(0);
    int num_features_master = bin_indices.size(1);

    int64_t rows_per_block = threads_per_block * rows_per_thread;
    int64_t row_tiles = (N + rows_per_block - 1) / rows_per_block;

    if (N == 0 || F == 0) return;
    TORCH_CHECK(F <= properties->maxGridSize[0] && row_tiles <= properties->maxGridSize[1],
                "Histogram launch exceeds CUDA grid limits");
    const int64_t num_eras = grad_hist.size(0);
    const int64_t shared_mem_bytes = 2 * num_eras * num_bins * sizeof(float);
    const size_t shared_mem_limit = std::max(properties->sharedMemPerBlock, properties->sharedMemPerBlockOptin);
    TORCH_CHECK(shared_mem_bytes <= shared_mem_limit,
                "Histogram needs ", shared_mem_bytes, " shared-memory bytes; device limit is ", shared_mem_limit);
    if (shared_mem_bytes > properties->sharedMemPerBlock) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(histogram_tiled_configurable_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_mem_limit)));
    }
    dim3 blocks(F, row_tiles);
    dim3 threads(threads_per_block);
    const auto stream = c10::cuda::getCurrentCUDAStream(device.index());

    histogram_tiled_configurable_kernel<<<blocks, threads, shared_mem_bytes, stream.stream()>>>(
        bin_indices.data_ptr<int8_t>(),
        residuals.data_ptr<float>(),
        sample_indices.data_ptr<int32_t>(),
        feature_indices.data_ptr<int32_t>(),
        era_indices.data_ptr<int32_t>(),
        grad_hist.data_ptr<float>(),
        hess_hist.data_ptr<float>(),
        N, num_features_master, F, num_bins, num_eras,
        rows_per_thread);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool SINGLE_ERA, bool HAS_HESS>
__global__ void histogram_fm_shared_kernel(
    const int8_t *__restrict__ bin_indices_fm, // [F_master, N_total]
    const float *__restrict__ grads,           // [N_total]
    const float *__restrict__ hessians,        // [N_total] or nullptr
    const int32_t *__restrict__ sample_indices,// [N_node]
    const int32_t *__restrict__ feature_indices,// [F]
    const int32_t *__restrict__ era_indices,   // [N_total]
    float *__restrict__ grad_hist,             // [E * F * B]
    float *__restrict__ hess_hist,             // [E * F * B]
    int64_t N_node,
    int64_t N_total,
    int64_t F,
    int64_t B,
    int64_t num_eras,
    int rows_per_thread)
{
    int hist_feat_idx = blockIdx.x;
    int feat = feature_indices[hist_feat_idx];
    const int8_t *feat_bins = bin_indices_fm + static_cast<int64_t>(feat) * N_total;
    int64_t row_start = (static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x) * rows_per_thread;

    extern __shared__ float shmem[];
    const int64_t hist_elems = num_eras * B;
    float *sh_grad = shmem;
    float *sh_hess = &sh_grad[hist_elems];

    for (int64_t i = threadIdx.x; i < hist_elems; i += blockDim.x)
    {
        sh_grad[i] = 0.0f;
        sh_hess[i] = 0.0f;
    }
    __syncthreads();

    for (int r = 0; r < rows_per_thread; ++r)
    {
        int64_t row = row_start + r;
        if (row < N_node)
        {
            int32_t sample = sample_indices[row];
            int8_t bin = feat_bins[sample];
            if (bin >= 0 && bin < B)
            {
                int32_t era = SINGLE_ERA ? 0 : era_indices[sample];
                int64_t sh_idx = static_cast<int64_t>(era) * B + bin;
                atomicAdd(&sh_grad[sh_idx], grads[sample]);
                atomicAdd(&sh_hess[sh_idx], HAS_HESS ? hessians[sample] : 1.0f);
            }
        }
    }
    __syncthreads();

    for (int64_t b = threadIdx.x; b < hist_elems; b += blockDim.x)
    {
        float g_val = sh_grad[b];
        float h_val = sh_hess[b];
        if (h_val != 0.0f || g_val != 0.0f)
        {
            int64_t e = b / B;
            int64_t bin = b % B;
            int64_t idx = e * F * B + static_cast<int64_t>(hist_feat_idx) * B + bin;
            atomicAdd(&grad_hist[idx], g_val);
            atomicAdd(&hess_hist[idx], h_val);
        }
    }
}

template <bool SINGLE_ERA, bool HAS_HESS>
__global__ void histogram_fm_global_kernel(
    const int8_t *__restrict__ bin_indices_fm, // [F_master, N_total]
    const float *__restrict__ grads,           // [N_total]
    const float *__restrict__ hessians,        // [N_total] or nullptr
    const int32_t *__restrict__ sample_indices,// [N_node]
    const int32_t *__restrict__ feature_indices,// [F]
    const int32_t *__restrict__ era_indices,   // [N_total]
    float *__restrict__ grad_hist,             // [E * F * B]
    float *__restrict__ hess_hist,             // [E * F * B]
    int64_t N_node,
    int64_t N_total,
    int64_t F,
    int64_t B,
    int rows_per_thread)
{
    int hist_feat_idx = blockIdx.x;
    int feat = feature_indices[hist_feat_idx];
    const int8_t *feat_bins = bin_indices_fm + static_cast<int64_t>(feat) * N_total;
    int64_t row_start = (static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x) * rows_per_thread;

    for (int r = 0; r < rows_per_thread; ++r)
    {
        int64_t row = row_start + r;
        if (row < N_node)
        {
            int32_t sample = sample_indices[row];
            int8_t bin = feat_bins[sample];
            if (bin >= 0 && bin < B)
            {
                int32_t era = SINGLE_ERA ? 0 : era_indices[sample];
                int64_t idx = static_cast<int64_t>(era) * F * B + static_cast<int64_t>(hist_feat_idx) * B + bin;
                atomicAdd(&grad_hist[idx], grads[sample]);
                atomicAdd(&hess_hist[idx], HAS_HESS ? hessians[sample] : 1.0f);
            }
        }
    }
}

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
    int rows_per_thread = 4)
{
    const auto device = bin_indices_fm.device();
    check_cuda_tensor(bin_indices_fm, at::kChar, device, "bin_indices_fm");
    check_cuda_tensor(grads, at::kFloat, device, "grads");
    check_cuda_tensor(sample_indices, at::kInt, device, "sample_indices");
    check_cuda_tensor(feature_indices, at::kInt, device, "feature_indices");
    check_cuda_tensor(era_indices, at::kInt, device, "era_indices");
    check_cuda_tensor(grad_hist, at::kFloat, device, "grad_hist");
    check_cuda_tensor(hess_hist, at::kFloat, device, "hess_hist");

    const bool has_hess = hessians.numel() > 0;
    if (has_hess)
    {
        check_cuda_tensor(hessians, at::kFloat, device, "hessians");
        TORCH_CHECK(hessians.dim() == 1 && hessians.size(0) == grads.size(0),
                    "hessians must be 1D matching grads length");
    }

    TORCH_CHECK(bin_indices_fm.dim() == 2 && grads.dim() == 1 &&
                sample_indices.dim() == 1 && feature_indices.dim() == 1 && era_indices.dim() == 1,
                "Histogram FM inputs have incorrect dimensions");
    const int64_t F_master = bin_indices_fm.size(0);
    const int64_t N_total = bin_indices_fm.size(1);
    TORCH_CHECK(grads.size(0) == N_total && era_indices.size(0) == N_total,
                "grads and era_indices must match bin_indices_fm sample count");
    TORCH_CHECK(num_bins >= 2 && num_bins <= 127, "num_bins must be in [2, 127]");
    TORCH_CHECK(grad_hist.dim() == 3 && grad_hist.size(0) > 0 &&
                grad_hist.size(1) == feature_indices.size(0) && grad_hist.size(2) == num_bins &&
                hess_hist.sizes() == grad_hist.sizes(),
                "Histogram output shapes must be [eras, features, bins]");

    const c10::cuda::CUDAGuard guard(device);
    const auto *properties = at::cuda::getDeviceProperties(device.index());
    TORCH_CHECK(threads_per_block > 0 && threads_per_block <= properties->maxThreadsPerBlock &&
                threads_per_block % 32 == 0, "threads_per_block must be a supported positive multiple of 32");
    TORCH_CHECK(rows_per_thread >= 1 && rows_per_thread <= 16, "rows_per_thread must be in [1, 16]");

    const int64_t N_node = sample_indices.size(0);
    const int64_t F = feature_indices.size(0);
    if (N_node == 0 || F == 0) return;
    TORCH_CHECK(F_master <= std::numeric_limits<int32_t>::max() && N_total <= std::numeric_limits<int32_t>::max(),
                "Histogram FM dimensions exceed int32 capacity");

    const int64_t rows_per_block = static_cast<int64_t>(threads_per_block) * rows_per_thread;
    const int64_t row_tiles = (N_node + rows_per_block - 1) / rows_per_block;
    TORCH_CHECK(F <= properties->maxGridSize[0] && row_tiles <= properties->maxGridSize[1],
                "Histogram FM launch exceeds CUDA grid limits");

    const int64_t num_eras = grad_hist.size(0);
    const int64_t shared_mem_bytes = 2 * num_eras * num_bins * sizeof(float);
    const size_t shared_mem_limit = std::max(properties->sharedMemPerBlock, properties->sharedMemPerBlockOptin);

    dim3 blocks(F, row_tiles);
    dim3 threads(threads_per_block);
    const auto stream = c10::cuda::getCurrentCUDAStream(device.index());
    const float *hess_ptr = has_hess ? hessians.data_ptr<float>() : nullptr;

    if (static_cast<size_t>(shared_mem_bytes) <= shared_mem_limit)
    {
        if (shared_mem_bytes > properties->sharedMemPerBlock)
        {
            if (num_eras == 1 && !has_hess) {
                C10_CUDA_CHECK(cudaFuncSetAttribute(histogram_fm_shared_kernel<true, false>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_mem_limit)));
            } else if (num_eras == 1 && has_hess) {
                C10_CUDA_CHECK(cudaFuncSetAttribute(histogram_fm_shared_kernel<true, true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_mem_limit)));
            } else if (num_eras > 1 && !has_hess) {
                C10_CUDA_CHECK(cudaFuncSetAttribute(histogram_fm_shared_kernel<false, false>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_mem_limit)));
            } else {
                C10_CUDA_CHECK(cudaFuncSetAttribute(histogram_fm_shared_kernel<false, true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_mem_limit)));
            }
        }
        if (num_eras == 1 && !has_hess) {
            histogram_fm_shared_kernel<true, false><<<blocks, threads, shared_mem_bytes, stream.stream()>>>(
                bin_indices_fm.data_ptr<int8_t>(), grads.data_ptr<float>(), hess_ptr,
                sample_indices.data_ptr<int32_t>(), feature_indices.data_ptr<int32_t>(),
                era_indices.data_ptr<int32_t>(), grad_hist.data_ptr<float>(), hess_hist.data_ptr<float>(),
                N_node, N_total, F, num_bins, num_eras, rows_per_thread);
        } else if (num_eras == 1 && has_hess) {
            histogram_fm_shared_kernel<true, true><<<blocks, threads, shared_mem_bytes, stream.stream()>>>(
                bin_indices_fm.data_ptr<int8_t>(), grads.data_ptr<float>(), hess_ptr,
                sample_indices.data_ptr<int32_t>(), feature_indices.data_ptr<int32_t>(),
                era_indices.data_ptr<int32_t>(), grad_hist.data_ptr<float>(), hess_hist.data_ptr<float>(),
                N_node, N_total, F, num_bins, num_eras, rows_per_thread);
        } else if (num_eras > 1 && !has_hess) {
            histogram_fm_shared_kernel<false, false><<<blocks, threads, shared_mem_bytes, stream.stream()>>>(
                bin_indices_fm.data_ptr<int8_t>(), grads.data_ptr<float>(), hess_ptr,
                sample_indices.data_ptr<int32_t>(), feature_indices.data_ptr<int32_t>(),
                era_indices.data_ptr<int32_t>(), grad_hist.data_ptr<float>(), hess_hist.data_ptr<float>(),
                N_node, N_total, F, num_bins, num_eras, rows_per_thread);
        } else {
            histogram_fm_shared_kernel<false, true><<<blocks, threads, shared_mem_bytes, stream.stream()>>>(
                bin_indices_fm.data_ptr<int8_t>(), grads.data_ptr<float>(), hess_ptr,
                sample_indices.data_ptr<int32_t>(), feature_indices.data_ptr<int32_t>(),
                era_indices.data_ptr<int32_t>(), grad_hist.data_ptr<float>(), hess_hist.data_ptr<float>(),
                N_node, N_total, F, num_bins, num_eras, rows_per_thread);
        }
    }
    else
    {
        if (num_eras == 1 && !has_hess) {
            histogram_fm_global_kernel<true, false><<<blocks, threads, 0, stream.stream()>>>(
                bin_indices_fm.data_ptr<int8_t>(), grads.data_ptr<float>(), hess_ptr,
                sample_indices.data_ptr<int32_t>(), feature_indices.data_ptr<int32_t>(),
                era_indices.data_ptr<int32_t>(), grad_hist.data_ptr<float>(), hess_hist.data_ptr<float>(),
                N_node, N_total, F, num_bins, rows_per_thread);
        } else if (num_eras == 1 && has_hess) {
            histogram_fm_global_kernel<true, true><<<blocks, threads, 0, stream.stream()>>>(
                bin_indices_fm.data_ptr<int8_t>(), grads.data_ptr<float>(), hess_ptr,
                sample_indices.data_ptr<int32_t>(), feature_indices.data_ptr<int32_t>(),
                era_indices.data_ptr<int32_t>(), grad_hist.data_ptr<float>(), hess_hist.data_ptr<float>(),
                N_node, N_total, F, num_bins, rows_per_thread);
        } else if (num_eras > 1 && !has_hess) {
            histogram_fm_global_kernel<false, false><<<blocks, threads, 0, stream.stream()>>>(
                bin_indices_fm.data_ptr<int8_t>(), grads.data_ptr<float>(), hess_ptr,
                sample_indices.data_ptr<int32_t>(), feature_indices.data_ptr<int32_t>(),
                era_indices.data_ptr<int32_t>(), grad_hist.data_ptr<float>(), hess_hist.data_ptr<float>(),
                N_node, N_total, F, num_bins, rows_per_thread);
        } else {
            histogram_fm_global_kernel<false, true><<<blocks, threads, 0, stream.stream()>>>(
                bin_indices_fm.data_ptr<int8_t>(), grads.data_ptr<float>(), hess_ptr,
                sample_indices.data_ptr<int32_t>(), feature_indices.data_ptr<int32_t>(),
                era_indices.data_ptr<int32_t>(), grad_hist.data_ptr<float>(), hess_hist.data_ptr<float>(),
                N_node, N_total, F, num_bins, rows_per_thread);
        }
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool SINGLE_ERA, bool HAS_HESS, bool PACK4, bool HAS_MISSING, bool LOCK_FREE>
__global__ void histogram_fm_v2_kernel(
    const uint8_t *__restrict__ bin_data,      // [F_master, N_total] int8 or [F_master, N_packed] uint8
    const float *__restrict__ grads,           // [N_total] when !HAS_HESS
    const float2 *__restrict__ gh_packed,      // [N_total] when HAS_HESS
    const int32_t *__restrict__ sample_indices,// [N_node]
    const int32_t *__restrict__ feature_indices,// [F]
    const int32_t *__restrict__ era_indices,   // [N_total]
    float *__restrict__ grad_hist,             // [E, F, B]
    float *__restrict__ hess_hist,             // [E, F, B]
    float *__restrict__ miss_grad,             // [E, F] or nullptr
    float *__restrict__ miss_hess,             // [E, F] or nullptr
    int64_t N_node,
    int64_t N_stride,
    int64_t F,
    int64_t B,
    int64_t num_eras,
    int rows_per_thread)
{
    const int hist_feat_idx = blockIdx.x;
    const int feat = feature_indices[hist_feat_idx];
    const uint8_t *feat_bins = bin_data + static_cast<int64_t>(feat) * N_stride;
    const int64_t block_base = static_cast<int64_t>(blockIdx.y) * blockDim.x * rows_per_thread;
    const int B_store = HAS_MISSING ? static_cast<int>(B + 1) : static_cast<int>(B);
    const int64_t hist_elems = num_eras * B_store;

    extern __shared__ float shmem[];
    const int64_t sh_stride = LOCK_FREE ? blockDim.x : 1;
    const int64_t sh_count = hist_elems * sh_stride;
    float *sh_grad = shmem;
    float *sh_hess = &sh_grad[sh_count];

    for (int64_t i = threadIdx.x; i < sh_count; i += blockDim.x)
    {
        sh_grad[i] = 0.0f;
        sh_hess[i] = 0.0f;
    }
    __syncthreads();

    for (int r = 0; r < rows_per_thread; ++r)
    {
        const int64_t row = block_base + threadIdx.x + static_cast<int64_t>(r) * blockDim.x;
        if (row < N_node)
        {
            const int32_t sample = sample_indices[row];
            int32_t bin;
            if (PACK4)
            {
                const uint8_t bv = feat_bins[sample >> 1];
                const int32_t nib = (sample & 1) ? (bv >> 4) : (bv & 0x0F);
                bin = nib - 1;
            }
            else
            {
                bin = static_cast<int32_t>(reinterpret_cast<const int8_t *>(feat_bins)[sample]);
            }

            const int32_t b_slot = (bin >= 0 && bin < B) ? bin : (HAS_MISSING ? static_cast<int32_t>(B) : -1);
            if (b_slot >= 0)
            {
                float g, h;
                if (HAS_HESS)
                {
                    const float2 gh = gh_packed[sample];
                    g = gh.x;
                    h = gh.y;
                }
                else
                {
                    g = grads[sample];
                    h = 1.0f;
                }
                const int32_t era = SINGLE_ERA ? 0 : era_indices[sample];
                const int64_t hist_idx = static_cast<int64_t>(era) * B_store + b_slot;
                if (LOCK_FREE)
                {
                    const int64_t slot = hist_idx * blockDim.x + threadIdx.x;
                    sh_grad[slot] += g;
                    sh_hess[slot] += h;
                }
                else
                {
                    atomicAdd(&sh_grad[hist_idx], g);
                    atomicAdd(&sh_hess[hist_idx], h);
                }
            }
        }
    }
    __syncthreads();

    if (LOCK_FREE)
    {
        const int warp_id = threadIdx.x >> 5;
        const int lane = threadIdx.x & 31;
        const int num_warps = blockDim.x >> 5;

        for (int64_t b = warp_id; b < hist_elems; b += num_warps)
        {
            float sum_g = 0.0f;
            float sum_h = 0.0f;
            const int64_t base_b = b * blockDim.x;
            for (int t = lane; t < blockDim.x; t += 32)
            {
                sum_g += sh_grad[base_b + t];
                sum_h += sh_hess[base_b + t];
            }
            for (int offset = 16; offset > 0; offset >>= 1)
            {
                sum_g += __shfl_down_sync(0xFFFFFFFF, sum_g, offset);
                sum_h += __shfl_down_sync(0xFFFFFFFF, sum_h, offset);
            }
            if (lane == 0 && (sum_h != 0.0f || sum_g != 0.0f))
            {
                const int64_t e = b / B_store;
                const int64_t bin_s = b % B_store;
                if (HAS_MISSING && bin_s == B)
                {
                    const int64_t m_idx = e * F + hist_feat_idx;
                    atomicAdd(&miss_grad[m_idx], sum_g);
                    atomicAdd(&miss_hess[m_idx], sum_h);
                }
                else
                {
                    const int64_t g_idx = e * F * B + static_cast<int64_t>(hist_feat_idx) * B + bin_s;
                    atomicAdd(&grad_hist[g_idx], sum_g);
                    atomicAdd(&hess_hist[g_idx], sum_h);
                }
            }
        }
    }
    else
    {
        for (int64_t b = threadIdx.x; b < hist_elems; b += blockDim.x)
        {
            const float g_val = sh_grad[b];
            const float h_val = sh_hess[b];
            if (h_val != 0.0f || g_val != 0.0f)
            {
                const int64_t e = b / B_store;
                const int64_t bin_s = b % B_store;
                if (HAS_MISSING && bin_s == B)
                {
                    const int64_t m_idx = e * F + hist_feat_idx;
                    atomicAdd(&miss_grad[m_idx], g_val);
                    atomicAdd(&miss_hess[m_idx], h_val);
                }
                else
                {
                    const int64_t g_idx = e * F * B + static_cast<int64_t>(hist_feat_idx) * B + bin_s;
                    atomicAdd(&grad_hist[g_idx], g_val);
                    atomicAdd(&hess_hist[g_idx], h_val);
                }
            }
        }
    }
}

template <bool SINGLE_ERA, bool HAS_HESS, bool PACK4, bool HAS_MISSING>
__global__ void histogram_fm_v2_global_kernel(
    const uint8_t *__restrict__ bin_data,
    const float *__restrict__ grads,
    const float2 *__restrict__ gh_packed,
    const int32_t *__restrict__ sample_indices,
    const int32_t *__restrict__ feature_indices,
    const int32_t *__restrict__ era_indices,
    float *__restrict__ grad_hist,
    float *__restrict__ hess_hist,
    float *__restrict__ miss_grad,
    float *__restrict__ miss_hess,
    int64_t N_node,
    int64_t N_stride,
    int64_t F,
    int64_t B,
    int rows_per_thread)
{
    const int hist_feat_idx = blockIdx.x;
    const int feat = feature_indices[hist_feat_idx];
    const uint8_t *feat_bins = bin_data + static_cast<int64_t>(feat) * N_stride;
    const int64_t block_base = static_cast<int64_t>(blockIdx.y) * blockDim.x * rows_per_thread;

    for (int r = 0; r < rows_per_thread; ++r)
    {
        const int64_t row = block_base + threadIdx.x + static_cast<int64_t>(r) * blockDim.x;
        if (row < N_node)
        {
            const int32_t sample = sample_indices[row];
            int32_t bin;
            if (PACK4)
            {
                const uint8_t bv = feat_bins[sample >> 1];
                const int32_t nib = (sample & 1) ? (bv >> 4) : (bv & 0x0F);
                bin = nib - 1;
            }
            else
            {
                bin = static_cast<int32_t>(reinterpret_cast<const int8_t *>(feat_bins)[sample]);
            }

            const bool is_valid = (bin >= 0 && bin < B);
            if (is_valid || HAS_MISSING)
            {
                float g, h;
                if (HAS_HESS)
                {
                    const float2 gh = gh_packed[sample];
                    g = gh.x;
                    h = gh.y;
                }
                else
                {
                    g = grads[sample];
                    h = 1.0f;
                }
                const int32_t era = SINGLE_ERA ? 0 : era_indices[sample];
                if (is_valid)
                {
                    const int64_t g_idx = static_cast<int64_t>(era) * F * B + static_cast<int64_t>(hist_feat_idx) * B + bin;
                    atomicAdd(&grad_hist[g_idx], g);
                    atomicAdd(&hess_hist[g_idx], h);
                }
                else if (HAS_MISSING)
                {
                    const int64_t m_idx = static_cast<int64_t>(era) * F + hist_feat_idx;
                    atomicAdd(&miss_grad[m_idx], g);
                    atomicAdd(&miss_hess[m_idx], h);
                }
            }
        }
    }
}

template <bool SINGLE_ERA, bool HAS_HESS, bool PACK4, bool HAS_MISSING>
static void dispatch_histogram_fm_v2(
    const uint8_t *bin_ptr,
    const float *grads_ptr,
    const float2 *gh_ptr,
    const int32_t *samples_ptr,
    const int32_t *feats_ptr,
    const int32_t *eras_ptr,
    float *grad_hist_ptr,
    float *hess_hist_ptr,
    float *miss_grad_ptr,
    float *miss_hess_ptr,
    int64_t N_node,
    int64_t N_stride,
    int64_t F,
    int64_t B,
    int64_t num_eras,
    dim3 blocks,
    dim3 threads,
    int rows_per_thread,
    size_t lock_free_bytes,
    size_t shared_bytes,
    const cudaDeviceProp *properties,
    cudaStream_t stream)
{
    if (lock_free_bytes <= static_cast<size_t>(properties->sharedMemPerBlock))
    {
        histogram_fm_v2_kernel<SINGLE_ERA, HAS_HESS, PACK4, HAS_MISSING, true>
            <<<blocks, threads, lock_free_bytes, stream>>>(
                bin_ptr, grads_ptr, gh_ptr, samples_ptr, feats_ptr, eras_ptr,
                grad_hist_ptr, hess_hist_ptr, miss_grad_ptr, miss_hess_ptr,
                N_node, N_stride, F, B, num_eras, rows_per_thread);
    }
    else
    {
        const size_t shared_limit = std::max(properties->sharedMemPerBlock, properties->sharedMemPerBlockOptin);
        if (shared_bytes <= shared_limit)
        {
            if (shared_bytes > static_cast<size_t>(properties->sharedMemPerBlock))
            {
                C10_CUDA_CHECK(cudaFuncSetAttribute(
                    histogram_fm_v2_kernel<SINGLE_ERA, HAS_HESS, PACK4, HAS_MISSING, false>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(shared_limit)));
            }
            histogram_fm_v2_kernel<SINGLE_ERA, HAS_HESS, PACK4, HAS_MISSING, false>
                <<<blocks, threads, shared_bytes, stream>>>(
                    bin_ptr, grads_ptr, gh_ptr, samples_ptr, feats_ptr, eras_ptr,
                    grad_hist_ptr, hess_hist_ptr, miss_grad_ptr, miss_hess_ptr,
                    N_node, N_stride, F, B, num_eras, rows_per_thread);
        }
        else
        {
            histogram_fm_v2_global_kernel<SINGLE_ERA, HAS_HESS, PACK4, HAS_MISSING>
                <<<blocks, threads, 0, stream>>>(
                    bin_ptr, grads_ptr, gh_ptr, samples_ptr, feats_ptr, eras_ptr,
                    grad_hist_ptr, hess_hist_ptr, miss_grad_ptr, miss_hess_ptr,
                    N_node, N_stride, F, B, rows_per_thread);
        }
    }
}

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
    int rows_per_thread = 16)
{
    const auto device = bin_data.device();
    check_cuda_tensor(bin_data, pack4 ? at::kByte : at::kChar, device, "bin_data");
    check_cuda_tensor(sample_indices, at::kInt, device, "sample_indices");
    check_cuda_tensor(feature_indices, at::kInt, device, "feature_indices");
    check_cuda_tensor(era_indices, at::kInt, device, "era_indices");
    check_cuda_tensor(grad_hist, at::kFloat, device, "grad_hist");
    check_cuda_tensor(hess_hist, at::kFloat, device, "hess_hist");

    const bool has_hess = gh_packed.numel() > 0;
    if (has_hess)
    {
        check_cuda_tensor(gh_packed, at::kFloat, device, "gh_packed");
        TORCH_CHECK(gh_packed.dim() == 2 && gh_packed.size(1) == 2 &&
                    gh_packed.size(0) == era_indices.size(0),
                    "gh_packed must have shape [N_total, 2]");
    }
    else
    {
        check_cuda_tensor(grads, at::kFloat, device, "grads");
        TORCH_CHECK(grads.dim() == 1 && grads.size(0) == era_indices.size(0),
                    "grads must be 1D matching era_indices");
    }

    const int64_t N_node = sample_indices.size(0);
    const int64_t F = feature_indices.size(0);
    if (N_node == 0 || F == 0) return;

    const int64_t N_stride = bin_data.size(1);
    const int64_t num_eras = grad_hist.size(0);
    if (has_missing)
    {
        check_cuda_tensor(miss_grad, at::kFloat, device, "miss_grad");
        check_cuda_tensor(miss_hess, at::kFloat, device, "miss_hess");
        TORCH_CHECK(miss_grad.sizes() == at::IntArrayRef({num_eras, F}) &&
                    miss_hess.sizes() == miss_grad.sizes(),
                    "miss_grad/miss_hess must have shape [E, F]");
    }

    const c10::cuda::CUDAGuard guard(device);
    const auto *properties = at::cuda::getDeviceProperties(device.index());
    const auto stream = c10::cuda::getCurrentCUDAStream(device.index());

    // Adapt tile size so small nodes keep enough blocks and large nodes minimize global atomics
    int eff_threads = threads_per_block;
    int eff_rpt = rows_per_thread;
    while (eff_rpt > 1 && ((N_node + eff_threads * eff_rpt - 1) / (eff_threads * eff_rpt)) * F < 128)
    {
        eff_rpt >>= 1;
    }
    if (eff_threads < 32) eff_threads = 64;

    const int64_t rows_per_block = static_cast<int64_t>(eff_threads) * eff_rpt;
    const int64_t row_tiles = (N_node + rows_per_block - 1) / rows_per_block;
    dim3 blocks(F, row_tiles);
    dim3 threads(eff_threads);

    const int64_t B_store = has_missing ? (num_bins + 1) : num_bins;
    const int64_t hist_elems = num_eras * B_store;
    const size_t lock_free_bytes = static_cast<size_t>(2) * hist_elems * eff_threads * sizeof(float);
    const size_t shared_bytes = static_cast<size_t>(2) * hist_elems * sizeof(float);

    const uint8_t *bin_ptr = pack4
        ? bin_data.data_ptr<uint8_t>()
        : reinterpret_cast<const uint8_t *>(bin_data.data_ptr<int8_t>());
    const float *g_ptr = has_hess ? nullptr : grads.data_ptr<float>();
    const float2 *gh_ptr = has_hess ? reinterpret_cast<const float2 *>(gh_packed.data_ptr<float>()) : nullptr;
    const int32_t *s_ptr = sample_indices.data_ptr<int32_t>();
    const int32_t *f_ptr = feature_indices.data_ptr<int32_t>();
    const int32_t *e_ptr = era_indices.data_ptr<int32_t>();
    float *hist_g_ptr = grad_hist.data_ptr<float>();
    float *hist_h_ptr = hess_hist.data_ptr<float>();
    float *mg_ptr = has_missing ? miss_grad.data_ptr<float>() : nullptr;
    float *mh_ptr = has_missing ? miss_hess.data_ptr<float>() : nullptr;

    const int flags = (num_eras == 1 ? 8 : 0) | (has_hess ? 4 : 0) | (pack4 ? 2 : 0) | (has_missing ? 1 : 0);
    switch (flags)
    {
        case 0:  dispatch_histogram_fm_v2<false, false, false, false>(bin_ptr, g_ptr, gh_ptr, s_ptr, f_ptr, e_ptr, hist_g_ptr, hist_h_ptr, mg_ptr, mh_ptr, N_node, N_stride, F, num_bins, num_eras, blocks, threads, eff_rpt, lock_free_bytes, shared_bytes, properties, stream.stream()); break;
        case 1:  dispatch_histogram_fm_v2<false, false, false, true >(bin_ptr, g_ptr, gh_ptr, s_ptr, f_ptr, e_ptr, hist_g_ptr, hist_h_ptr, mg_ptr, mh_ptr, N_node, N_stride, F, num_bins, num_eras, blocks, threads, eff_rpt, lock_free_bytes, shared_bytes, properties, stream.stream()); break;
        case 2:  dispatch_histogram_fm_v2<false, false, true,  false>(bin_ptr, g_ptr, gh_ptr, s_ptr, f_ptr, e_ptr, hist_g_ptr, hist_h_ptr, mg_ptr, mh_ptr, N_node, N_stride, F, num_bins, num_eras, blocks, threads, eff_rpt, lock_free_bytes, shared_bytes, properties, stream.stream()); break;
        case 3:  dispatch_histogram_fm_v2<false, false, true,  true >(bin_ptr, g_ptr, gh_ptr, s_ptr, f_ptr, e_ptr, hist_g_ptr, hist_h_ptr, mg_ptr, mh_ptr, N_node, N_stride, F, num_bins, num_eras, blocks, threads, eff_rpt, lock_free_bytes, shared_bytes, properties, stream.stream()); break;
        case 4:  dispatch_histogram_fm_v2<false, true,  false, false>(bin_ptr, g_ptr, gh_ptr, s_ptr, f_ptr, e_ptr, hist_g_ptr, hist_h_ptr, mg_ptr, mh_ptr, N_node, N_stride, F, num_bins, num_eras, blocks, threads, eff_rpt, lock_free_bytes, shared_bytes, properties, stream.stream()); break;
        case 5:  dispatch_histogram_fm_v2<false, true,  false, true >(bin_ptr, g_ptr, gh_ptr, s_ptr, f_ptr, e_ptr, hist_g_ptr, hist_h_ptr, mg_ptr, mh_ptr, N_node, N_stride, F, num_bins, num_eras, blocks, threads, eff_rpt, lock_free_bytes, shared_bytes, properties, stream.stream()); break;
        case 6:  dispatch_histogram_fm_v2<false, true,  true,  false>(bin_ptr, g_ptr, gh_ptr, s_ptr, f_ptr, e_ptr, hist_g_ptr, hist_h_ptr, mg_ptr, mh_ptr, N_node, N_stride, F, num_bins, num_eras, blocks, threads, eff_rpt, lock_free_bytes, shared_bytes, properties, stream.stream()); break;
        case 7:  dispatch_histogram_fm_v2<false, true,  true,  true >(bin_ptr, g_ptr, gh_ptr, s_ptr, f_ptr, e_ptr, hist_g_ptr, hist_h_ptr, mg_ptr, mh_ptr, N_node, N_stride, F, num_bins, num_eras, blocks, threads, eff_rpt, lock_free_bytes, shared_bytes, properties, stream.stream()); break;
        case 8:  dispatch_histogram_fm_v2<true,  false, false, false>(bin_ptr, g_ptr, gh_ptr, s_ptr, f_ptr, e_ptr, hist_g_ptr, hist_h_ptr, mg_ptr, mh_ptr, N_node, N_stride, F, num_bins, num_eras, blocks, threads, eff_rpt, lock_free_bytes, shared_bytes, properties, stream.stream()); break;
        case 9:  dispatch_histogram_fm_v2<true,  false, false, true >(bin_ptr, g_ptr, gh_ptr, s_ptr, f_ptr, e_ptr, hist_g_ptr, hist_h_ptr, mg_ptr, mh_ptr, N_node, N_stride, F, num_bins, num_eras, blocks, threads, eff_rpt, lock_free_bytes, shared_bytes, properties, stream.stream()); break;
        case 10: dispatch_histogram_fm_v2<true,  false, true,  false>(bin_ptr, g_ptr, gh_ptr, s_ptr, f_ptr, e_ptr, hist_g_ptr, hist_h_ptr, mg_ptr, mh_ptr, N_node, N_stride, F, num_bins, num_eras, blocks, threads, eff_rpt, lock_free_bytes, shared_bytes, properties, stream.stream()); break;
        case 11: dispatch_histogram_fm_v2<true,  false, true,  true >(bin_ptr, g_ptr, gh_ptr, s_ptr, f_ptr, e_ptr, hist_g_ptr, hist_h_ptr, mg_ptr, mh_ptr, N_node, N_stride, F, num_bins, num_eras, blocks, threads, eff_rpt, lock_free_bytes, shared_bytes, properties, stream.stream()); break;
        case 12: dispatch_histogram_fm_v2<true,  true,  false, false>(bin_ptr, g_ptr, gh_ptr, s_ptr, f_ptr, e_ptr, hist_g_ptr, hist_h_ptr, mg_ptr, mh_ptr, N_node, N_stride, F, num_bins, num_eras, blocks, threads, eff_rpt, lock_free_bytes, shared_bytes, properties, stream.stream()); break;
        case 13: dispatch_histogram_fm_v2<true,  true,  false, true >(bin_ptr, g_ptr, gh_ptr, s_ptr, f_ptr, e_ptr, hist_g_ptr, hist_h_ptr, mg_ptr, mh_ptr, N_node, N_stride, F, num_bins, num_eras, blocks, threads, eff_rpt, lock_free_bytes, shared_bytes, properties, stream.stream()); break;
        case 14: dispatch_histogram_fm_v2<true,  true,  true,  false>(bin_ptr, g_ptr, gh_ptr, s_ptr, f_ptr, e_ptr, hist_g_ptr, hist_h_ptr, mg_ptr, mh_ptr, N_node, N_stride, F, num_bins, num_eras, blocks, threads, eff_rpt, lock_free_bytes, shared_bytes, properties, stream.stream()); break;
        case 15: dispatch_histogram_fm_v2<true,  true,  true,  true >(bin_ptr, g_ptr, gh_ptr, s_ptr, f_ptr, e_ptr, hist_g_ptr, hist_h_ptr, mg_ptr, mh_ptr, N_node, N_stride, F, num_bins, num_eras, blocks, threads, eff_rpt, lock_free_bytes, shared_bytes, properties, stream.stream()); break;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool PACK4>
__global__ void extract_split_mask_kernel(
    const uint8_t *__restrict__ feat_bins,
    const int32_t *__restrict__ sample_indices,
    int64_t N_node,
    int32_t split_bin,
    bool default_left,
    bool *__restrict__ out_mask)
{
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= N_node) return;
    const int32_t sample = sample_indices[row];
    int32_t bin;
    if (PACK4)
    {
        const uint8_t bv = feat_bins[sample >> 1];
        const int32_t nib = (sample & 1) ? (bv >> 4) : (bv & 0x0F);
        bin = nib - 1;
    }
    else
    {
        bin = static_cast<int32_t>(reinterpret_cast<const int8_t *>(feat_bins)[sample]);
    }
    out_mask[row] = (bin < 0) ? default_left : (bin <= split_bin);
}

void launch_extract_split_mask(
    const at::Tensor &bin_data,
    const at::Tensor &sample_indices,
    int64_t feat_idx,
    int32_t split_bin,
    bool default_left,
    bool pack4,
    at::Tensor &out_mask)
{
    const auto device = bin_data.device();
    check_cuda_tensor(bin_data, pack4 ? at::kByte : at::kChar, device, "bin_data");
    check_cuda_tensor(sample_indices, at::kInt, device, "sample_indices");
    check_cuda_tensor(out_mask, at::kBool, device, "out_mask");
    TORCH_CHECK(bin_data.dim() == 2 && sample_indices.dim() == 1 && out_mask.dim() == 1,
                "extract_split_mask inputs have incorrect dimensions");
    TORCH_CHECK(feat_idx >= 0 && feat_idx < bin_data.size(0), "feat_idx out of bounds");
    const int64_t N_node = sample_indices.size(0);
    TORCH_CHECK(out_mask.size(0) >= N_node, "out_mask too small for sample_indices");
    if (N_node == 0) return;
    const int64_t N_stride = bin_data.size(1);
    const c10::cuda::CUDAGuard guard(device);
    const auto stream = c10::cuda::getCurrentCUDAStream(device.index());
    const int threads = 256;
    const int64_t blocks = (N_node + threads - 1) / threads;
    const uint8_t *feat_ptr = (pack4
        ? bin_data.data_ptr<uint8_t>()
        : reinterpret_cast<const uint8_t *>(bin_data.data_ptr<int8_t>())) + feat_idx * N_stride;
    if (pack4)
    {
        extract_split_mask_kernel<true><<<blocks, threads, 0, stream.stream()>>>(
            feat_ptr, sample_indices.data_ptr<int32_t>(), N_node, split_bin, default_left, out_mask.data_ptr<bool>());
    }
    else
    {
        extract_split_mask_kernel<false><<<blocks, threads, 0, stream.stream()>>>(
            feat_ptr, sample_indices.data_ptr<int32_t>(), N_node, split_bin, default_left, out_mask.data_ptr<bool>());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
