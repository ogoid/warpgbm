#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"

__global__ void predict_forest_kernel(
    const int8_t *__restrict__ bin_indices, // [N x F]
    const float *__restrict__ tree_tensor,  // [T x max_nodes x 6]
    int64_t N, int64_t F, int64_t T, int64_t max_nodes,
    float learning_rate,
    float *__restrict__ out // [N]
)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total_jobs = N * T;
    if (idx >= total_jobs)
        return;

    int64_t i = idx % N; // sample index
    int64_t t = idx / N; // tree index

    const float *tree = tree_tensor + t * max_nodes * 6;

    int node_id = 0;
    while (true)
    {
        float is_leaf = tree[node_id * 6 + 4];
        if (is_leaf > 0.5f)
        {
            float val = tree[node_id * 6 + 5];
            atomicAdd(&out[i], learning_rate * val);
            return;
        }

        int feat = static_cast<int>(tree[node_id * 6 + 0]);
        int split_bin = static_cast<int>(tree[node_id * 6 + 1]);
        int left_id = static_cast<int>(tree[node_id * 6 + 2]);
        int right_id = static_cast<int>(tree[node_id * 6 + 3]);

        // prevent overflow
        int64_t bin_idx = i * F + feat;
        int8_t bin = bin_indices[bin_idx];

        node_id = (bin <= split_bin) ? left_id : right_id;
    }
}


void predict_with_forest(
    const at::Tensor &bin_indices,
    const at::Tensor &tree_tensor,
    float learning_rate,
    at::Tensor &out
)
{
    const auto device = bin_indices.device();
    check_cuda_tensor(bin_indices, at::kChar, device, "bin_indices");
    check_cuda_tensor(tree_tensor, at::kFloat, device, "tree_tensor");
    check_cuda_tensor(out, at::kFloat, device, "out");
    TORCH_CHECK(bin_indices.dim() == 2 && tree_tensor.dim() == 3 && tree_tensor.size(1) > 0 &&
                tree_tensor.size(2) == 6, "Prediction inputs must have shapes [rows, features] and [trees, nodes, 6]");
    TORCH_CHECK(out.dim() == 1 && out.size(0) == bin_indices.size(0), "out must have one value per row");
    TORCH_CHECK(std::isfinite(learning_rate), "learning_rate must be finite");
    const c10::cuda::CUDAGuard guard(device);
    int64_t N = bin_indices.size(0);
    int64_t F = bin_indices.size(1);
    int64_t T = tree_tensor.size(0);
    int64_t max_nodes = tree_tensor.size(1);

    if (N == 0 || T == 0) return;
    TORCH_CHECK(N <= std::numeric_limits<int64_t>::max() / T, "Prediction job count overflows int64");
    int64_t total_jobs = N * T;
    int threads_per_block = 256;
    int64_t blocks = (total_jobs - 1) / threads_per_block + 1;
    const auto *properties = at::cuda::getDeviceProperties(device.index());
    TORCH_CHECK(blocks <= properties->maxGridSize[0], "Prediction launch exceeds CUDA grid limits");
    const auto stream = c10::cuda::getCurrentCUDAStream(device.index());

    predict_forest_kernel<<<blocks, threads_per_block, 0, stream.stream()>>>(
        bin_indices.data_ptr<int8_t>(),
        tree_tensor.data_ptr<float>(),
        N, F, T, max_nodes,
        learning_rate,
        out.data_ptr<float>());

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool USE_ATOMIC, bool PACKED_NODES>
__global__ void predict_forest_fm_kernel(
    const int8_t *__restrict__ bin_indices_fm, // [F, N]
    const void *__restrict__ tree_tensor,      // [T, max_nodes, 7] float32 or [T, max_nodes, 4] int32
    int64_t N, int64_t F, int64_t T, int64_t max_nodes,
    float learning_rate,
    int64_t out_col_offset,
    int64_t out_stride,
    float *__restrict__ out)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int64_t chunk_idx = blockIdx.y;
    int64_t num_chunks = gridDim.y;
    int64_t trees_per_chunk = (T + num_chunks - 1) / num_chunks;
    int64_t t_start = chunk_idx * trees_per_chunk;
    int64_t t_end = min(T, t_start + trees_per_chunk);

    float acc = 0.0f;

    if (PACKED_NODES)
    {
        const int4 *trees_i4 = reinterpret_cast<const int4 *>(tree_tensor);
        for (int64_t t = t_start; t < t_end; ++t)
        {
            const int4 *tree = trees_i4 + t * max_nodes;
            int node_id = 0;
            while (true)
            {
                const int4 nd = tree[node_id];
                if (nd.x < 0)
                {
                    acc += __int_as_float(nd.y);
                    break;
                }
                const int feat = nd.x;
                const int split_bin = nd.y & 0xFF;
                const int8_t bin = bin_indices_fm[static_cast<int64_t>(feat) * N + i];
                const bool go_left = (bin < 0) ? ((nd.y & 0x100) != 0) : (bin <= split_bin);
                node_id = go_left ? nd.z : nd.w;
            }
        }
    }
    else
    {
        const float *trees_f = reinterpret_cast<const float *>(tree_tensor);
        const int64_t tree_stride = max_nodes * 7;
        for (int64_t t = t_start; t < t_end; ++t)
        {
            const float *tree = trees_f + t * tree_stride;
            int node_id = 0;
            while (true)
            {
                const float *node = tree + node_id * 7;
                if (node[4] > 0.5f)
                {
                    acc += node[5];
                    break;
                }
                int feat = static_cast<int>(node[0]);
                int split_bin = static_cast<int>(node[1]);
                int left_id = static_cast<int>(node[2]);
                int right_id = static_cast<int>(node[3]);
                int8_t bin = bin_indices_fm[static_cast<int64_t>(feat) * N + i];
                bool go_left = (bin < 0) ? (node[6] > 0.5f) : (bin <= split_bin);
                node_id = go_left ? left_id : right_id;
            }
        }
    }

    int64_t out_idx = i * out_stride + out_col_offset;
    if (USE_ATOMIC)
    {
        atomicAdd(&out[out_idx], learning_rate * acc);
    }
    else
    {
        out[out_idx] += learning_rate * acc;
    }
}

void predict_with_forest_fm(
    const at::Tensor &bin_indices_fm, // [F, N] int8
    const at::Tensor &tree_tensor,    // [T, max_nodes, 7] float32 or [T, max_nodes, 4] int32
    float learning_rate,
    int64_t out_col_offset,
    int64_t out_stride,
    at::Tensor &out)
{
    const auto device = bin_indices_fm.device();
    check_cuda_tensor(bin_indices_fm, at::kChar, device, "bin_indices_fm");
    const bool packed_nodes = (tree_tensor.scalar_type() == at::kInt);
    check_cuda_tensor(tree_tensor, packed_nodes ? at::kInt : at::kFloat, device, "tree_tensor");
    check_cuda_tensor(out, at::kFloat, device, "out");
    TORCH_CHECK(bin_indices_fm.dim() == 2 && tree_tensor.dim() == 3 &&
                tree_tensor.size(1) > 0 && tree_tensor.size(2) == (packed_nodes ? 4 : 7),
                "FM prediction inputs must have shapes [features, rows] and [trees, nodes, 7 or int4]");
    TORCH_CHECK(out_stride >= 1 && out_col_offset >= 0 && out_col_offset < out_stride,
                "Invalid out_col_offset / out_stride");
    const int64_t F = bin_indices_fm.size(0);
    const int64_t N = bin_indices_fm.size(1);
    const int64_t T = tree_tensor.size(0);
    const int64_t max_nodes = tree_tensor.size(1);
    TORCH_CHECK(out.numel() == N * out_stride, "out size must equal rows * out_stride");
    TORCH_CHECK(std::isfinite(learning_rate), "learning_rate must be finite");

    if (N == 0 || T == 0) return;

    const c10::cuda::CUDAGuard guard(device);
    const auto *properties = at::cuda::getDeviceProperties(device.index());
    const auto stream = c10::cuda::getCurrentCUDAStream(device.index());

    const int threads_per_block = 256;
    const int64_t blocks_x = (N + threads_per_block - 1) / threads_per_block;
    const int64_t blocks_y = (N < 32768 && T >= 64) ? std::min<int64_t>(16, (T + 15) / 16) : 1;
    TORCH_CHECK(blocks_x <= properties->maxGridSize[0] && blocks_y <= properties->maxGridSize[1],
                "FM prediction launch exceeds CUDA grid limits");

    dim3 blocks(blocks_x, blocks_y);
    const void *t_ptr = packed_nodes
        ? static_cast<const void *>(tree_tensor.data_ptr<int32_t>())
        : static_cast<const void *>(tree_tensor.data_ptr<float>());

    if (blocks_y > 1 && packed_nodes) {
        predict_forest_fm_kernel<true, true><<<blocks, threads_per_block, 0, stream.stream()>>>(
            bin_indices_fm.data_ptr<int8_t>(), t_ptr, N, F, T, max_nodes, learning_rate, out_col_offset, out_stride, out.data_ptr<float>());
    } else if (blocks_y > 1) {
        predict_forest_fm_kernel<true, false><<<blocks, threads_per_block, 0, stream.stream()>>>(
            bin_indices_fm.data_ptr<int8_t>(), t_ptr, N, F, T, max_nodes, learning_rate, out_col_offset, out_stride, out.data_ptr<float>());
    } else if (packed_nodes) {
        predict_forest_fm_kernel<false, true><<<blocks, threads_per_block, 0, stream.stream()>>>(
            bin_indices_fm.data_ptr<int8_t>(), t_ptr, N, F, T, max_nodes, learning_rate, out_col_offset, out_stride, out.data_ptr<float>());
    } else {
        predict_forest_fm_kernel<false, false><<<blocks, threads_per_block, 0, stream.stream()>>>(
            bin_indices_fm.data_ptr<int8_t>(), t_ptr, N, F, T, max_nodes, learning_rate, out_col_offset, out_stride, out.data_ptr<float>());
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool USE_ATOMIC, bool PACKED_NODES>
__global__ void predict_forest_fm_packed_kernel(
    const uint8_t *__restrict__ bin_packed_fm, // [F, N_packed]
    const void *__restrict__ tree_tensor,      // [T, max_nodes, 7] float32 or [T, max_nodes, 4] int32
    int64_t N, int64_t N_packed, int64_t F, int64_t T, int64_t max_nodes,
    float learning_rate,
    int64_t out_col_offset,
    int64_t out_stride,
    float *__restrict__ out)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int64_t chunk_idx = blockIdx.y;
    int64_t num_chunks = gridDim.y;
    int64_t trees_per_chunk = (T + num_chunks - 1) / num_chunks;
    int64_t t_start = chunk_idx * trees_per_chunk;
    int64_t t_end = min(T, t_start + trees_per_chunk);

    float acc = 0.0f;

    if (PACKED_NODES)
    {
        const int4 *trees_i4 = reinterpret_cast<const int4 *>(tree_tensor);
        for (int64_t t = t_start; t < t_end; ++t)
        {
            const int4 *tree = trees_i4 + t * max_nodes;
            int node_id = 0;
            while (true)
            {
                const int4 nd = tree[node_id];
                if (nd.x < 0)
                {
                    acc += __int_as_float(nd.y);
                    break;
                }
                const int feat = nd.x;
                const int split_bin = nd.y & 0xFF;
                const uint8_t bv = bin_packed_fm[static_cast<int64_t>(feat) * N_packed + (i >> 1)];
                const int32_t nib = (i & 1) ? (bv >> 4) : (bv & 0x0F);
                const int8_t bin = static_cast<int8_t>(nib - 1);
                const bool go_left = (bin < 0) ? ((nd.y & 0x100) != 0) : (bin <= split_bin);
                node_id = go_left ? nd.z : nd.w;
            }
        }
    }
    else
    {
        const float *trees_f = reinterpret_cast<const float *>(tree_tensor);
        const int64_t tree_stride = max_nodes * 7;
        for (int64_t t = t_start; t < t_end; ++t)
        {
            const float *tree = trees_f + t * tree_stride;
            int node_id = 0;
            while (true)
            {
                const float *node = tree + node_id * 7;
                if (node[4] > 0.5f)
                {
                    acc += node[5];
                    break;
                }
                int feat = static_cast<int>(node[0]);
                int split_bin = static_cast<int>(node[1]);
                int left_id = static_cast<int>(node[2]);
                int right_id = static_cast<int>(node[3]);
                const uint8_t bv = bin_packed_fm[static_cast<int64_t>(feat) * N_packed + (i >> 1)];
                const int32_t nib = (i & 1) ? (bv >> 4) : (bv & 0x0F);
                const int8_t bin = static_cast<int8_t>(nib - 1);
                bool go_left = (bin < 0) ? (node[6] > 0.5f) : (bin <= split_bin);
                node_id = go_left ? left_id : right_id;
            }
        }
    }

    int64_t out_idx = i * out_stride + out_col_offset;
    if (USE_ATOMIC)
    {
        atomicAdd(&out[out_idx], learning_rate * acc);
    }
    else
    {
        out[out_idx] += learning_rate * acc;
    }
}

void predict_with_forest_fm_packed(
    const at::Tensor &bin_packed_fm,  // [F, N_packed] uint8
    int64_t N,
    const at::Tensor &tree_tensor,    // [T, max_nodes, 7] float32 or [T, max_nodes, 4] int32
    float learning_rate,
    int64_t out_col_offset,
    int64_t out_stride,
    at::Tensor &out)
{
    const auto device = bin_packed_fm.device();
    check_cuda_tensor(bin_packed_fm, at::kByte, device, "bin_packed_fm");
    const bool packed_nodes = (tree_tensor.scalar_type() == at::kInt);
    check_cuda_tensor(tree_tensor, packed_nodes ? at::kInt : at::kFloat, device, "tree_tensor");
    check_cuda_tensor(out, at::kFloat, device, "out");
    TORCH_CHECK(bin_packed_fm.dim() == 2 && tree_tensor.dim() == 3 &&
                tree_tensor.size(1) > 0 && tree_tensor.size(2) == (packed_nodes ? 4 : 7),
                "Packed FM prediction inputs must have shapes [features, packed_rows] and [trees, nodes, 7 or int4]");
    TORCH_CHECK(out_stride >= 1 && out_col_offset >= 0 && out_col_offset < out_stride,
                "Invalid out_col_offset / out_stride");
    const int64_t F = bin_packed_fm.size(0);
    const int64_t N_packed = bin_packed_fm.size(1);
    const int64_t T = tree_tensor.size(0);
    const int64_t max_nodes = tree_tensor.size(1);
    TORCH_CHECK(out.numel() == N * out_stride, "out size must equal rows * out_stride");
    TORCH_CHECK(std::isfinite(learning_rate), "learning_rate must be finite");
    if (N == 0 || T == 0) return;
    TORCH_CHECK(N_packed >= ((N + 1) >> 1), "bin_packed_fm width too small for N");

    const c10::cuda::CUDAGuard guard(device);
    const auto *properties = at::cuda::getDeviceProperties(device.index());
    const auto stream = c10::cuda::getCurrentCUDAStream(device.index());

    const int threads_per_block = 256;
    const int64_t blocks_x = (N + threads_per_block - 1) / threads_per_block;
    const int64_t blocks_y = (N < 32768 && T >= 64) ? std::min<int64_t>(16, (T + 15) / 16) : 1;
    TORCH_CHECK(blocks_x <= properties->maxGridSize[0] && blocks_y <= properties->maxGridSize[1],
                "Packed FM prediction launch exceeds CUDA grid limits");

    dim3 blocks(blocks_x, blocks_y);
    const void *t_ptr = packed_nodes
        ? static_cast<const void *>(tree_tensor.data_ptr<int32_t>())
        : static_cast<const void *>(tree_tensor.data_ptr<float>());

    if (blocks_y > 1 && packed_nodes) {
        predict_forest_fm_packed_kernel<true, true><<<blocks, threads_per_block, 0, stream.stream()>>>(
            bin_packed_fm.data_ptr<uint8_t>(), t_ptr, N, N_packed, F, T, max_nodes, learning_rate, out_col_offset, out_stride, out.data_ptr<float>());
    } else if (blocks_y > 1) {
        predict_forest_fm_packed_kernel<true, false><<<blocks, threads_per_block, 0, stream.stream()>>>(
            bin_packed_fm.data_ptr<uint8_t>(), t_ptr, N, N_packed, F, T, max_nodes, learning_rate, out_col_offset, out_stride, out.data_ptr<float>());
    } else if (packed_nodes) {
        predict_forest_fm_packed_kernel<false, true><<<blocks, threads_per_block, 0, stream.stream()>>>(
            bin_packed_fm.data_ptr<uint8_t>(), t_ptr, N, N_packed, F, T, max_nodes, learning_rate, out_col_offset, out_stride, out.data_ptr<float>());
    } else {
        predict_forest_fm_packed_kernel<false, false><<<blocks, threads_per_block, 0, stream.stream()>>>(
            bin_packed_fm.data_ptr<uint8_t>(), t_ptr, N, N_packed, F, T, max_nodes, learning_rate, out_col_offset, out_stride, out.data_ptr<float>());
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
