#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"

__global__ void bin_column_kernel(
    const float *__restrict__ X,         // [N]
    const float *__restrict__ bin_edges, // [B - 1]
    int8_t *__restrict__ bin_indices,    // [N]
    int N,
    int B_minus1)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    float val = X[idx];
    int bin = 0;

    // Linear scan over edges: bin_edges is sorted
    while (bin < B_minus1 && val >= bin_edges[bin])
    {
        ++bin;
    }

    bin_indices[idx] = static_cast<int8_t>(bin);
}

// C++ launcher for calling from Python
void launch_bin_column_kernel(
    at::Tensor X,          // [N]
    at::Tensor bin_edges,  // [B - 1]
    at::Tensor bin_indices // [N]
)
{
    const auto device = X.device();
    check_cuda_tensor(X, at::kFloat, device, "X");
    check_cuda_tensor(bin_edges, at::kFloat, device, "bin_edges");
    check_cuda_tensor(bin_indices, at::kChar, device, "bin_indices");
    TORCH_CHECK(X.dim() == 1 && bin_edges.dim() == 1 && bin_indices.dim() == 1 &&
                bin_indices.size(0) == X.size(0), "Binner inputs must be vectors with matching input/output lengths");
    TORCH_CHECK(X.size(0) <= std::numeric_limits<int>::max() && bin_edges.size(0) <= 126,
                "Binner supports int32 row counts and at most 126 edges");
    const c10::cuda::CUDAGuard guard(device);
    const int N = X.size(0);
    const int B = bin_edges.size(0);

    if (N == 0) return;
    const int threads = 256;
    const int blocks = (N - 1) / threads + 1;
    const auto *properties = at::cuda::getDeviceProperties(device.index());
    TORCH_CHECK(blocks <= properties->maxGridSize[0], "Binner launch exceeds CUDA grid limits");
    const auto stream = c10::cuda::getCurrentCUDAStream(device.index());

    bin_column_kernel<<<blocks, threads, 0, stream.stream()>>>(
        X.data_ptr<float>(),
        bin_edges.data_ptr<float>(),
        bin_indices.data_ptr<int8_t>(),
        N,
        B);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__device__ __forceinline__ int8_t find_bin_val(
    float val,
    const float *__restrict__ edges_f,
    int B_minus1)
{
    if (isnan(val))
    {
        return static_cast<int8_t>(-1);
    }
    if (B_minus1 <= 8)
    {
        int bin = 0;
        while (bin < B_minus1 && val >= edges_f[bin])
        {
            ++bin;
        }
        return static_cast<int8_t>(bin);
    }
    int lo = 0;
    int hi = B_minus1;
    while (lo < hi)
    {
        int mid = (lo + hi) >> 1;
        if (val >= edges_f[mid])
        {
            lo = mid + 1;
        }
        else
        {
            hi = mid;
        }
    }
    return static_cast<int8_t>(lo);
}

__global__ void bin_2d_fm_kernel(
    const float *__restrict__ X,          // [N, F] row-major
    const float *__restrict__ bin_edges,  // [F, B - 1]
    int8_t *__restrict__ bin_indices_fm,  // [F, N] feature-major
    int64_t N,
    int64_t F,
    int B_minus1)
{
    __shared__ int8_t tile[16][17];
    int64_t f = static_cast<int64_t>(blockIdx.x) * 16 + threadIdx.x;
    int64_t n = static_cast<int64_t>(blockIdx.y) * 16 + threadIdx.y;

    if (n < N && f < F)
    {
        float val = X[n * F + f];
        tile[threadIdx.y][threadIdx.x] = find_bin_val(val, bin_edges + f * B_minus1, B_minus1);
    }
    __syncthreads();

    int64_t f_out = static_cast<int64_t>(blockIdx.x) * 16 + threadIdx.y;
    int64_t n_out = static_cast<int64_t>(blockIdx.y) * 16 + threadIdx.x;
    if (f_out < F && n_out < N)
    {
        bin_indices_fm[f_out * N + n_out] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void bin_2d_rm_kernel(
    const float *__restrict__ X,          // [N, F] row-major
    const float *__restrict__ bin_edges,  // [F, B - 1]
    int8_t *__restrict__ bin_indices_rm,  // [N, F] row-major
    int64_t N,
    int64_t F,
    int B_minus1)
{
    int64_t f = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t n = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (n < N && f < F)
    {
        float val = X[n * F + f];
        bin_indices_rm[n * F + f] = find_bin_val(val, bin_edges + f * B_minus1, B_minus1);
    }
}

void launch_bin_2d_kernel(
    const at::Tensor &X,          // [N, F] float32
    const at::Tensor &bin_edges,  // [F, B - 1] float32
    at::Tensor &bin_indices,      // [F, N] if feature_major else [N, F] int8
    bool feature_major)
{
    const auto device = X.device();
    check_cuda_tensor(X, at::kFloat, device, "X");
    check_cuda_tensor(bin_edges, at::kFloat, device, "bin_edges");
    check_cuda_tensor(bin_indices, at::kChar, device, "bin_indices");
    TORCH_CHECK(X.dim() == 2 && bin_edges.dim() == 2 && bin_indices.dim() == 2,
                "2D binner inputs must be 2D tensors");
    const int64_t N = X.size(0);
    const int64_t F = X.size(1);
    const int64_t B_minus1 = bin_edges.size(1);
    TORCH_CHECK(bin_edges.size(0) == F && B_minus1 >= 1 && B_minus1 <= 126,
                "bin_edges must have shape [F, B-1] with 1 <= B-1 <= 126");
    if (feature_major)
    {
        TORCH_CHECK(bin_indices.size(0) == F && bin_indices.size(1) == N,
                    "feature-major bin_indices must have shape [F, N]");
    }
    else
    {
        TORCH_CHECK(bin_indices.size(0) == N && bin_indices.size(1) == F,
                    "row-major bin_indices must have shape [N, F]");
    }
    if (N == 0 || F == 0) return;

    const c10::cuda::CUDAGuard guard(device);
    const auto *properties = at::cuda::getDeviceProperties(device.index());
    const auto stream = c10::cuda::getCurrentCUDAStream(device.index());

    if (feature_major)
    {
        dim3 threads(16, 16);
        dim3 blocks((F + 15) / 16, (N + 15) / 16);
        TORCH_CHECK(blocks.x <= properties->maxGridSize[0] && blocks.y <= properties->maxGridSize[1],
                    "2D binner launch exceeds CUDA grid limits");
        bin_2d_fm_kernel<<<blocks, threads, 0, stream.stream()>>>(
            X.data_ptr<float>(),
            bin_edges.data_ptr<float>(),
            bin_indices.data_ptr<int8_t>(),
            N, F, static_cast<int>(B_minus1));
    }
    else
    {
        dim3 threads(32, 8);
        dim3 blocks((F + 31) / 32, (N + 7) / 8);
        TORCH_CHECK(blocks.x <= properties->maxGridSize[0] && blocks.y <= properties->maxGridSize[1],
                    "2D binner launch exceeds CUDA grid limits");
        bin_2d_rm_kernel<<<blocks, threads, 0, stream.stream()>>>(
            X.data_ptr<float>(),
            bin_edges.data_ptr<float>(),
            bin_indices.data_ptr<int8_t>(),
            N, F, static_cast<int>(B_minus1));
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void pack_int8_to_4bit_fm_kernel(
    const int8_t *__restrict__ X,       // [N_chunk, F] row-major
    uint8_t *__restrict__ out_packed,   // [F, N_total_packed]
    int32_t *__restrict__ flags,        // [2]: flags[0]=out_of_range, flags[1]=has_missing (or nullptr)
    int64_t N_chunk,
    int64_t F,
    int64_t N_total_packed,
    int64_t p_offset,
    int num_bins)
{
    // Each block handles 16 features x 16 packed pairs (32 rows)
    __shared__ uint8_t tile[16][17]; // [feat_local][pair_local]
    const int64_t f = static_cast<int64_t>(blockIdx.x) * 16 + threadIdx.x;
    const int64_t p = static_cast<int64_t>(blockIdx.y) * 16 + threadIdx.y;
    const int64_t n0 = 2 * p;
    const int64_t n1 = n0 + 1;

    uint8_t packed = 0;
    if (f < F && n0 < N_chunk)
    {
        const int v0 = static_cast<int>(X[n0 * F + f]);
        if (flags != nullptr)
        {
            if (v0 < -1 || v0 >= num_bins) atomicOr(&flags[0], 1);
            if (v0 < 0) atomicOr(&flags[1], 1);
        }
        const uint8_t nib0 = static_cast<uint8_t>((v0 + 1) & 0x0F);
        uint8_t nib1 = 0;
        if (n1 < N_chunk)
        {
            const int v1 = static_cast<int>(X[n1 * F + f]);
            if (flags != nullptr)
            {
                if (v1 < -1 || v1 >= num_bins) atomicOr(&flags[0], 1);
                if (v1 < 0) atomicOr(&flags[1], 1);
            }
            nib1 = static_cast<uint8_t>((v1 + 1) & 0x0F);
        }
        packed = nib0 | static_cast<uint8_t>(nib1 << 4);
    }
    tile[threadIdx.y][threadIdx.x] = packed;
    __syncthreads();

    const int64_t f_out = static_cast<int64_t>(blockIdx.x) * 16 + threadIdx.y;
    const int64_t p_out = static_cast<int64_t>(blockIdx.y) * 16 + threadIdx.x;
    const int64_t N_chunk_packed = (N_chunk + 1) >> 1;
    if (f_out < F && p_out < N_chunk_packed)
    {
        out_packed[f_out * N_total_packed + p_offset + p_out] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void bin_2d_pack4_fm_kernel(
    const float *__restrict__ X,          // [N_chunk, F] float32
    const float *__restrict__ bin_edges,  // [F, B - 1] float32
    uint8_t *__restrict__ out_packed,     // [F, N_total_packed]
    int64_t N_chunk,
    int64_t F,
    int64_t N_total_packed,
    int64_t p_offset,
    int B_minus1)
{
    __shared__ uint8_t tile[16][17];
    const int64_t f = static_cast<int64_t>(blockIdx.x) * 16 + threadIdx.x;
    const int64_t p = static_cast<int64_t>(blockIdx.y) * 16 + threadIdx.y;
    const int64_t n0 = 2 * p;
    const int64_t n1 = n0 + 1;

    uint8_t packed = 0;
    if (f < F && n0 < N_chunk)
    {
        const float *edges_f = bin_edges + f * B_minus1;
        const int8_t b0 = find_bin_val(X[n0 * F + f], edges_f, B_minus1);
        const uint8_t nib0 = static_cast<uint8_t>((b0 + 1) & 0x0F);
        uint8_t nib1 = 0;
        if (n1 < N_chunk)
        {
            const int8_t b1 = find_bin_val(X[n1 * F + f], edges_f, B_minus1);
            nib1 = static_cast<uint8_t>((b1 + 1) & 0x0F);
        }
        packed = nib0 | static_cast<uint8_t>(nib1 << 4);
    }
    tile[threadIdx.y][threadIdx.x] = packed;
    __syncthreads();

    const int64_t f_out = static_cast<int64_t>(blockIdx.x) * 16 + threadIdx.y;
    const int64_t p_out = static_cast<int64_t>(blockIdx.y) * 16 + threadIdx.x;
    const int64_t N_chunk_packed = (N_chunk + 1) >> 1;
    if (f_out < F && p_out < N_chunk_packed)
    {
        out_packed[f_out * N_total_packed + p_offset + p_out] = tile[threadIdx.x][threadIdx.y];
    }
}

void launch_pack_int8_to_4bit_fm(
    const at::Tensor &X_int8,
    at::Tensor &out_packed,
    at::Tensor &flags,
    int64_t p_offset,
    int num_bins)
{
    const auto device = X_int8.device();
    check_cuda_tensor(X_int8, at::kChar, device, "X_int8");
    check_cuda_tensor(out_packed, at::kByte, device, "out_packed");
    TORCH_CHECK(X_int8.dim() == 2 && out_packed.dim() == 2, "pack_int8_to_4bit_fm inputs must be 2D");
    const int64_t N_chunk = X_int8.size(0);
    const int64_t F = X_int8.size(1);
    const int64_t N_total_packed = out_packed.size(1);
    TORCH_CHECK(out_packed.size(0) == F && num_bins >= 2 && num_bins <= 15,
                "out_packed must have matching features and num_bins in [2, 15]");
    if (N_chunk == 0 || F == 0) return;
    const int64_t N_chunk_packed = (N_chunk + 1) >> 1;
    TORCH_CHECK(p_offset >= 0 && p_offset + N_chunk_packed <= N_total_packed,
                "pack_int8_to_4bit_fm p_offset out of bounds");

    int32_t *flags_ptr = nullptr;
    if (flags.numel() > 0)
    {
        check_cuda_tensor(flags, at::kInt, device, "flags");
        TORCH_CHECK(flags.numel() >= 2, "flags must have at least 2 int32 elements");
        flags_ptr = flags.data_ptr<int32_t>();
    }
    const c10::cuda::CUDAGuard guard(device);
    const auto stream = c10::cuda::getCurrentCUDAStream(device.index());
    dim3 threads(16, 16);
    dim3 blocks((F + 15) / 16, (N_chunk_packed + 15) / 16);
    pack_int8_to_4bit_fm_kernel<<<blocks, threads, 0, stream.stream()>>>(
        X_int8.data_ptr<int8_t>(), out_packed.data_ptr<uint8_t>(), flags_ptr,
        N_chunk, F, N_total_packed, p_offset, num_bins);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_bin_2d_pack4_fm(
    const at::Tensor &X,
    const at::Tensor &bin_edges,
    at::Tensor &out_packed,
    int64_t p_offset)
{
    const auto device = X.device();
    check_cuda_tensor(X, at::kFloat, device, "X");
    check_cuda_tensor(bin_edges, at::kFloat, device, "bin_edges");
    check_cuda_tensor(out_packed, at::kByte, device, "out_packed");
    TORCH_CHECK(X.dim() == 2 && bin_edges.dim() == 2 && out_packed.dim() == 2,
                "bin_2d_pack4_fm inputs must be 2D");
    const int64_t N_chunk = X.size(0);
    const int64_t F = X.size(1);
    const int64_t B_minus1 = bin_edges.size(1);
    const int64_t N_total_packed = out_packed.size(1);
    TORCH_CHECK(bin_edges.size(0) == F && out_packed.size(0) == F && B_minus1 >= 1 && B_minus1 <= 14,
                "bin_2d_pack4_fm requires matching features and 1 <= B-1 <= 14");
    if (N_chunk == 0 || F == 0) return;
    const int64_t N_chunk_packed = (N_chunk + 1) >> 1;
    TORCH_CHECK(p_offset >= 0 && p_offset + N_chunk_packed <= N_total_packed,
                "bin_2d_pack4_fm p_offset out of bounds");

    const c10::cuda::CUDAGuard guard(device);
    const auto stream = c10::cuda::getCurrentCUDAStream(device.index());
    dim3 threads(16, 16);
    dim3 blocks((F + 15) / 16, (N_chunk_packed + 15) / 16);
    bin_2d_pack4_fm_kernel<<<blocks, threads, 0, stream.stream()>>>(
        X.data_ptr<float>(), bin_edges.data_ptr<float>(), out_packed.data_ptr<uint8_t>(),
        N_chunk, F, N_total_packed, p_offset, static_cast<int>(B_minus1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
