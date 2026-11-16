#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <algorithm>
#include <cmath>
#include <limits>

inline void check_cuda_tensor(const at::Tensor &tensor, at::ScalarType dtype,
                              const c10::Device &device, const char *name)
{
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.device() == device, name, " must be on ", device);
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has incorrect dtype; expected ", dtype);
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}
