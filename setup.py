import os
from pathlib import Path
import sys
from setuptools import setup, find_packages

# Try importing torch and checking CUDA support
try:
    import torch
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME
    torch_available = True
    has_cuda = torch.version.cuda is not None and CUDA_HOME is not None
except ImportError:
    torch_available = False
    has_cuda = False

def get_extensions():
    extensions = []
    if has_cuda:
        print(f"Building with CUDA (found at {CUDA_HOME})")
        extensions.append(
            CUDAExtension(
                name="warpgbm.cuda.node_kernel",
                sources=[
                    "warpgbm/cuda/histogram_kernel.cu",
                    "warpgbm/cuda/best_split_kernel.cu",
                    "warpgbm/cuda/binner.cu",
                    "warpgbm/cuda/predict.cu",
                    "warpgbm/cuda/node_kernel.cpp",
                ],
                depends=["warpgbm/cuda/cuda_utils.h"],
                extra_compile_args={
                    "cxx": ["/Zc:preprocessor"] if os.name == "nt" else [],
                    "nvcc": ["-Xcompiler", "/Zc:preprocessor"] if os.name == "nt" else [],
                }
            )
        )
    elif any(arg.startswith("bdist") or arg in {"build", "build_ext", "editable_wheel", "install", "develop"}
             for arg in sys.argv[1:]):
        raise RuntimeError(
            "WarpGBM requires CUDA-enabled PyTorch and a CUDA toolkit to build. "
            "Install matching CUDA PyTorch, set CUDA_HOME, and build with --no-build-isolation. "
            "A CPU-only wheel is not supported."
        )
    return extensions

# Read version
with Path(__file__).with_name("version.txt").open() as f:
    version = f.read().strip()

setup(
    name="warpgbm",
    version=version,
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension} if torch_available else {},
    install_requires=[
        "torch",
        "numpy",
        "scikit-learn",
        "tqdm",
    ],
)
