"""CUDA extension loader for ch6 kernels."""

from pathlib import Path
import torch
import sys

_EXTENSIONS = {}


def _get_extension_dir():
    """Get the directory containing CUDA extension files."""
    return Path(__file__).parent


def load_coalescing_extension():
    """Load the coalescing kernels CUDA extension."""
    if "coalescing" not in _EXTENSIONS:
        try:
            from torch.utils.cpp_extension import load
            
            extension_dir = _get_extension_dir()
            cuda_source = extension_dir / "coalescing_kernels.cu"
            
            # Use load() to compile and load the extension
            # Include common headers directory for profiling_helpers.cuh
            common_headers = Path(__file__).parent.parent.parent / "common" / "headers"
            build_dir = extension_dir / "build"
            build_dir.mkdir(parents=True, exist_ok=True)
            _EXTENSIONS["coalescing"] = load(
                name="coalescing_kernels",
                sources=[str(cuda_source)],
                extra_cuda_cflags=["-lineinfo", f"-I{common_headers}"],
                verbose=False,
                build_directory=str(build_dir),
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load coalescing CUDA extension: {e}\n"
                f"Make sure CUDA toolkit is installed and accessible."
            ) from e
    
    return _EXTENSIONS["coalescing"]


def load_bank_conflicts_extension():
    """Load the bank conflicts kernels CUDA extension."""
    if "bank_conflicts" not in _EXTENSIONS:
        try:
            from torch.utils.cpp_extension import load
            
            extension_dir = _get_extension_dir()
            cuda_source = extension_dir / "bank_conflicts_kernels.cu"
            
            common_headers = Path(__file__).parent.parent.parent / "common" / "headers"
            build_dir = extension_dir / "build"
            build_dir.mkdir(parents=True, exist_ok=True)
            _EXTENSIONS["bank_conflicts"] = load(
                name="bank_conflicts_kernels",
                sources=[str(cuda_source)],
                extra_cuda_cflags=["-lineinfo", f"-I{common_headers}"],
                verbose=False,
                build_directory=str(build_dir),
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load bank_conflicts CUDA extension: {e}"
            ) from e
    
    return _EXTENSIONS["bank_conflicts"]


def load_ilp_extension():
    """Load the ILP kernels CUDA extension."""
    if "ilp" not in _EXTENSIONS:
        try:
            from torch.utils.cpp_extension import load
            
            extension_dir = _get_extension_dir()
            cuda_source = extension_dir / "ilp_kernels.cu"
            
            common_headers = Path(__file__).parent.parent.parent / "common" / "headers"
            build_dir = extension_dir / "build"
            build_dir.mkdir(parents=True, exist_ok=True)
            _EXTENSIONS["ilp"] = load(
                name="ilp_kernels",
                sources=[str(cuda_source)],
                extra_cuda_cflags=["-lineinfo", f"-I{common_headers}"],
                verbose=False,
                build_directory=str(build_dir),
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load ILP CUDA extension: {e}"
            ) from e
    
    return _EXTENSIONS["ilp"]


def load_launch_bounds_extension():
    """Load the launch bounds CUDA extension."""
    if "launch_bounds" not in _EXTENSIONS:
        try:
            from torch.utils.cpp_extension import load
            
            extension_dir = _get_extension_dir()
            cuda_source = extension_dir / "launch_bounds_kernels.cu"
            
            common_headers = Path(__file__).parent.parent.parent / "common" / "headers"
            build_dir = extension_dir / "build"
            build_dir.mkdir(parents=True, exist_ok=True)
            _EXTENSIONS["launch_bounds"] = load(
                name="launch_bounds_kernels",
                sources=[str(cuda_source)],
                extra_cuda_cflags=["-lineinfo", f"-I{common_headers}"],
                verbose=False,
                build_directory=str(build_dir),
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load launch_bounds CUDA extension: {e}"
            ) from e
    
    return _EXTENSIONS["launch_bounds"]

