"""CUDA extension loader for ch12 kernels."""

from pathlib import Path
import torch
import sys

# Import build utilities to prevent hangs from stale locks
try:
    from common.python.build_utils import ensure_clean_build_directory
except ImportError:
    # Fallback if build_utils not available
    def ensure_clean_build_directory(build_dir, max_lock_age_seconds=300):
        pass

_EXTENSIONS = {}


def _get_extension_dir():
    """Get the directory containing CUDA extension files."""
    return Path(__file__).parent


def load_kernel_fusion_extension():
    """Load the kernel fusion CUDA extension."""
    if "kernel_fusion" not in _EXTENSIONS:
        try:
            from torch.utils.cpp_extension import load
            
            extension_dir = _get_extension_dir()
            cuda_source = extension_dir / "kernel_fusion_kernels.cu"
            
            common_headers = Path(__file__).parent.parent.parent / "common" / "headers"
            build_dir = extension_dir / "build"
            build_dir.mkdir(parents=True, exist_ok=True)
            # Clean stale locks before building to prevent hangs
            ensure_clean_build_directory(build_dir)
            _EXTENSIONS["kernel_fusion"] = load(
                name="kernel_fusion_kernels",
                sources=[str(cuda_source)],
                extra_cuda_cflags=["-lineinfo", f"-I{common_headers}"],
                verbose=True,
                build_directory=str(build_dir),
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load kernel_fusion CUDA extension: {e}"
            ) from e
    
    return _EXTENSIONS["kernel_fusion"]


def load_graph_bandwidth_extension():
    """Load the graph bandwidth CUDA extension."""
    if "graph_bandwidth" not in _EXTENSIONS:
        try:
            from torch.utils.cpp_extension import load
            
            extension_dir = _get_extension_dir()
            cuda_source = extension_dir / "graph_bandwidth_kernels.cu"
            
            common_headers = Path(__file__).parent.parent.parent / "common" / "headers"
            build_dir = extension_dir / "build"
            build_dir.mkdir(parents=True, exist_ok=True)
            # Clean stale locks before building to prevent hangs
            ensure_clean_build_directory(build_dir)
            _EXTENSIONS["graph_bandwidth"] = load(
                name="graph_bandwidth_kernels",
                sources=[str(cuda_source)],
                extra_cuda_cflags=["-lineinfo", f"-I{common_headers}"],
                verbose=False,
                build_directory=str(build_dir),
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load graph_bandwidth CUDA extension: {e}"
            ) from e
    
    return _EXTENSIONS["graph_bandwidth"]


def load_work_queue_extension():
    """Load the work queue CUDA extension."""
    if "work_queue" not in _EXTENSIONS:
        try:
            from torch.utils.cpp_extension import load
            import shutil
            
            extension_dir = _get_extension_dir()
            cuda_source = extension_dir / "work_queue_kernels.cu"
            
            common_headers = Path(__file__).parent.parent.parent / "common" / "headers"
            build_dir = extension_dir / "build"
            build_dir.mkdir(parents=True, exist_ok=True)
            
            # Clean stale locks before building to prevent hangs
            ensure_clean_build_directory(build_dir)
            
            # If the .so file exists but fails to load (undefined symbol), remove it to force rebuild
            so_file = build_dir / "work_queue_kernels.so"
            if so_file.exists():
                try:
                    # Try loading the existing .so to check if it's valid
                    import ctypes
                    ctypes.CDLL(str(so_file))
                except (OSError, ImportError):
                    # If loading fails (undefined symbol), remove it
                    so_file.unlink(missing_ok=True)
            
            _EXTENSIONS["work_queue"] = load(
                name="work_queue_kernels",
                sources=[str(cuda_source)],
                extra_cuda_cflags=["-lineinfo", f"-I{common_headers}"],
                verbose=False,
                build_directory=str(build_dir),
            )
        except RuntimeError as e:
            error_str = str(e)
            # Handle undefined symbol error by cleaning and retrying
            if "undefined symbol" in error_str.lower() or "_ZN2at10TensorBase" in error_str:
                # Clean build directory and retry once
                build_dir = extension_dir / "build"
                if build_dir.exists():
                    so_file = build_dir / "work_queue_kernels.so"
                    so_file.unlink(missing_ok=True)
                    # Retry loading (will rebuild)
                    _EXTENSIONS["work_queue"] = load(
                        name="work_queue_kernels",
                        sources=[str(cuda_source)],
                        extra_cuda_cflags=["-lineinfo", f"-I{common_headers}"],
                        verbose=False,
                        build_directory=str(build_dir),
                    )
                else:
                    raise RuntimeError(
                        f"Failed to load work_queue CUDA extension: {e}"
                    ) from e
            else:
                raise RuntimeError(
                    f"Failed to load work_queue CUDA extension: {e}"
                ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to load work_queue CUDA extension: {e}"
            ) from e
    
    return _EXTENSIONS["work_queue"]


def load_cuda_graphs_extension():
    """Load the CUDA graphs extension."""
    if "cuda_graphs" not in _EXTENSIONS:
        try:
            from torch.utils.cpp_extension import load
            
            extension_dir = _get_extension_dir()
            cuda_source = extension_dir / "cuda_graphs_kernels.cu"
            
            common_headers = Path(__file__).parent.parent.parent / "common" / "headers"
            build_dir = extension_dir / "build"
            build_dir.mkdir(parents=True, exist_ok=True)
            # Clean stale locks before building to prevent hangs
            ensure_clean_build_directory(build_dir)
            _EXTENSIONS["cuda_graphs"] = load(
                name="cuda_graphs_kernels",
                sources=[str(cuda_source)],
                extra_cuda_cflags=["-lineinfo", f"-I{common_headers}"],
                verbose=False,
                build_directory=str(build_dir),
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load cuda_graphs CUDA extension: {e}"
            ) from e
    
    return _EXTENSIONS["cuda_graphs"]

