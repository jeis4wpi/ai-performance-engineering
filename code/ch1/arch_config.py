# Runtime configuration helpers for Chapter 1 examples targeting NVIDIA Blackwell.
# Applies PyTorch knobs that keep us on Tensor Core fast paths and exposes a
# reusable torch.compile wrapper with safe defaults for static workloads.

from __future__ import annotations

import os
import torch


def _configure_torch_defaults() -> None:
    """Enable TF32 Tensor Core math and cuDNN autotune."""
    try:
        torch.set_float32_matmul_precision("high")
    except (AttributeError, RuntimeError):
        # Older PyTorch builds might not expose this helper; ignore quietly.
        pass

    if torch.backends.cuda.is_built():
        matmul_backend = getattr(torch.backends.cuda, "matmul", None)
        if matmul_backend is not None and hasattr(matmul_backend, "fp32_precision"):
            try:
                matmul_backend.fp32_precision = "tf32"
            except RuntimeError:
                # If the new API is unavailable or conflicts with prior configuration,
                # leave PyTorch defaults unchanged.
                pass
    if torch.backends.cudnn.is_available():
        cudnn_conv = getattr(torch.backends.cudnn, "conv", None)
        if cudnn_conv is not None and hasattr(cudnn_conv, "fp32_precision"):
            try:
                cudnn_conv.fp32_precision = "tf32"
            except RuntimeError:
                pass
        torch.backends.cudnn.benchmark = True


def _configure_environment() -> None:
    """Set default TorchInductor knobs that play nicely on Blackwell."""
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", ".torch_inductor")
    os.environ.setdefault("TORCHINDUCTOR_FUSE_TRANSPOSE", "1")
    os.environ.setdefault("TORCHINDUCTOR_FUSE_ROTARY", "1")
    os.environ.setdefault("TORCHINDUCTOR_SCHEDULING", "1")


def compile_model(module: torch.nn.Module, *, mode: str = "reduce-overhead",
                  fullgraph: bool = False, dynamic: bool = False) -> torch.nn.Module:
    """
    Compile a model with torch.compile when available.

    Defaults target steady-state inference/training loops with static shapes.
    """
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return module
    return compile_fn(module, mode=mode, fullgraph=fullgraph, dynamic=dynamic)


_configure_environment()
_configure_torch_defaults()
