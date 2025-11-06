"""Chapter 8 helpers for configuring PyTorch on Blackwell-era GPUs.

This module keeps the PyTorch demos aligned with the book narrative by:
  * enabling TF32 Tensor Core math by default (as recommended for training/profiling)
  * exposing a small wrapper around torch.compile that can be toggled via env vars

Environment variables:
  USE_COMPILE / TORCH_USE_COMPILE    -> 0/1 toggle for torch.compile (default: 0)
  COMPILE_MODE / TORCH_COMPILE_MODE  -> torch.compile mode name (default: reduce-overhead)
"""

from __future__ import annotations

import os
import warnings

import torch

from common.python.compile_utils import enable_tf32

enable_tf32()

_VALID_MODES = {
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
}


def _env_flag(name: str, default: str = "0") -> bool:
    """Parse an integer-like environment flag safely."""
    raw = os.getenv(name, default)
    try:
        return bool(int(raw))
    except (TypeError, ValueError):
        warnings.warn(f"Expected {name} to be 0 or 1; got {raw!r}. Treating as disabled.")
        return False


def should_use_compile() -> bool:
    """Return True if torch.compile should be applied based on env toggles."""
    if not hasattr(torch, "compile"):
        return False
    return _env_flag("USE_COMPILE", os.getenv("TORCH_USE_COMPILE", "0"))


def get_compile_mode(default: str = "reduce-overhead") -> str:
    """Return the requested torch.compile mode, falling back to a safe default."""
    mode = os.getenv("COMPILE_MODE", os.getenv("TORCH_COMPILE_MODE", default))
    if mode not in _VALID_MODES:
        warnings.warn(
            f"Unsupported torch.compile mode {mode!r}; "
            f"falling back to {default!r}."
        )
        return default
    return mode


def maybe_compile(fn, *, default_mode: str = "reduce-overhead"):
    """Return a compiled version of `fn` if torch.compile is requested and available."""
    if not should_use_compile():
        return fn

    mode = get_compile_mode(default_mode)
    try:
        return torch.compile(fn, mode=mode)
    except Exception as exc:  # pragma: no cover - torch.compile failures are rare
        warnings.warn(f"torch.compile failed in mode {mode!r}: {exc}. Running eager path.")
        return fn
