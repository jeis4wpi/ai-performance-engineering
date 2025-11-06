"""Blackwell architecture helper utilities for Chapter 20."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass(frozen=True)
class ArchInfo:
    compute_capability: Tuple[int, int]
    arch: str
    sm: str


_BLACKWELL_VARIANTS: List[ArchInfo] = [
    ArchInfo((10, 0), "compute_100", "sm_100"),  # B200 / GB200
    ArchInfo((10, 3), "compute_103", "sm_103"),  # B300 / GB300 (Blackwell Ultra)
    ArchInfo((12, 0), "compute_120", "sm_120"),  # GB10 family (consumer Blackwell)
    ArchInfo((12, 1), "compute_121", "sm_121"),  # GB10 refresh
]


def current_arch() -> ArchInfo:
    major, minor = torch.cuda.get_device_capability()
    for info in _BLACKWELL_VARIANTS:
        if (major, minor) == info.compute_capability:
            return info
    if major in (10, 12):
        return ArchInfo((major, minor), f"compute_{major}{minor}", f"sm_{major}{minor}")
    raise RuntimeError(
        f"Unsupported device capability SM{major}{minor}; expected Blackwell SM10x/SM12x"
    )


def nvcc_gencodes(include_arch_a: bool = False) -> List[str]:
    """Return recommended -gencode arguments for nvcc."""
    flags: List[str] = []
    for info in _BLACKWELL_VARIANTS:
        flags.extend(
            [
                "-gencode",
                f"arch={info.arch},code={info.sm}",
                "-gencode",
                f"arch={info.arch},code={info.arch}",
            ]
        )
    if include_arch_a:
        flags.extend(
            [
                "-gencode",
                "arch=compute_100,code=sm_100a",
            ]
        )
    return flags
