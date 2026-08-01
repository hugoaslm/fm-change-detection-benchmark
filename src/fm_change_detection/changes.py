"""Controlled synthetic change synthesis for detectability-frontier experiments.

A synthetic change is an additive brightness offset applied to a compact,
axis-aligned region of the T2 timestamp that the real labels mark as unchanged.
Because both the spatial extent and the intensity of the change are known
exactly, the resulting per-pixel change mask is controlled ground truth. This
lets the benchmark measure how detection performance degrades as changes
become smaller or fainter (the "detectability frontier") without relying on
the fixed real LEVIR-CD transitions.
"""

import math

import numpy as np
import torch
from torch import Tensor


def pick_change_region(
    real_mask: Tensor,
    area_fraction: float,
    rng: np.random.Generator,
    max_attempts: int = 48,
    max_overlap: float = 0.05,
) -> Tensor | None:
    """Sample a square change region mostly free of real change pixels.

    Args:
        real_mask: Binary [H, W] tensor marking real change in the tile.
        area_fraction: Requested change area as a fraction of the tile.
        rng: Seeded NumPy generator for deterministic placement.
        max_attempts: Number of rejection-sampling attempts.
        max_overlap: Maximum allowed fraction of the region covered by real
            change before the region is rejected.

    Returns:
        Boolean [H, W] region tensor, or None if no clean region was found.
    """
    height, width = real_mask.shape
    side = max(1, round(math.sqrt(area_fraction * height * width)))
    side = min(side, height, width)

    for _ in range(max_attempts):
        top = int(rng.integers(0, height - side + 1))
        left = int(rng.integers(0, width - side + 1))
        region = torch.zeros((height, width), dtype=torch.bool)
        region[top : top + side, left : left + side] = True
        region_area = float(region.sum())
        if region_area == 0.0:
            continue
        overlap = float((region & real_mask).sum()) / region_area
        if overlap <= max_overlap:
            return region
    return None


def apply_additive_change(image: Tensor, region_mask: Tensor, magnitude: float) -> Tensor:
    """Add an intensity offset inside a region of a [3, H, W] image in [0, 1].

    Pixels outside ``region_mask`` are left untouched. The result is clamped to
    the valid image range so the offset never produces out-of-range values.
    """
    modified = image.clone()
    modified[:, region_mask] = (modified[:, region_mask] + magnitude).clamp(0.0, 1.0)
    return modified


def synthetic_change_mask(region_mask: Tensor, real_mask: Tensor) -> Tensor:
    """Ground-truth mask for a synthetic change.

    Only pixels that were actually modified (the region excluding any real
    change pixels) count as changed.
    """
    return region_mask & ~real_mask
