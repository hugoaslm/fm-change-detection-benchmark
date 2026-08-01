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
    modified = image.clone()
    modified[:, region_mask] = (modified[:, region_mask] + magnitude).clamp(0.0, 1.0)
    return modified


def synthetic_change_mask(region_mask: Tensor, real_mask: Tensor) -> Tensor:
    return region_mask & ~real_mask
