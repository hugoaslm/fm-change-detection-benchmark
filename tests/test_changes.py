"""Tests for controlled synthetic change synthesis."""

import numpy as np
import torch

from fm_change_detection.changes import (
    apply_additive_change,
    pick_change_region,
    synthetic_change_mask,
)


def test_apply_additive_change_only_touches_region():
    image = torch.zeros(3, 16, 16)
    region = torch.zeros(16, 16, dtype=torch.bool)
    region[2:6, 2:6] = True
    modified = apply_additive_change(image, region, 0.25)
    assert torch.allclose(modified[:, ~region], torch.zeros(3, 16, 16)[:, ~region])
    assert bool(torch.all(modified[:, region] == 0.25))


def test_apply_additive_change_clamps_to_unit_range():
    image = torch.ones(3, 8, 8) * 0.9
    region = torch.ones(8, 8, dtype=torch.bool)
    modified = apply_additive_change(image, region, 0.4)
    assert modified.max() <= 1.0


def test_pick_change_region_respects_area_and_avoids_real_change():
    mask = torch.zeros(64, 64, dtype=torch.bool)
    mask[48:64, 48:64] = True  # real change in the bottom-right corner
    rng = np.random.default_rng(42)
    region = pick_change_region(mask, area_fraction=0.09, rng=rng)
    assert region is not None
    requested = 0.09 * 64 * 64
    assert abs(region.sum() - requested) <= 0.3 * requested
    overlap = (region & mask).sum() / region.sum()
    assert overlap <= 0.05


def test_pick_change_region_is_deterministic():
    mask = torch.zeros(64, 64, dtype=torch.bool)
    first = pick_change_region(mask, 0.04, np.random.default_rng(7))
    second = pick_change_region(mask, 0.04, np.random.default_rng(7))
    assert torch.equal(first, second)


def test_pick_change_region_returns_none_when_fully_changed():
    mask = torch.ones(32, 32, dtype=torch.bool)
    region = pick_change_region(mask, 0.25, np.random.default_rng(1), max_attempts=8)
    assert region is None


def test_synthetic_change_mask_excludes_real_change_pixels():
    region = torch.ones(16, 16, dtype=torch.bool)
    real = torch.zeros(16, 16, dtype=torch.bool)
    real[0, 0] = True
    mask = synthetic_change_mask(region, real)
    assert mask.sum() == region.sum() - 1
    assert not mask[0, 0]
