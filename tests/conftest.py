"""Pytest fixtures for fm_change_detection test suite."""

import tempfile
from pathlib import Path

import pytest
import torch

from fm_change_detection.data import generate_synthetic_dataset


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def synthetic_dataset_dir(tmp_dir):
    data_path = tmp_dir / "synthetic"
    generate_synthetic_dataset(
        output_root=data_path, num_scenes=3, tiles_per_scene=2, image_size=64, seed=42
    )
    return data_path


@pytest.fixture
def dummy_image_pair():
    # [1, 3, 64, 64] float tensors in [0, 1]
    t1 = torch.rand(1, 3, 64, 64, dtype=torch.float32)
    t2 = t1.clone()
    # Add synthetic change patch to t2
    t2[:, :, 16:32, 16:32] += 0.5
    t2 = torch.clamp(t2, 0.0, 1.0)
    return t1, t2
