"""Tests for dataset loading, validation, tiling, and mask properties."""

import pytest
import torch

from fm_change_detection.data import (
    LEVIRCDDataset,
    extract_scene_id,
    validate_dataset_layout,
)


def test_synthetic_dataset_layout_and_loading(synthetic_dataset_dir):
    counts = validate_dataset_layout(synthetic_dataset_dir)
    assert "train" in counts and "val" in counts and "test" in counts
    assert counts["train"] > 0

    ds = LEVIRCDDataset(synthetic_dataset_dir, split="train", input_size=64)
    assert len(ds) == counts["train"]

    sample = ds[0]
    assert "sample_id" in sample
    assert "scene_id" in sample
    assert isinstance(sample["image_t1"], torch.Tensor)
    assert isinstance(sample["image_t2"], torch.Tensor)
    assert isinstance(sample["change_mask"], torch.Tensor)

    assert sample["image_t1"].shape[0] == 3
    assert sample["image_t2"].shape[0] == 3
    assert sample["change_mask"].dtype == torch.bool


def test_scene_id_extraction():
    assert extract_scene_id("scene_001_tile_02") == "scene_001"
    assert extract_scene_id("levir_042_tile_01") == "levir_042"
    assert extract_scene_id("simple_sample") == "simple"


def test_validator_rejects_split_overlap(synthetic_dataset_dir):
    train_list = synthetic_dataset_dir / "list" / "train.txt"
    test_list = synthetic_dataset_dir / "list" / "test.txt"
    duplicate = train_list.read_text(encoding="utf-8").splitlines()[0]
    with open(test_list, "a", encoding="utf-8") as stream:
        stream.write(f"{duplicate}\n")

    with pytest.raises(ValueError, match="Split leakage"):
        validate_dataset_layout(synthetic_dataset_dir)


def test_validator_can_check_train_and_val_without_test_list(synthetic_dataset_dir):
    (synthetic_dataset_dir / "list" / "test.txt").unlink()
    counts = validate_dataset_layout(synthetic_dataset_dir, splits=("train", "val"))
    assert set(counts) == {"train", "val"}
