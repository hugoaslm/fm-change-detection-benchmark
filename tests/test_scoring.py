"""Tests for cosine distance, standardized Euclidean distance, and upsampling."""

import torch

from fm_change_detection.scoring import (
    cosine_score,
    standardized_euclidean_score,
    upsample_score_map,
)


def test_cosine_score_zero_for_identical_features():
    t1 = torch.rand(2, 16, 8, 8)
    t2 = t1.clone()

    score = cosine_score(t1, t2)
    assert torch.allclose(score, torch.zeros_like(score), atol=1e-5)


def test_cosine_score_zero_for_two_zero_vectors():
    features = torch.zeros(1, 4, 2, 2)
    assert torch.equal(cosine_score(features, features), torch.zeros(1, 2, 2))


def test_larger_known_changes_produce_larger_scores():
    t1 = torch.ones(1, 4, 4, 4)
    # Small change
    t2_small = t1.clone()
    t2_small[:, :2, :, :] += 0.1

    # Large change
    t2_large = t1.clone()
    t2_large[:, :2, :, :] += 5.0

    score_small = cosine_score(t1, t2_small).mean()
    score_large = cosine_score(t1, t2_large).mean()

    assert score_large > score_small


def test_standardized_euclidean_score():
    t1 = torch.zeros(1, 4, 2, 2)
    t2 = torch.ones(1, 4, 2, 2)
    std = torch.tensor([1.0, 2.0, 1.0, 2.0])

    score = standardized_euclidean_score(t1, t2, std)
    assert score.shape == (1, 2, 2)
    assert (score > 0).all()


def test_upsample_score_map():
    score_map = torch.rand(2, 8, 8)
    upsampled = upsample_score_map(score_map, target_size=(32, 32))
    assert upsampled.shape == (2, 32, 32)
