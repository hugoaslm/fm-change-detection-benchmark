import numpy as np
import pytest

from fm_change_detection.metrics import (
    compute_binary_metrics,
    compute_cluster_bootstrap_ci,
)


def test_known_confusion_matrix():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    masks = np.array([True, False, True, False])
    threshold = 0.5

    res = compute_binary_metrics(scores, masks, threshold)

    assert res.precision == pytest.approx(0.5)
    assert res.recall == pytest.approx(0.5)
    assert res.f1 == pytest.approx(0.5)
    assert res.iou == pytest.approx(1.0 / 3.0)
    assert res.false_positive_rate == pytest.approx(0.5)


def test_cluster_bootstrap_ci():
    sample_scores = [np.random.rand(10, 10) for _ in range(6)]
    sample_masks = [np.random.rand(10, 10) > 0.5 for _ in range(6)]
    scene_ids = ["scene1", "scene1", "scene2", "scene2", "scene3", "scene3"]

    ci = compute_cluster_bootstrap_ci(
        sample_scores, sample_masks, scene_ids, threshold=0.5, num_resamples=50, seed=42
    )

    assert "ap" in ci and "f1" in ci and "iou" in ci
    for lower, upper in ci.values():
        assert 0.0 <= lower <= upper <= 1.0
