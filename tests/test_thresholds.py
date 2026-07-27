"""Tests for Otsu thresholding and calibrated max-F1 search."""

import numpy as np

from fm_change_detection.thresholds import (
    compute_otsu_threshold,
    fit_calibrated_f1_threshold,
)


def test_otsu_threshold_bimodal():
    # Bimodal distribution around 0.2 and 0.8
    bg = np.random.normal(loc=0.2, scale=0.05, size=500)
    fg = np.random.normal(loc=0.8, scale=0.05, size=500)
    scores = np.concatenate([bg, fg])

    th = compute_otsu_threshold(scores)
    assert 0.4 <= th <= 0.6


def test_calibrated_f1_threshold():
    scores = np.linspace(0.0, 1.0, 100)
    masks = scores >= 0.7  # True threshold at 0.7

    th = fit_calibrated_f1_threshold(scores, masks, num_candidates=50)
    assert 0.65 <= th <= 0.75
