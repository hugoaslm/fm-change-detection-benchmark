"""Threshold estimation methods (Otsu unlabeled and Validation F1 calibration)."""

import numpy as np
from torch import Tensor


def compute_otsu_threshold(score_maps: Tensor | np.ndarray, num_bins: int = 256) -> float:
    """Compute global Otsu threshold from score maps without labels.

    Args:
        score_maps: Tensor or array of continuous anomaly scores.
        num_bins: Number of histogram bins.

    Returns:
        Optimal Otsu threshold float value.
    """
    if isinstance(score_maps, Tensor):
        scores = score_maps.detach().cpu().numpy().ravel()
    else:
        scores = score_maps.ravel()

    # Filter out NaNs or Inf if any
    valid_scores = scores[np.isfinite(scores)]
    if len(valid_scores) == 0:
        return 0.5

    min_val, max_val = float(valid_scores.min()), float(valid_scores.max())
    if min_val == max_val:
        return min_val

    counts, bin_edges = np.histogram(valid_scores, bins=num_bins, range=(min_val, max_val))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total = valid_scores.size

    w1 = np.cumsum(counts) / total
    w2 = 1.0 - w1
    m1 = np.cumsum(counts * bin_centers) / (np.cumsum(counts) + 1e-10)
    m2 = (np.sum(counts * bin_centers) - np.cumsum(counts * bin_centers)) / (
        np.sum(counts) - np.cumsum(counts) + 1e-10
    )
    var = w1 * w2 * (m1 - m2) ** 2
    max_var = np.max(var)
    max_indices = np.where(np.isclose(var, max_var, rtol=1e-5, atol=1e-5))[0]
    idx = int(np.mean(max_indices))
    return float(bin_centers[idx])


def fit_calibrated_f1_threshold(
    score_maps: Tensor | np.ndarray,
    masks: Tensor | np.ndarray,
    num_candidates: int = 512,
) -> float:
    """Find global threshold maximizing F1 score on validation set with ground-truth masks.

    Args:
        score_maps: Continuous anomaly scores.
        masks: Ground truth binary change masks (bool or 0/1).
        num_candidates: Number of candidate thresholds to search.

    Returns:
        Threshold float maximizing F1.
    """
    if isinstance(score_maps, Tensor):
        scores = score_maps.detach().cpu().numpy().ravel()
    else:
        scores = score_maps.ravel()

    if isinstance(masks, Tensor):
        targets = masks.detach().cpu().numpy().ravel().astype(bool)
    else:
        targets = masks.ravel().astype(bool)

    valid_mask = np.isfinite(scores)
    scores = scores[valid_mask]
    targets = targets[valid_mask]

    if len(scores) == 0:
        return 0.5

    min_s, max_s = float(scores.min()), float(scores.max())
    if min_s == max_s:
        return min_s

    thresholds = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, num_candidates)))
    best_f1 = -1.0
    best_thresh = float(thresholds[0])

    # Vectorized / efficient search
    total_positives = np.sum(targets)
    if total_positives == 0:
        return float(min_s + (max_s - min_s) * 0.5)

    for th in thresholds:
        preds = scores >= th
        tp = np.sum(preds & targets)
        fp = np.sum(preds & ~targets)
        fn = total_positives - tp

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(th)

    return best_thresh
