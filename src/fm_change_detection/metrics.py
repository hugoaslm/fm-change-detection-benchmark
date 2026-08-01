from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from torch import Tensor


@dataclass
class MetricResults:
    iou: float
    average_precision: float
    auroc: float
    f1: float
    precision: float
    recall: float
    balanced_accuracy: float
    false_positive_rate: float
    threshold: float
    num_pixels: int
    num_images: int

    ci_95: dict[str, tuple[float, float]] | None = None


def compute_binary_metrics(
    score_maps: list[Tensor] | Tensor | np.ndarray,
    masks: list[Tensor] | Tensor | np.ndarray,
    threshold: float,
) -> MetricResults:
    if isinstance(score_maps, list):
        scores_arr = np.concatenate(
            [
                s.detach().cpu().numpy().ravel() if isinstance(s, Tensor) else np.asarray(s).ravel()
                for s in score_maps
            ]
        )
        masks_arr = np.concatenate(
            [
                m.detach().cpu().numpy().ravel().astype(bool)
                if isinstance(m, Tensor)
                else np.asarray(m).ravel().astype(bool)
                for m in masks
            ]
        )
        num_images = len(score_maps)
    else:
        scores_arr = (
            score_maps.detach().cpu().numpy().ravel()
            if isinstance(score_maps, Tensor)
            else np.asarray(score_maps).ravel()
        )
        masks_arr = (
            masks.detach().cpu().numpy().ravel().astype(bool)
            if isinstance(masks, Tensor)
            else np.asarray(masks).ravel().astype(bool)
        )
        num_images = 1

    valid_mask = np.isfinite(scores_arr)
    y_score = scores_arr[valid_mask]
    y_true = masks_arr[valid_mask]

    num_pixels = len(y_true)
    if num_pixels == 0:
        return MetricResults(
            iou=0.0,
            average_precision=0.0,
            auroc=0.5,
            f1=0.0,
            precision=0.0,
            recall=0.0,
            balanced_accuracy=0.5,
            false_positive_rate=0.0,
            threshold=threshold,
            num_pixels=0,
            num_images=num_images,
        )

    if len(np.unique(y_true)) < 2:
        ap = 1.0 if np.all(y_true) else 0.0
        auroc = 0.5
    else:
        ap = float(average_precision_score(y_true, y_score))
        auroc = float(roc_auc_score(y_true, y_score))

    y_pred = y_score >= threshold

    tp = np.sum(y_pred & y_true)
    fp = np.sum(y_pred & ~y_true)
    tn = np.sum(~y_pred & ~y_true)
    fn = np.sum(~y_pred & y_true)

    precision = float(tp / (tp + fp + 1e-10))
    recall = float(tp / (tp + fn + 1e-10))
    f1 = float(2 * precision * recall / (precision + recall + 1e-10))
    iou = float(tp / (tp + fp + fn + 1e-10))

    fpr = float(fp / (fp + tn + 1e-10))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))

    return MetricResults(
        iou=iou,
        average_precision=ap,
        auroc=auroc,
        f1=f1,
        precision=precision,
        recall=recall,
        balanced_accuracy=balanced_acc,
        false_positive_rate=fpr,
        threshold=threshold,
        num_pixels=num_pixels,
        num_images=num_images,
    )


def compute_cluster_bootstrap_ci(
    sample_scores: Sequence[Tensor | np.ndarray],
    sample_masks: Sequence[Tensor | np.ndarray],
    scene_ids: Sequence[str],
    threshold: float,
    num_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    unique_scenes = np.unique(scene_ids)
    num_scenes = len(unique_scenes)

    if num_scenes == 0:
        return {}

    scene_to_indices: dict[str, list[int]] = {s: [] for s in unique_scenes}
    for idx, s_id in enumerate(scene_ids):
        scene_to_indices[s_id].append(idx)

    boot_metrics: dict[str, list[float]] = {
        "ap": [],
        "auroc": [],
        "f1": [],
        "iou": [],
        "fpr": [],
    }

    alpha = (1.0 - confidence_level) / 2.0

    for _ in range(num_resamples):
        sampled_scenes = rng.choice(unique_scenes, size=num_scenes, replace=True)
        sampled_indices = []
        for s_id in sampled_scenes:
            sampled_indices.extend(scene_to_indices[s_id])

        sub_scores = [sample_scores[i] for i in sampled_indices]
        sub_masks = [sample_masks[i] for i in sampled_indices]

        res = compute_binary_metrics(sub_scores, sub_masks, threshold)
        boot_metrics["ap"].append(res.average_precision)
        boot_metrics["auroc"].append(res.auroc)
        boot_metrics["f1"].append(res.f1)
        boot_metrics["iou"].append(res.iou)
        boot_metrics["fpr"].append(res.false_positive_rate)

    ci_results = {}
    for metric_name, values in boot_metrics.items():
        lower = float(np.percentile(values, alpha * 100))
        upper = float(np.percentile(values, (1.0 - alpha) * 100))
        ci_results[metric_name] = (lower, upper)

    return ci_results
