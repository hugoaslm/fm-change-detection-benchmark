"""Validation-only layer and anomaly-score selection."""

import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from fm_change_detection.cache import FeatureCache, compute_config_hash
from fm_change_detection.config import BenchmarkConfig
from fm_change_detection.data import (
    LEVIRCDDataset,
    compute_dataset_manifest_hash,
    validate_dataset_layout,
)
from fm_change_detection.encoders import get_encoder
from fm_change_detection.metrics import compute_binary_metrics
from fm_change_detection.pipeline import (
    _get_cached_pair,
    _iter_limited,
    _move_encoder,
    resolve_device,
)
from fm_change_detection.reporting import get_git_commit
from fm_change_detection.scoring import (
    FeatureStatsTracker,
    cosine_score,
    standardized_euclidean_score,
    upsample_score_map,
)
from fm_change_detection.thresholds import (
    compute_otsu_threshold,
    fit_calibrated_f1_threshold,
)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _fit_channel_std(
    train_dataset: LEVIRCDDataset,
    layer: str,
    encoder,
    cache: FeatureCache,
    config: BenchmarkConfig,
    device: torch.device,
) -> torch.Tensor:
    tracker = FeatureStatsTracker()
    for sample in _iter_limited(
        train_dataset, config.runtime.max_train_samples, config.dataset.seed
    ):
        features_t1, features_t2 = _get_cached_pair(
            sample, "train", layer, encoder, cache, config, device
        )
        tracker.update(features_t1)
        tracker.update(features_t2)
    return tracker.get_std()


def _evaluate_validation_candidate(
    validation_dataset: LEVIRCDDataset,
    layer: str,
    score_method: str,
    channel_std: torch.Tensor | None,
    encoder,
    cache: FeatureCache,
    config: BenchmarkConfig,
    device: torch.device,
) -> dict[str, Any]:
    score_maps = []
    masks = []
    started = time.time()
    for sample in _iter_limited(
        validation_dataset, config.runtime.max_val_samples, config.dataset.seed + 1
    ):
        features_t1, features_t2 = _get_cached_pair(
            sample, "val", layer, encoder, cache, config, device
        )
        if score_method == "standardized_euclidean":
            if channel_std is None:
                raise RuntimeError("Standardized Euclidean requires training feature statistics")
            score_map = standardized_euclidean_score(features_t1, features_t2, channel_std)
        else:
            score_map = cosine_score(features_t1, features_t2)
        score_maps.append(
            upsample_score_map(score_map, target_size=tuple(sample["change_mask"].shape))
            .squeeze(0)
            .cpu()
        )
        masks.append(sample["change_mask"])

    if not score_maps:
        raise RuntimeError("Validation selection received no samples")

    stacked_scores = torch.stack(score_maps)
    stacked_masks = torch.stack(masks)
    otsu_threshold = compute_otsu_threshold(stacked_scores)
    calibrated_threshold = fit_calibrated_f1_threshold(stacked_scores, stacked_masks)
    otsu_metrics = compute_binary_metrics(score_maps, masks, otsu_threshold)
    calibrated_metrics = compute_binary_metrics(score_maps, masks, calibrated_threshold)

    return {
        "score": score_method,
        "layer": layer,
        "num_images": calibrated_metrics.num_images,
        "num_pixels": calibrated_metrics.num_pixels,
        "average_precision": calibrated_metrics.average_precision,
        "auroc": calibrated_metrics.auroc,
        "calibrated_threshold": calibrated_threshold,
        "calibrated_f1": calibrated_metrics.f1,
        "calibrated_iou": calibrated_metrics.iou,
        "calibrated_precision": calibrated_metrics.precision,
        "calibrated_recall": calibrated_metrics.recall,
        "calibrated_fpr": calibrated_metrics.false_positive_rate,
        "otsu_threshold": otsu_threshold,
        "otsu_f1": otsu_metrics.f1,
        "otsu_iou": otsu_metrics.iou,
        "otsu_fpr": otsu_metrics.false_positive_rate,
        "runtime_seconds": time.time() - started,
    }


def _write_candidates_csv(path: Path, candidates: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not candidates:
        raise RuntimeError("No selection candidates were produced")
    fieldnames = list(candidates[0])
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidates)
    temporary.replace(path)


def _write_selection_report(
    path: Path, candidates: list[dict[str, Any]], winners: list[dict[str, Any]]
) -> None:
    winner_keys = {(row["encoder"], row["layer"], row["score"]) for row in winners}
    lines = [
        "# Validation-only representation selection",
        "",
        "Selection criterion: validation average precision (tie-breakers: AUROC, calibrated F1).",
        "The test split was not loaded or evaluated.",
        "",
        "| Selected | Encoder | Layer | Score | Val AP | Val AUROC | Val F1 | Val IoU |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    ordered = sorted(
        candidates,
        key=lambda row: (
            row["encoder"],
            -row["average_precision"],
            -row["auroc"],
            -row["calibrated_f1"],
        ),
    )
    for row in ordered:
        selected = "✓" if (row["encoder"], row["layer"], row["score"]) in winner_keys else ""
        lines.append(
            f"| {selected} | `{row['encoder']}` | `{row['layer']}` | `{row['score']}` | "
            f"{row['average_precision']:.4f} | {row['auroc']:.4f} | "
            f"{row['calibrated_f1']:.4f} | {row['calibrated_iou']:.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_final_config(config: BenchmarkConfig, winners: list[dict[str, Any]]) -> dict[str, Any]:
    source_encoders = {encoder.name: encoder for encoder in config.encoders}
    selected_encoders = []
    for winner in winners:
        source = source_encoders[winner["encoder"]]
        selected_encoders.append(
            {
                "name": source.name,
                "checkpoint": source.checkpoint,
                "layers": [winner["layer"]],
                "scores": [winner["score"]],
            }
        )

    return {
        "dataset": {
            "name": config.dataset.name,
            "root": config.dataset.root,
            "tile_size": config.dataset.tile_size,
            "input_size": config.dataset.input_size,
            "seed": config.dataset.seed,
        },
        "encoders": selected_encoders,
        "scoring": {"methods": ["cosine", "standardized_euclidean"]},
        "thresholds": {"methods": ["unlabeled", "calibrated"]},
        "bootstrap": {
            # The exact pixel-level cluster bootstrap is intentionally disabled
            # for the full test set. A scalable bootstrap is added after the
            # selected configuration is frozen.
            "num_resamples": 0,
            "confidence_level": 0.95,
            "seed": config.bootstrap.seed,
        },
        "runtime": {
            "device": "auto",
            "max_train_samples": None,
            "max_val_samples": None,
            "max_test_samples": None,
            "cache_dtype": config.runtime.cache_dtype,
        },
        "output_dir": "outputs/final_results",
        "cache_dir": config.cache_dir,
    }


def run_validation_selection(config: BenchmarkConfig) -> dict[str, Any]:
    """Select one layer and score per encoder without opening the test split."""
    root = config.dataset.root
    accessed_splits = ("train", "val")
    validate_dataset_layout(root, splits=accessed_splits)
    manifest_hash = compute_dataset_manifest_hash(root, splits=accessed_splits)
    train_dataset = LEVIRCDDataset(root, split="train", input_size=config.dataset.input_size)
    validation_dataset = LEVIRCDDataset(root, split="val", input_size=config.dataset.input_size)
    device = resolve_device(config.runtime.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[dict[str, Any]] = []
    for encoder_config in config.encoders:
        layers = tuple(encoder_config.layers)
        if not layers:
            raise ValueError(f"Encoder {encoder_config.name} has no selection layers")
        encoder = _move_encoder(
            get_encoder(
                encoder_config.name,
                checkpoint=encoder_config.checkpoint,
                layers=layers,
            ),
            device,
        )
        cache_hash = compute_config_hash(
            dataset_name=config.dataset.name,
            encoder_name=encoder_config.name,
            checkpoint=encoder.metadata.checkpoint,
            layers=layers,
            input_size=config.dataset.input_size,
            manifest_hash=manifest_hash,
            preprocessing={
                "mean": encoder.metadata.normalization_mean,
                "std": encoder.metadata.normalization_std,
                "cache_dtype": config.runtime.cache_dtype,
                "selection_splits": accessed_splits,
            },
        )
        cache = FeatureCache(config.cache_dir, config.dataset.name, cache_hash)
        cache.write_metadata(
            {
                "purpose": "validation_selection",
                "accessed_splits": accessed_splits,
                "manifest_hash": manifest_hash,
                "encoder": encoder.metadata.name,
                "checkpoint": encoder.metadata.checkpoint,
                "layers": layers,
                "input_size": config.dataset.input_size,
                "cache_dtype": config.runtime.cache_dtype,
            }
        )

        score_methods = encoder_config.scores or config.scoring.methods
        for layer in layers:
            channel_std = None
            if "standardized_euclidean" in score_methods:
                channel_std = _fit_channel_std(train_dataset, layer, encoder, cache, config, device)
            for score_method in score_methods:
                candidate = _evaluate_validation_candidate(
                    validation_dataset,
                    layer,
                    score_method,
                    channel_std,
                    encoder,
                    cache,
                    config,
                    device,
                )
                candidate.update(
                    {
                        "encoder": encoder_config.name,
                        "checkpoint": encoder.metadata.checkpoint,
                        "feature_stride": encoder.metadata.feature_strides[layer],
                        "cache_config_hash": cache_hash,
                    }
                )
                candidates.append(candidate)

        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()

    winners = []
    for encoder_config in config.encoders:
        encoder_candidates = [row for row in candidates if row["encoder"] == encoder_config.name]
        winners.append(
            max(
                encoder_candidates,
                key=lambda row: (
                    row["average_precision"],
                    row["auroc"],
                    row["calibrated_f1"],
                ),
            )
        )

    metadata = {
        "git_commit": get_git_commit(),
        "dataset": config.dataset.name,
        "manifest_hash": manifest_hash,
        "accessed_splits": accessed_splits,
        "selection_criterion": "average_precision",
        "tie_breakers": ["auroc", "calibrated_f1"],
        "max_train_samples": config.runtime.max_train_samples,
        "max_val_samples": config.runtime.max_val_samples,
        "device": str(device),
        "candidates": candidates,
        "selected": winners,
    }
    _atomic_json(output_dir / "selection.json", metadata)
    _write_candidates_csv(output_dir / "candidates.csv", candidates)
    _write_selection_report(output_dir / "selection.md", candidates, winners)

    final_config = _build_final_config(config, winners)
    final_config_path = output_dir / "final_selected.yaml"
    temporary = final_config_path.with_suffix(".yaml.tmp")
    temporary.write_text(yaml.safe_dump(final_config, sort_keys=False), encoding="utf-8")
    temporary.replace(final_config_path)

    return {
        "selected": winners,
        "selection_path": str(output_dir / "selection.json"),
        "report_path": str(output_dir / "selection.md"),
        "final_config_path": str(final_config_path),
        "config": asdict(config.runtime),
    }
