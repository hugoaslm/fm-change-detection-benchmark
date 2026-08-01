import random
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from fm_change_detection.cache import FeatureCache, compute_config_hash
from fm_change_detection.changes import (
    apply_additive_change,
    pick_change_region,
    synthetic_change_mask,
)
from fm_change_detection.config import BenchmarkConfig
from fm_change_detection.data import (
    ChangeSample,
    LEVIRCDDataset,
    compute_dataset_manifest_hash,
    generate_synthetic_dataset,
    validate_dataset_layout,
)
from fm_change_detection.encoders import get_encoder
from fm_change_detection.features import extract_pair_features, preprocess_images_for_encoder
from fm_change_detection.metrics import (
    compute_binary_metrics,
    compute_cluster_bootstrap_ci,
)
from fm_change_detection.reporting import (
    generate_benchmark_report,
    generate_frontier_report,
    save_result_record,
)
from fm_change_detection.robustness import apply_perturbation
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


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available")
    return device


def _iter_limited(dataset: LEVIRCDDataset, limit: int | None, seed: int) -> Iterator[ChangeSample]:
    indices = list(range(len(dataset)))
    if limit is not None and limit < len(indices):
        indices = sorted(random.Random(seed).sample(indices, limit))
    for index in indices:
        yield dataset[index]


def _move_encoder(encoder, device: torch.device):
    if isinstance(encoder, nn.Module):
        encoder.to(device)
        encoder.eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
    return encoder


def _get_cached_pair(
    sample: ChangeSample,
    split: str,
    layer_name: str,
    encoder,
    cache: FeatureCache,
    config: BenchmarkConfig,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    sample_key = f"{split}__{sample['sample_id']}"
    t1_key, t2_key = f"{sample_key}__t1", f"{sample_key}__t2"
    cached_t1 = cache.get(t1_key, layer_name)
    cached_t2 = cache.get(t2_key, layer_name)
    if cached_t1 is not None and cached_t2 is not None:
        return cached_t1.unsqueeze(0).float(), cached_t2.unsqueeze(0).float()

    t1 = sample["image_t1"].unsqueeze(0).to(device)
    t2 = sample["image_t2"].unsqueeze(0).to(device)
    f1, f2 = extract_pair_features(encoder, t1, t2, input_size=config.dataset.input_size)

    cache.save_sample_features(t1_key, f1, dtype=config.runtime.cache_dtype)
    cache.save_sample_features(t2_key, f2, dtype=config.runtime.cache_dtype)
    return f1[layer_name].detach().cpu().float(), f2[layer_name].detach().cpu().float()


def _preload_features_batched(
    samples: list[ChangeSample],
    split: str,
    layer_name: str,
    encoder,
    cache: FeatureCache,
    config: BenchmarkConfig,
    device: torch.device,
) -> list[tuple[Tensor, Tensor]]:
    batch_size = config.runtime.frontier_batch_size
    features: list[tuple[Tensor, Tensor] | None] = [None] * len(samples)
    pending_indices: list[int] = []
    for i, sample in enumerate(samples):
        sample_key = f"{split}__{sample['sample_id']}"
        t1_key, t2_key = f"{sample_key}__t1", f"{sample_key}__t2"
        if cache.is_cached(t1_key, layer_name) and cache.is_cached(t2_key, layer_name):
            features[i] = (
                cache.get(t1_key, layer_name).float(),
                cache.get(t2_key, layer_name).float(),
            )
        else:
            pending_indices.append(i)

    for start in range(0, len(pending_indices), batch_size):
        idx_chunk = pending_indices[start : start + batch_size]
        chunk = [samples[i] for i in idx_chunk]
        t1_batch = torch.stack([s["image_t1"] for s in chunk]).to(device)
        t2_batch = torch.stack([s["image_t2"] for s in chunk]).to(device)
        f1_all, f2_all = extract_pair_features(
            encoder, t1_batch, t2_batch, input_size=config.dataset.input_size
        )
        for j, sample in enumerate(chunk):
            sample_key = f"{split}__{sample['sample_id']}"
            cache.save_sample_features(
                f"{sample_key}__t1",
                {ln: f1_all[ln][j] for ln in f1_all},
                dtype=config.runtime.cache_dtype,
            )
            cache.save_sample_features(
                f"{sample_key}__t2",
                {ln: f2_all[ln][j] for ln in f2_all},
                dtype=config.runtime.cache_dtype,
            )
            features[idx_chunk[j]] = (
                f1_all[layer_name][j].detach().cpu().float(),
                f2_all[layer_name][j].detach().cpu().float(),
            )

    result: list[tuple[Tensor, Tensor]] = []
    for item in features:
        if item is None:
            raise RuntimeError("Feature preload failed to cover a sample")
        result.append(item)
    return result


def run_smoke_test(config: BenchmarkConfig) -> dict:
    print("[SMOKE] Generating synthetic dataset...")
    syn_root = generate_synthetic_dataset(
        output_root=config.dataset.root,
        num_scenes=config.dataset.num_scenes,
        tiles_per_scene=config.dataset.tiles_per_scene,
        image_size=config.dataset.tile_size,
        seed=config.dataset.seed,
    )

    validate_dataset_layout(syn_root)

    print("[SMOKE] Loading datasets...")
    val_dataset = LEVIRCDDataset(syn_root, split="val", input_size=config.dataset.input_size)
    test_dataset = LEVIRCDDataset(syn_root, split="test", input_size=config.dataset.input_size)

    device = resolve_device(config.runtime.device)
    encoder = _move_encoder(get_encoder("mock_encoder", layers=("layer1", "layer2")), device)

    t0 = time.time()
    val_scores = []
    val_masks = []
    for sample in val_dataset:
        t1 = sample["image_t1"].unsqueeze(0).to(device)
        t2 = sample["image_t2"].unsqueeze(0).to(device)
        f1, f2 = extract_pair_features(encoder, t1, t2, input_size=config.dataset.input_size)

        s_map = cosine_score(f1["layer2"], f2["layer2"])
        s_upsampled = upsample_score_map(
            s_map, target_size=(config.dataset.tile_size, config.dataset.tile_size)
        )
        val_scores.append(s_upsampled.squeeze(0).cpu())
        val_masks.append(sample["change_mask"])

    calib_th = fit_calibrated_f1_threshold(torch.stack(val_scores), torch.stack(val_masks))

    test_scores = []
    test_masks = []
    test_scenes = []
    for sample in test_dataset:
        t1 = sample["image_t1"].unsqueeze(0).to(device)
        t2 = sample["image_t2"].unsqueeze(0).to(device)
        f1, f2 = extract_pair_features(encoder, t1, t2, input_size=config.dataset.input_size)

        s_map = cosine_score(f1["layer2"], f2["layer2"])
        s_upsampled = upsample_score_map(
            s_map, target_size=(config.dataset.tile_size, config.dataset.tile_size)
        )
        test_scores.append(s_upsampled.squeeze(0).cpu())
        test_masks.append(sample["change_mask"])
        test_scenes.append(sample["scene_id"])

    val_time = time.time() - t0

    metrics_calib = compute_binary_metrics(test_scores, test_masks, calib_th)
    metrics_calib.ci_95 = compute_cluster_bootstrap_ci(
        test_scores, test_masks, test_scenes, calib_th, num_resamples=config.bootstrap.num_resamples
    )

    run_id = f"smoke_{uuid.uuid4().hex[:8]}"
    saved_path = save_result_record(
        results_dir=config.output_dir,
        run_id=run_id,
        dataset="synthetic",
        manifest_hash="smoke_hash",
        encoder="mock_encoder",
        checkpoint="none",
        layer="layer2",
        score_method="cosine",
        threshold_method="calibrated",
        seed=config.dataset.seed,
        metrics=metrics_calib,
        runtime_seconds=val_time,
    )

    print(f"[SMOKE] Smoke test completed cleanly! Saved output record to {saved_path}")
    return {"run_id": run_id, "metrics": metrics_calib, "result_path": str(saved_path)}


def run_single_evaluation(
    config: BenchmarkConfig,
    encoder_name: str,
    layer_name: str,
    score_method: str = "cosine",
) -> dict:
    print(f"[EVAL] Evaluating encoder={encoder_name}, layer={layer_name}, score={score_method}...")
    dataset_root = config.dataset.root
    validate_dataset_layout(dataset_root)
    manifest_hash = compute_dataset_manifest_hash(dataset_root)

    val_ds = LEVIRCDDataset(dataset_root, split="val", input_size=config.dataset.input_size)
    test_ds = LEVIRCDDataset(dataset_root, split="test", input_size=config.dataset.input_size)

    enc_cfg = next((e for e in config.encoders if e.name == encoder_name), None)
    ckpt = enc_cfg.checkpoint if enc_cfg else "none"

    device = resolve_device(config.runtime.device)
    encoder = _move_encoder(
        get_encoder(encoder_name, checkpoint=ckpt, layers=(layer_name,)), device
    )
    print(f"[EVAL] device={device}")

    cfg_hash = compute_config_hash(
        dataset_name=config.dataset.name,
        encoder_name=encoder_name,
        checkpoint=ckpt,
        layers=(layer_name,),
        input_size=config.dataset.input_size,
        manifest_hash=manifest_hash,
        preprocessing={
            "mean": encoder.metadata.normalization_mean,
            "std": encoder.metadata.normalization_std,
            "cache_dtype": config.runtime.cache_dtype,
        },
    )
    cache = FeatureCache(config.cache_dir, config.dataset.name, cfg_hash)
    cache.write_metadata(
        {
            "manifest_hash": manifest_hash,
            "encoder": encoder.metadata.name,
            "checkpoint": encoder.metadata.checkpoint,
            "layer": layer_name,
            "input_size": config.dataset.input_size,
            "normalization_mean": encoder.metadata.normalization_mean,
            "normalization_std": encoder.metadata.normalization_std,
            "cache_dtype": config.runtime.cache_dtype,
        }
    )

    channel_std = None
    if score_method == "standardized_euclidean":
        print("[EVAL] Fitting training channel statistics for Standardized Euclidean...")
        train_ds = LEVIRCDDataset(dataset_root, split="train", input_size=config.dataset.input_size)
        tracker = FeatureStatsTracker()
        for sample in _iter_limited(
            train_ds, config.runtime.max_train_samples, config.dataset.seed
        ):
            f1_layer, f2_layer = _get_cached_pair(
                sample, "train", layer_name, encoder, cache, config, device
            )
            tracker.update(f1_layer)
            tracker.update(f2_layer)
        channel_std = tracker.get_std()

    t0 = time.time()

    val_scores, val_masks = [], []
    for sample in _iter_limited(val_ds, config.runtime.max_val_samples, config.dataset.seed + 1):
        f1_layer, f2_layer = _get_cached_pair(
            sample, "val", layer_name, encoder, cache, config, device
        )

        if score_method == "standardized_euclidean":
            s_map = standardized_euclidean_score(f1_layer, f2_layer, channel_std)
        else:
            s_map = cosine_score(f1_layer, f2_layer)

        s_up = upsample_score_map(s_map, target_size=tuple(sample["change_mask"].shape))
        val_scores.append(s_up.squeeze(0))
        val_masks.append(sample["change_mask"])

    val_scores_t = torch.stack(val_scores)
    val_masks_t = torch.stack(val_masks)

    otsu_th = compute_otsu_threshold(val_scores_t)
    calib_th = fit_calibrated_f1_threshold(val_scores_t, val_masks_t)

    test_scores, test_masks, test_scenes = [], [], []
    for sample in _iter_limited(test_ds, config.runtime.max_test_samples, config.dataset.seed + 2):
        f1_layer, f2_layer = _get_cached_pair(
            sample, "test", layer_name, encoder, cache, config, device
        )

        if score_method == "standardized_euclidean":
            s_map = standardized_euclidean_score(f1_layer, f2_layer, channel_std)
        else:
            s_map = cosine_score(f1_layer, f2_layer)

        s_up = upsample_score_map(s_map, target_size=tuple(sample["change_mask"].shape))
        test_scores.append(s_up.squeeze(0))
        test_masks.append(sample["change_mask"])
        test_scenes.append(sample["scene_id"])

    elapsed = time.time() - t0

    res_otsu = compute_binary_metrics(test_scores, test_masks, otsu_th)
    res_calib = compute_binary_metrics(test_scores, test_masks, calib_th)
    if config.bootstrap.num_resamples > 0:
        res_calib.ci_95 = compute_cluster_bootstrap_ci(
            test_scores,
            test_masks,
            test_scenes,
            calib_th,
            num_resamples=config.bootstrap.num_resamples,
            confidence_level=config.bootstrap.confidence_level,
            seed=config.bootstrap.seed,
        )

    result_paths = {}
    for threshold_method, metrics in (("unlabeled", res_otsu), ("calibrated", res_calib)):
        if threshold_method not in config.thresholds.methods:
            continue
        run_id = f"{encoder_name}_{layer_name}_{score_method}_{threshold_method}"
        result_paths[threshold_method] = str(
            save_result_record(
                results_dir=config.output_dir,
                run_id=run_id,
                dataset=config.dataset.name,
                manifest_hash=manifest_hash,
                encoder=encoder_name,
                checkpoint=encoder.metadata.checkpoint,
                layer=layer_name,
                score_method=score_method,
                threshold_method=threshold_method,
                seed=config.dataset.seed,
                metrics=metrics,
                runtime_seconds=elapsed,
                additional_fields={
                    "cache_config_hash": cfg_hash,
                    "device": str(device),
                    "max_train_samples": config.runtime.max_train_samples,
                    "max_val_samples": config.runtime.max_val_samples,
                    "max_test_samples": config.runtime.max_test_samples,
                },
            )
        )

    return {
        "calibrated_metrics": res_calib,
        "otsu_metrics": res_otsu,
        "result_paths": result_paths,
    }


def run_benchmark(config: BenchmarkConfig) -> list[dict]:
    results = []
    for enc in config.encoders:
        for layer in enc.layers:
            for score in enc.scores or config.scoring.methods:
                res = run_single_evaluation(config, enc.name, layer, score_method=score)
                results.append(res)

    generate_benchmark_report(config.output_dir, "reports/benchmark.md")
    return results


def run_robustness(config: BenchmarkConfig) -> list[dict]:
    print("[ROBUSTNESS] Running robustness evaluation...")
    dataset_root = config.dataset.root
    validate_dataset_layout(dataset_root)
    manifest_hash = compute_dataset_manifest_hash(dataset_root)
    val_ds = LEVIRCDDataset(dataset_root, split="val", input_size=config.dataset.input_size)
    test_ds = LEVIRCDDataset(dataset_root, split="test", input_size=config.dataset.input_size)
    device = resolve_device(config.runtime.device)

    results = []
    for enc_cfg in config.encoders:
        enc_name = enc_cfg.name
        layer = enc_cfg.layers[0] if enc_cfg.layers else "layer4"
        encoder = _move_encoder(
            get_encoder(enc_name, checkpoint=enc_cfg.checkpoint, layers=(layer,)), device
        )
        cfg_hash = compute_config_hash(
            dataset_name=config.dataset.name,
            encoder_name=enc_name,
            checkpoint=encoder.metadata.checkpoint,
            layers=(layer,),
            input_size=config.dataset.input_size,
            manifest_hash=manifest_hash,
            preprocessing={
                "mean": encoder.metadata.normalization_mean,
                "std": encoder.metadata.normalization_std,
                "cache_dtype": config.runtime.cache_dtype,
            },
        )
        cache = FeatureCache(config.cache_dir, config.dataset.name, cfg_hash)

        val_scores, val_masks = [], []
        for sample in _iter_limited(
            val_ds, config.runtime.max_val_samples, config.dataset.seed + 1
        ):
            f1_layer, f2_layer = _get_cached_pair(
                sample, "val", layer, encoder, cache, config, device
            )
            s_map = cosine_score(f1_layer, f2_layer)
            s_up = upsample_score_map(s_map, target_size=tuple(sample["change_mask"].shape))
            val_scores.append(s_up.squeeze(0))
            val_masks.append(sample["change_mask"])

        clean_th = fit_calibrated_f1_threshold(torch.stack(val_scores), torch.stack(val_masks))

        clean_scores, test_masks, test_scenes = [], [], []
        test_samples = list(
            _iter_limited(test_ds, config.runtime.max_test_samples, config.dataset.seed + 2)
        )
        for sample in test_samples:
            f1_layer, f2_layer = _get_cached_pair(
                sample, "test", layer, encoder, cache, config, device
            )
            s_map = cosine_score(f1_layer, f2_layer)
            s_up = upsample_score_map(s_map, target_size=tuple(sample["change_mask"].shape))
            clean_scores.append(s_up.squeeze(0))
            test_masks.append(sample["change_mask"])
            test_scenes.append(sample["scene_id"])

        clean_metrics = compute_binary_metrics(clean_scores, test_masks, clean_th)

        for pert_cfg in config.perturbations:
            for val in pert_cfg.values:
                p_scores = []
                started = time.time()
                for sample_index, sample in enumerate(test_samples):
                    f1_layer, _ = _get_cached_pair(
                        sample, "test", layer, encoder, cache, config, device
                    )
                    t2_pert = (
                        apply_perturbation(
                            sample["image_t2"],
                            pert_cfg.name,
                            val,
                            seed=config.dataset.seed + sample_index,
                        )
                        .unsqueeze(0)
                        .to(device)
                    )
                    t2_input = preprocess_images_for_encoder(
                        t2_pert, input_size=config.dataset.input_size
                    )
                    f2_layer = encoder.encode(t2_input)[layer].detach().cpu().float()
                    s_map = cosine_score(f1_layer, f2_layer)
                    s_up = upsample_score_map(s_map, target_size=tuple(sample["change_mask"].shape))
                    p_scores.append(s_up.squeeze(0).cpu())

                pert_metrics = compute_binary_metrics(p_scores, test_masks, clean_th)
                ap_drop = clean_metrics.average_precision - pert_metrics.average_precision
                f1_drop = clean_metrics.f1 - pert_metrics.f1
                fpr_increase = pert_metrics.false_positive_rate - clean_metrics.false_positive_rate

                run_id = f"robustness_{enc_name}_{pert_cfg.name}_{val}"
                saved = save_result_record(
                    results_dir=config.output_dir,
                    run_id=run_id,
                    dataset=config.dataset.name,
                    manifest_hash=manifest_hash,
                    encoder=enc_name,
                    checkpoint=encoder.metadata.checkpoint,
                    layer=layer,
                    score_method="cosine",
                    threshold_method="calibrated_frozen",
                    seed=config.dataset.seed,
                    metrics=pert_metrics,
                    runtime_seconds=time.time() - started,
                    additional_fields={
                        "perturbation": pert_cfg.name,
                        "perturbation_value": val,
                        "clean_ap": clean_metrics.average_precision,
                        "clean_f1": clean_metrics.f1,
                        "ap_drop": ap_drop,
                        "f1_drop": f1_drop,
                        "fpr_increase": fpr_increase,
                        "cache_config_hash": cfg_hash,
                        "device": str(device),
                        "max_val_samples": config.runtime.max_val_samples,
                        "max_test_samples": config.runtime.max_test_samples,
                    },
                )
                results.append({"run_id": run_id, "saved": str(saved)})

    return results


def run_detectability(config: BenchmarkConfig) -> dict:
    print("[FRONTIER] Running controlled-change detectability frontier...")
    dataset_root = config.dataset.root
    validate_dataset_layout(dataset_root)
    manifest_hash = compute_dataset_manifest_hash(dataset_root)
    val_ds = LEVIRCDDataset(dataset_root, split="val", input_size=config.dataset.input_size)
    test_ds = LEVIRCDDataset(dataset_root, split="test", input_size=config.dataset.input_size)
    device = resolve_device(config.runtime.device)

    synth = config.synthetic_changes
    val_samples = list(
        _iter_limited(val_ds, config.runtime.max_val_samples, config.dataset.seed + 1)
    )
    test_samples = list(
        _iter_limited(test_ds, config.runtime.max_test_samples, config.dataset.seed + 2)
    )
    if not test_samples:
        raise RuntimeError("Frontier evaluation received no test samples")

    frontier_records = []
    for enc_cfg in config.encoders:
        enc_name = enc_cfg.name
        layer = enc_cfg.layers[0] if enc_cfg.layers else "layer4"
        encoder = _move_encoder(
            get_encoder(enc_name, checkpoint=enc_cfg.checkpoint, layers=(layer,)), device
        )
        cfg_hash = compute_config_hash(
            dataset_name=config.dataset.name,
            encoder_name=enc_name,
            checkpoint=encoder.metadata.checkpoint,
            layers=(layer,),
            input_size=config.dataset.input_size,
            manifest_hash=manifest_hash,
            preprocessing={
                "mean": encoder.metadata.normalization_mean,
                "std": encoder.metadata.normalization_std,
                "cache_dtype": config.runtime.cache_dtype,
            },
        )
        cache = FeatureCache(config.cache_dir, config.dataset.name, cfg_hash)

        val_features = _preload_features_batched(
            val_samples, "val", layer, encoder, cache, config, device
        )
        val_scores, val_masks = [], []
        for sample, (f1_layer, f2_layer) in zip(val_samples, val_features):
            s_map = cosine_score(f1_layer.unsqueeze(0), f2_layer.unsqueeze(0))
            s_up = upsample_score_map(s_map, target_size=tuple(sample["change_mask"].shape))
            val_scores.append(s_up.squeeze(0))
            val_masks.append(sample["change_mask"])

        val_stacked = torch.stack(val_scores)
        val_mask_stacked = torch.stack(val_masks)
        calib_th = fit_calibrated_f1_threshold(val_stacked, val_mask_stacked)
        otsu_th = compute_otsu_threshold(val_stacked)

        test_features = _preload_features_batched(
            test_samples, "test", layer, encoder, cache, config, device
        )
        test_t1_by_id = {
            sample["sample_id"]: t1_layer
            for sample, (t1_layer, _t2) in zip(test_samples, test_features)
        }

        for area in synth.area_fractions:
            region_seed = (hash((synth.seed, area)) % (2**31)) & 0x7FFFFFFF
            usable = []
            for s_index, sample in enumerate(test_samples):
                rng = np.random.default_rng(region_seed + s_index)
                region = pick_change_region(sample["change_mask"], area, rng)
                if region is not None:
                    usable.append((sample, region))
            if not usable:
                print(f"[FRONTIER] No usable no-change region for {enc_name} area={area}; skipping")
                continue

            t1_features = [test_t1_by_id[sample["sample_id"]] for sample, _region in usable]

            batch_size = config.runtime.frontier_batch_size
            for mag in synth.magnitudes:
                scores, masks = [], []
                started = time.time()
                for start in range(0, len(usable), batch_size):
                    chunk = usable[start : start + batch_size]
                    t1_batch = torch.stack(
                        [t1_features[i] for i in range(start, start + len(chunk))]
                    )
                    t2_batch = torch.stack(
                        [
                            apply_additive_change(sample["image_t2"], region, mag)
                            for sample, region in chunk
                        ]
                    ).to(device)
                    t2_input = preprocess_images_for_encoder(
                        t2_batch, input_size=config.dataset.input_size
                    )
                    f2_batch = encoder.encode(t2_input)[layer].detach().cpu().float()
                    s_map = cosine_score(t1_batch, f2_batch)
                    s_up = upsample_score_map(
                        s_map, target_size=tuple(chunk[0][0]["change_mask"].shape)
                    )
                    for j, (sample, region) in enumerate(chunk):
                        scores.append(s_up[j].cpu())
                        masks.append(synthetic_change_mask(region, sample["change_mask"]))

                metrics_calib = compute_binary_metrics(scores, masks, calib_th)
                metrics_otsu = compute_binary_metrics(scores, masks, otsu_th)

                run_id = f"frontier_{enc_name}_{layer}_{mag}_{area}"
                save_result_record(
                    results_dir=config.output_dir,
                    run_id=run_id,
                    dataset=config.dataset.name,
                    manifest_hash=manifest_hash,
                    encoder=enc_name,
                    checkpoint=encoder.metadata.checkpoint,
                    layer=layer,
                    score_method="cosine",
                    threshold_method="calibrated_frozen",
                    seed=config.dataset.seed,
                    metrics=metrics_calib,
                    runtime_seconds=time.time() - started,
                    additional_fields={
                        "synthetic_magnitude": mag,
                        "synthetic_area_fraction": area,
                        "calibration_threshold": calib_th,
                        "otsu_threshold": otsu_th,
                        "synthetic_samples": metrics_calib.num_images,
                        "cache_config_hash": cfg_hash,
                        "device": str(device),
                        "max_val_samples": config.runtime.max_val_samples,
                        "max_test_samples": config.runtime.max_test_samples,
                    },
                )

                frontier_records.append(
                    {
                        "encoder": enc_name,
                        "layer": layer,
                        "checkpoint": encoder.metadata.checkpoint,
                        "magnitude": mag,
                        "area_fraction": area,
                        "num_samples": metrics_calib.num_images,
                        "num_pixels": metrics_calib.num_pixels,
                        "ap": metrics_calib.average_precision,
                        "auroc": metrics_calib.auroc,
                        "calibrated_f1": metrics_calib.f1,
                        "calibrated_iou": metrics_calib.iou,
                        "calibrated_fpr": metrics_calib.false_positive_rate,
                        "otsu_f1": metrics_otsu.f1,
                        "calibration_threshold": calib_th,
                        "otsu_threshold": otsu_th,
                    }
                )
                print(
                    f"[FRONTIER] {enc_name} area={area:.3f} mag={mag:.3f} AP={metrics_calib.average_precision:.4f}"
                )

    generate_frontier_report(
        frontier_records,
        results_dir=config.output_dir,
        report_path=str(Path(config.report_dir) / "frontier.md"),
        figures_dir=str(Path(config.report_dir) / "figures"),
    )
    return {
        "records": frontier_records,
        "report_path": str(Path(config.report_dir) / "frontier.md"),
        "summary_path": str(Path(config.output_dir) / "frontier.csv"),
    }
