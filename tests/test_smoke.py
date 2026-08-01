from fm_change_detection.config import (
    BenchmarkConfig,
    BootstrapConfig,
    DatasetConfig,
    EncoderConfig,
    RuntimeConfig,
)
from fm_change_detection.pipeline import run_single_evaluation, run_smoke_test


def test_smoke_pipeline_cpu(tmp_dir):
    cfg = BenchmarkConfig(
        dataset=DatasetConfig(
            name="synthetic",
            root=str(tmp_dir / "syn_data"),
            num_scenes=3,
            tiles_per_scene=2,
            tile_size=64,
            input_size=64,
            seed=42,
        ),
        bootstrap=BootstrapConfig(num_resamples=20, seed=42),
        output_dir=str(tmp_dir / "results"),
        cache_dir=str(tmp_dir / "cache"),
    )

    res = run_smoke_test(cfg)
    assert "run_id" in res
    assert res["metrics"].iou >= 0.0
    assert (tmp_dir / "results" / f"{res['run_id']}.json").exists()


def test_cached_rgb_evaluation_saves_both_threshold_protocols(tmp_dir, synthetic_dataset_dir):
    cfg = BenchmarkConfig(
        dataset=DatasetConfig(
            name="synthetic",
            root=str(synthetic_dataset_dir),
            tile_size=64,
            input_size=64,
            seed=42,
        ),
        encoders=[EncoderConfig(name="rgb_pixels", layers=["input"])],
        bootstrap=BootstrapConfig(num_resamples=5, seed=42),
        runtime=RuntimeConfig(device="cpu", cache_dtype="float16"),
        output_dir=str(tmp_dir / "results"),
        cache_dir=str(tmp_dir / "cache"),
    )

    result = run_single_evaluation(cfg, "rgb_pixels", "input", "cosine")

    assert set(result["result_paths"]) == {"unlabeled", "calibrated"}
    assert all(
        (tmp_dir / "results" / f"rgb_pixels_input_cosine_{name}.json").exists()
        for name in ("unlabeled", "calibrated")
    )
    assert list((tmp_dir / "cache").rglob("metadata.json"))
