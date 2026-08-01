"""End-to-end tests for the controlled-change detectability frontier."""

from fm_change_detection.config import (
    BenchmarkConfig,
    DatasetConfig,
    EncoderConfig,
    RuntimeConfig,
    SyntheticChangeConfig,
)
from fm_change_detection.pipeline import run_detectability


def _frontier_config(tmp_dir, synthetic_dataset_dir):
    return BenchmarkConfig(
        dataset=DatasetConfig(
            name="synthetic",
            root=str(synthetic_dataset_dir),
            tile_size=64,
            input_size=64,
            seed=42,
        ),
        encoders=[EncoderConfig(name="mock_encoder", layers=["layer2"])],
        runtime=RuntimeConfig(device="cpu", cache_dtype="float32"),
        synthetic_changes=SyntheticChangeConfig(
            magnitudes=[0.1, 0.3],
            area_fractions=[0.01, 0.04],
            seed=7,
        ),
        output_dir=str(tmp_dir / "results"),
        cache_dir=str(tmp_dir / "cache"),
        report_dir=str(tmp_dir / "reports"),
    )


def test_frontier_produces_records_and_artifacts(tmp_dir, synthetic_dataset_dir):
    cfg = _frontier_config(tmp_dir, synthetic_dataset_dir)
    result = run_detectability(cfg)

    records = result["records"]
    assert len(records) == 2 * 2  # 2 magnitudes x 2 area fractions
    for record in records:
        assert 0.0 <= record["ap"] <= 1.0
        assert record["num_samples"] >= 1

    assert (tmp_dir / "results" / "frontier.csv").exists()
    assert (tmp_dir / "reports" / "frontier.md").exists()
    assert (tmp_dir / "reports" / "figures" / "frontier_mock_encoder.png").exists()
    assert list((tmp_dir / "results").glob("frontier_*.json"))


def test_larger_magnitude_is_more_detectable(tmp_dir, synthetic_dataset_dir):
    cfg = _frontier_config(tmp_dir, synthetic_dataset_dir)
    result = run_detectability(cfg)

    by_key = {(r["magnitude"], r["area_fraction"]): r for r in result["records"]}
    faint = by_key[(0.1, 0.04)]["ap"]
    strong = by_key[(0.3, 0.04)]["ap"]
    assert strong >= faint
