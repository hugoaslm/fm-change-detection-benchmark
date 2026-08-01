import json
from pathlib import Path

import yaml

from fm_change_detection.config import (
    BenchmarkConfig,
    BootstrapConfig,
    DatasetConfig,
    EncoderConfig,
    RuntimeConfig,
    ScoringConfig,
)
from fm_change_detection.selection import run_validation_selection


def test_selection_never_requires_test_split(tmp_dir, synthetic_dataset_dir):
    (synthetic_dataset_dir / "list" / "test.txt").unlink()
    output_dir = tmp_dir / "selection"
    config = BenchmarkConfig(
        dataset=DatasetConfig(
            name="synthetic",
            root=str(synthetic_dataset_dir),
            tile_size=64,
            input_size=64,
            seed=42,
        ),
        encoders=[
            EncoderConfig(
                name="mock_encoder",
                layers=["layer1", "layer2"],
                scores=["cosine", "standardized_euclidean"],
            )
        ],
        scoring=ScoringConfig(methods=["cosine", "standardized_euclidean"]),
        bootstrap=BootstrapConfig(num_resamples=0),
        runtime=RuntimeConfig(
            device="cpu",
            max_train_samples=2,
            max_val_samples=2,
            max_test_samples=0,
            cache_dtype="float16",
        ),
        output_dir=str(output_dir),
        cache_dir=str(tmp_dir / "cache"),
    )

    result = run_validation_selection(config)

    assert Path(result["selection_path"]).exists()
    assert Path(result["report_path"]).exists()
    assert Path(result["final_config_path"]).exists()
    assert len(result["selected"]) == 1

    metadata = json.loads((output_dir / "selection.json").read_text(encoding="utf-8"))
    assert metadata["accessed_splits"] == ["train", "val"]
    assert len(metadata["candidates"]) == 4
    assert {row["score"] for row in metadata["candidates"]} == {
        "cosine",
        "standardized_euclidean",
    }

    final_config = yaml.safe_load((output_dir / "final_selected.yaml").read_text(encoding="utf-8"))
    assert len(final_config["encoders"]) == 1
    assert len(final_config["encoders"][0]["layers"]) == 1
    assert len(final_config["encoders"][0]["scores"]) == 1
    assert final_config["bootstrap"]["num_resamples"] == 0
    assert final_config["runtime"]["max_test_samples"] is None
