"""Configuration management for fm_change_detection benchmark."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    name: str = "synthetic"
    root: str = "outputs/synthetic_dataset"
    tile_size: int = 256
    input_size: int = 252
    seed: int = 42
    num_scenes: int = 4
    tiles_per_scene: int = 2


@dataclass
class EncoderConfig:
    name: str
    checkpoint: str = "none"
    layers: list[str] = field(default_factory=list)
    scores: list[str] | None = None


@dataclass
class ScoringConfig:
    methods: list[str] = field(default_factory=lambda: ["cosine"])


@dataclass
class ThresholdsConfig:
    methods: list[str] = field(default_factory=lambda: ["unlabeled", "calibrated"])


@dataclass
class BootstrapConfig:
    num_resamples: int = 1000
    confidence_level: float = 0.95
    seed: int = 42


@dataclass
class PerturbationConfig:
    name: str
    values: list[Any]


@dataclass
class SyntheticChangeConfig:
    """Controlled synthetic change grid for detectability-frontier runs."""

    magnitudes: list[float] = field(default_factory=lambda: [0.05, 0.10, 0.20, 0.40])
    area_fractions: list[float] = field(default_factory=lambda: [0.01, 0.04, 0.16])
    seed: int = 7


@dataclass
class RuntimeConfig:
    """Execution controls shared by local and notebook runs."""

    device: str = "auto"
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    max_test_samples: int | None = None
    cache_dtype: str = "float16"


@dataclass
class BenchmarkConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    encoders: list[EncoderConfig] = field(default_factory=list)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    perturbations: list[PerturbationConfig] = field(default_factory=list)
    synthetic_changes: SyntheticChangeConfig = field(default_factory=SyntheticChangeConfig)
    output_dir: str = "outputs/results"
    cache_dir: str = "outputs/cache"
    report_dir: str = "reports"
    raw_dict: dict[str, Any] = field(default_factory=dict)


def load_config(config_path: str | Path) -> BenchmarkConfig:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    ds_data = data.get("dataset", {})
    dataset_cfg = DatasetConfig(
        name=ds_data.get("name", "synthetic"),
        root=ds_data.get("root", "outputs/synthetic_dataset"),
        tile_size=ds_data.get("tile_size", 256),
        input_size=ds_data.get("input_size", 252),
        seed=ds_data.get("seed", 42),
        num_scenes=ds_data.get("num_scenes", 4),
        tiles_per_scene=ds_data.get("tiles_per_scene", 2),
    )

    encoder_cfgs = []
    if "encoder" in data:
        enc_data = data["encoder"]
        encoder_cfgs.append(
            EncoderConfig(
                name=enc_data.get("name", "mock_encoder"),
                checkpoint=enc_data.get("checkpoint", "none"),
                layers=enc_data.get("layers", []),
                scores=enc_data.get("scores"),
            )
        )
    elif "encoders" in data:
        for enc_data in data["encoders"]:
            encoder_cfgs.append(
                EncoderConfig(
                    name=enc_data.get("name"),
                    checkpoint=enc_data.get("checkpoint", "none"),
                    layers=enc_data.get("layers", []),
                    scores=enc_data.get("scores"),
                )
            )

    sc_data = data.get("scoring", {})
    if isinstance(sc_data, dict):
        if "methods" in sc_data:
            scoring_methods = sc_data["methods"]
        elif "method" in sc_data:
            scoring_methods = [sc_data["method"]]
        else:
            scoring_methods = ["cosine"]
    else:
        scoring_methods = ["cosine"]

    th_data = data.get("thresholds", {})
    if isinstance(th_data, dict):
        th_methods = th_data.get("methods", ["unlabeled", "calibrated"])
    else:
        th_methods = ["unlabeled", "calibrated"]

    bs_data = data.get("bootstrap", {})
    bootstrap_cfg = BootstrapConfig(
        num_resamples=bs_data.get("num_resamples", 1000),
        confidence_level=bs_data.get("confidence_level", 0.95),
        seed=bs_data.get("seed", 42),
    )

    pert_cfgs = []
    if "perturbations" in data:
        for p in data["perturbations"]:
            pert_cfgs.append(PerturbationConfig(name=p["name"], values=p["values"]))

    synth_data = data.get("synthetic_changes", {})
    synth_cfg = SyntheticChangeConfig(
        magnitudes=list(synth_data.get("magnitudes", [0.05, 0.10, 0.20, 0.40])),
        area_fractions=list(synth_data.get("area_fractions", [0.01, 0.04, 0.16])),
        seed=int(synth_data.get("seed", 7)),
    )

    runtime_data = data.get("runtime", {})
    runtime_cfg = RuntimeConfig(
        device=str(runtime_data.get("device", "auto")),
        max_train_samples=runtime_data.get("max_train_samples"),
        max_val_samples=runtime_data.get("max_val_samples"),
        max_test_samples=runtime_data.get("max_test_samples"),
        cache_dtype=str(runtime_data.get("cache_dtype", "float16")),
    )

    allowed_scores = {"cosine", "standardized_euclidean"}
    unknown_scores = set(scoring_methods) - allowed_scores
    if unknown_scores:
        raise ValueError(f"Unknown scoring methods: {sorted(unknown_scores)}")
    for encoder_cfg in encoder_cfgs:
        if encoder_cfg.scores is not None:
            unknown_encoder_scores = set(encoder_cfg.scores) - allowed_scores
            if unknown_encoder_scores:
                raise ValueError(
                    f"Unknown scoring methods for {encoder_cfg.name}: "
                    f"{sorted(unknown_encoder_scores)}"
                )
    allowed_thresholds = {"unlabeled", "calibrated"}
    unknown_thresholds = set(th_methods) - allowed_thresholds
    if unknown_thresholds:
        raise ValueError(f"Unknown threshold methods: {sorted(unknown_thresholds)}")

    return BenchmarkConfig(
        dataset=dataset_cfg,
        encoders=encoder_cfgs,
        scoring=ScoringConfig(methods=scoring_methods),
        thresholds=ThresholdsConfig(methods=th_methods),
        bootstrap=bootstrap_cfg,
        runtime=runtime_cfg,
        perturbations=pert_cfgs,
        synthetic_changes=synth_cfg,
        output_dir=data.get("output_dir", "outputs/results"),
        cache_dir=data.get("cache_dir", "outputs/cache"),
        report_dir=data.get("report_dir", "reports"),
        raw_dict=data,
    )
