"""Resumable, hashed feature cache for extraction runs."""

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

CACHE_SCHEMA_VERSION = "1.0"


def compute_config_hash(
    dataset_name: str,
    encoder_name: str,
    checkpoint: str,
    layers: tuple[str, ...],
    input_size: int,
    manifest_hash: str = "unknown",
    preprocessing: dict[str, Any] | None = None,
    additional_metadata: dict[str, Any] | None = None,
) -> str:
    """Compute a deterministic SHA-256 hash string for caching extracted features."""
    payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "encoder_name": encoder_name,
        "checkpoint": checkpoint,
        "layers": sorted(layers),
        "input_size": input_size,
        "manifest_hash": manifest_hash,
        "preprocessing": preprocessing or {},
        "additional": additional_metadata or {},
    }
    dumped = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()[:16]


class FeatureCache:
    """Manages cached spatial feature maps per sample on disk."""

    def __init__(self, cache_root: str | Path, dataset_name: str, config_hash: str) -> None:
        self.cache_dir = Path(cache_root) / dataset_name / config_hash
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.cache_dir / "metadata.json"

    def is_cached(self, sample_id: str, layer: str) -> bool:
        """Check if feature map for sample_id and layer exists."""
        p = self.cache_dir / f"{sample_id}_{layer}.pt"
        return p.exists()

    def get(self, sample_id: str, layer: str) -> Tensor | None:
        """Load cached feature map tensor [C, h, w]."""
        p = self.cache_dir / f"{sample_id}_{layer}.pt"
        if not p.exists():
            return None
        return torch.load(p, map_location="cpu", weights_only=True)

    def save_sample_features(
        self, sample_id: str, features: dict[str, Tensor], dtype: str = "float32"
    ) -> None:
        """Save features for sample_id atomically using temporary files."""
        for layer_name, fmap in features.items():
            final_path = self.cache_dir / f"{sample_id}_{layer_name}.pt"
            if final_path.exists():
                continue
            tmp_path = self.cache_dir / f"{sample_id}_{layer_name}.pt.tmp"
            # Feature map shape: [1, C, h, w] or [C, h, w] -> save [C, h, w]
            tensor_to_save = fmap.squeeze(0).cpu() if fmap.ndim == 4 else fmap.cpu()
            if dtype == "float16" and tensor_to_save.is_floating_point():
                tensor_to_save = tensor_to_save.to(torch.float16)
            elif dtype != "float32":
                raise ValueError(f"Unsupported cache dtype: {dtype}")
            torch.save(tensor_to_save, tmp_path)
            tmp_path.replace(final_path)

    def write_metadata(self, metadata: dict[str, Any]) -> None:
        """Write metadata JSON atomically."""
        tmp_meta = self.cache_dir / "metadata.json.tmp"
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        tmp_meta.replace(self.metadata_path)
