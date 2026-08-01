from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from torch import Tensor


@dataclass(frozen=True)
class EncoderMetadata:
    name: str
    checkpoint: str
    layers: tuple[str, ...]
    feature_strides: dict[str, int]
    parameter_count: int
    normalization_mean: tuple[float, ...]
    normalization_std: tuple[float, ...]
    license: str


@runtime_checkable
class FeatureEncoder(Protocol):
    metadata: EncoderMetadata

    def encode(self, images: Tensor) -> dict[str, Tensor]: ...


def validate_feature_maps(features: dict[str, Tensor], expected_batch_size: int) -> None:
    for layer_name, fmap in features.items():
        if not isinstance(fmap, Tensor):
            raise TypeError(
                f"Feature map for layer '{layer_name}' must be torch.Tensor, got {type(fmap)}"
            )
        if fmap.ndim != 4:
            raise ValueError(
                f"Feature map for layer '{layer_name}' must have shape [B, C, H, W], got ndim={fmap.ndim} shape={fmap.shape}"
            )
        if fmap.shape[0] != expected_batch_size:
            raise ValueError(
                f"Expected batch size {expected_batch_size}, but layer '{layer_name}' got {fmap.shape[0]}"
            )
