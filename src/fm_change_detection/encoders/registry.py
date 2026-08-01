from typing import Any

from fm_change_detection.encoders.base import FeatureEncoder

S2_RGB_MOCO_URL = (
    "https://hf.co/torchgeo/resnet50_sentinel2_rgb_moco/resolve/"
    "efd9723b59a88e9dc1420dc1e96afb25b0630a3c/"
    "resnet50_sentinel2_rgb_moco-2b57ba8b.pth"
)


def _resnet_encoder_class():
    from fm_change_detection.encoders.resnet import ResNet50Encoder

    return ResNet50Encoder


def get_encoder(
    name: str,
    checkpoint: str = "none",
    layers: tuple[str, ...] | None = None,
    **kwargs: Any,
) -> FeatureEncoder:
    name_lower = name.lower()

    if name_lower == "rgb_pixels":
        from fm_change_detection.encoders.simple import RGBPixelsEncoder

        return RGBPixelsEncoder()

    elif name_lower == "resnet50_imagenet":
        ResNet50Encoder = _resnet_encoder_class()
        requested_layers = layers or ("layer2", "layer3", "layer4")
        return ResNet50Encoder(
            name="resnet50_imagenet",
            weights="DEFAULT",
            layers=tuple(requested_layers),
            **kwargs,
        )

    elif name_lower in ("resnet50_s2_moco", "resnet50_remote_sensing"):
        ResNet50Encoder = _resnet_encoder_class()
        requested_layers = layers or ("layer2", "layer3", "layer4")
        return ResNet50Encoder(
            name="resnet50_s2_moco",
            weights=None,
            checkpoint_url=S2_RGB_MOCO_URL,
            layers=tuple(requested_layers),
            normalization_mean=(0.0, 0.0, 0.0),
            normalization_std=(1.0, 1.0, 1.0),
            license="CC-BY-4.0",
            **kwargs,
        )

    elif name_lower in ("dinov2_vits14", "dinov2"):
        from fm_change_detection.encoders.dinov2 import DINOv2Encoder

        requested_layers = layers or ("block3", "block6", "block9", "block12")
        ckpt = (
            checkpoint
            if checkpoint and checkpoint != "none"
            else "vit_small_patch14_dinov2.lvd142m"
        )
        return DINOv2Encoder(
            checkpoint=ckpt,
            layers=tuple(requested_layers),
            **kwargs,
        )

    elif name_lower == "mock_encoder":
        from fm_change_detection.encoders.simple import MockEncoder

        requested_layers = layers or ("layer1", "layer2")
        return MockEncoder(layers=tuple(requested_layers))

    else:
        raise ValueError(
            f"Unknown encoder name '{name}'. Available: 'rgb_pixels', 'resnet50_imagenet', 'resnet50_s2_moco', 'dinov2_vits14', 'mock_encoder'"
        )
