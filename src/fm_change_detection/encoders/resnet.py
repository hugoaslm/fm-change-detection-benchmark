import torch
from torch import Tensor, nn
from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.feature_extraction import create_feature_extractor

from fm_change_detection.encoders.base import (
    EncoderMetadata,
    validate_feature_maps,
)


class ResNet50Encoder(nn.Module):
    def __init__(
        self,
        name: str = "resnet50_imagenet",
        weights: str | None = "DEFAULT",
        layers: tuple[str, ...] = ("layer2", "layer3", "layer4"),
        custom_state_dict: dict[str, Tensor] | None = None,
        checkpoint_url: str | None = None,
        normalization_mean: tuple[float, ...] = (0.485, 0.456, 0.406),
        normalization_std: tuple[float, ...] = (0.229, 0.224, 0.225),
        license: str = "BSD-3-Clause",
    ) -> None:
        super().__init__()

        if checkpoint_url is not None:
            model = resnet50(weights=None)
            state_dict = load_state_dict_from_url(
                checkpoint_url, progress=True, check_hash=True, weights_only=True
            )
            incompatible = model.load_state_dict(state_dict, strict=False)
            allowed_missing = {"fc.weight", "fc.bias"}
            unexpected = set(incompatible.unexpected_keys)
            missing = set(incompatible.missing_keys) - allowed_missing
            if unexpected or missing:
                raise RuntimeError(
                    "Remote-sensing checkpoint is incompatible with ResNet-50: "
                    f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
                )
        elif custom_state_dict is None:
            if weights == "DEFAULT" or weights == "IMAGENET1K_V1":
                model_weights = ResNet50_Weights.DEFAULT
            else:
                model_weights = None
            model = resnet50(weights=model_weights)
        else:
            model = resnet50(weights=None)
            model.load_state_dict(custom_state_dict, strict=False)

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        return_nodes = {layer: layer for layer in layers}
        self.feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)
        self.feature_extractor.eval()

        strides = {"layer1": 4, "layer2": 8, "layer3": 16, "layer4": 32}
        feat_strides = {l: strides.get(l, 1) for l in layers}

        param_count = sum(p.numel() for p in model.parameters())

        self.metadata = EncoderMetadata(
            name=name,
            checkpoint=checkpoint_url or (str(weights) if weights else "custom"),
            layers=layers,
            feature_strides=feat_strides,
            parameter_count=param_count,
            normalization_mean=normalization_mean,
            normalization_std=normalization_std,
            license=license,
        )

        self.mean = torch.tensor(normalization_mean).view(1, 3, 1, 1)
        self.std = torch.tensor(normalization_std).view(1, 3, 1, 1)

    @torch.no_grad()
    def encode(self, images: Tensor) -> dict[str, Tensor]:
        self.eval()
        mean = self.mean.to(device=images.device, dtype=images.dtype)
        std = self.std.to(device=images.device, dtype=images.dtype)
        norm_images = (images - mean) / std

        features = self.feature_extractor(norm_images)
        validate_feature_maps(features, images.shape[0])
        return features
