import torch
import torch.nn.functional as F
from torch import Tensor, nn

from fm_change_detection.encoders.base import EncoderMetadata, validate_feature_maps


class RGBPixelsEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.metadata = EncoderMetadata(
            name="rgb_pixels",
            checkpoint="none",
            layers=("input",),
            feature_strides={"input": 1},
            parameter_count=0,
            normalization_mean=(0.0, 0.0, 0.0),
            normalization_std=(1.0, 1.0, 1.0),
            license="MIT",
        )
        self.eval()

    @torch.no_grad()
    def encode(self, images: Tensor) -> dict[str, Tensor]:
        self.eval()
        outputs = {"input": images.detach()}
        validate_feature_maps(outputs, images.shape[0])
        return outputs


class MockEncoder(nn.Module):
    def __init__(self, layers: tuple[str, ...] = ("layer1", "layer2")) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

        self.metadata = EncoderMetadata(
            name="mock_encoder",
            checkpoint="none",
            layers=layers,
            feature_strides={"layer1": 1, "layer2": 2},
            parameter_count=sum(parameter.numel() for parameter in self.parameters()),
            normalization_mean=(0.5, 0.5, 0.5),
            normalization_std=(0.5, 0.5, 0.5),
            license="MIT",
        )

    @torch.no_grad()
    def encode(self, images: Tensor) -> dict[str, Tensor]:
        self.eval()
        h1 = F.relu(self.conv1(images))
        h2 = F.max_pool2d(F.relu(self.conv2(h1)), 2)
        outputs = {}
        if "layer1" in self.metadata.layers:
            outputs["layer1"] = h1
        if "layer2" in self.metadata.layers:
            outputs["layer2"] = h2
        validate_feature_maps(outputs, images.shape[0])
        return outputs
