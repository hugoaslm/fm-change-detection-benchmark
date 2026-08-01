import torch
import torch.nn.functional as F
from torch import Tensor

from fm_change_detection.encoders.base import FeatureEncoder


def preprocess_images_for_encoder(images: Tensor, input_size: int = 252) -> Tensor:
    _b, _c, h, w = images.shape
    if h == input_size and w == input_size:
        return images
    return F.interpolate(
        images, size=(input_size, input_size), mode="bilinear", align_corners=False
    )


@torch.no_grad()
def extract_pair_features(
    encoder: FeatureEncoder,
    t1_images: Tensor,
    t2_images: Tensor,
    input_size: int = 252,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    t1_input = preprocess_images_for_encoder(t1_images, input_size=input_size)
    t2_input = preprocess_images_for_encoder(t2_images, input_size=input_size)

    f1_features = encoder.encode(t1_input)
    f2_features = encoder.encode(t2_input)

    return f1_features, f2_features
