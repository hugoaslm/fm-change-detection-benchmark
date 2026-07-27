"""Feature extraction pipelines and tensor resizing utilities."""

import torch
import torch.nn.functional as F
from torch import Tensor

from fm_change_detection.encoders.base import FeatureEncoder


def preprocess_images_for_encoder(images: Tensor, input_size: int = 252) -> Tensor:
    """Resize images [B, 3, H, W] to model input size [B, 3, input_size, input_size]."""
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
    """Extract features for a batch of T1 and T2 image pairs.

    Returns:
        tuple (f1_dict, f2_dict) mapping layer_name -> Tensor [B, C, h, w].
    """
    t1_input = preprocess_images_for_encoder(t1_images, input_size=input_size)
    t2_input = preprocess_images_for_encoder(t2_images, input_size=input_size)

    f1_features = encoder.encode(t1_input)
    f2_features = encoder.encode(t2_input)

    return f1_features, f2_features
