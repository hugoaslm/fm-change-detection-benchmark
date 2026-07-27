"""Robustness perturbations applied to T2 image timestamp."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def apply_brightness(image: Tensor, factor: float) -> Tensor:
    """Adjust brightness of image [3, H, W] in [0, 1].

    Factor can be additive (e.g. +0.15) or multiplicative (e.g. 1.15).
    """
    if abs(factor) < 1.0:
        # Additive shift
        return torch.clamp(image + factor, 0.0, 1.0)
    else:
        # Multiplicative scaling
        return torch.clamp(image * factor, 0.0, 1.0)


def apply_contrast(image: Tensor, factor: float) -> Tensor:
    """Adjust contrast of image tensor [3, H, W] around channel means."""
    mean = image.mean(dim=(-2, -1), keepdim=True)
    return torch.clamp((image - mean) * factor + mean, 0.0, 1.0)


def apply_gaussian_noise(image: Tensor, sigma: float = 0.03, seed: int = 42) -> Tensor:
    """Add zero-mean Gaussian noise with standard deviation sigma."""
    generator = torch.Generator(device=image.device).manual_seed(seed)
    noise = (
        torch.randn(image.shape, generator=generator, device=image.device, dtype=image.dtype)
        * sigma
    )
    return torch.clamp(image + noise, 0.0, 1.0)


def apply_gaussian_blur(image: Tensor, sigma: float = 1.0) -> Tensor:
    """Apply Gaussian blur with standard deviation sigma."""
    kernel_size = int(2 * math.ceil(2 * sigma) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Create 1D Gaussian kernel
    x = torch.arange(kernel_size, dtype=torch.float32, device=image.device) - (kernel_size // 2)
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    # 2D kernel
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_4d = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)

    pad = kernel_size // 2
    if image.ndim == 3:
        img_4d = image.unsqueeze(0)
        padded = F.pad(img_4d, (pad, pad, pad, pad), mode="reflect")
        blurred = F.conv2d(padded, kernel_4d, groups=3)
        return torch.clamp(blurred.squeeze(0), 0.0, 1.0)
    else:
        padded = F.pad(image, (pad, pad, pad, pad), mode="reflect")
        blurred = F.conv2d(padded, kernel_4d, groups=3)
        return torch.clamp(blurred, 0.0, 1.0)


def apply_translation(image: Tensor, pixels: int = 2) -> Tensor:
    """Translate image tensor by [pixels, pixels] using zero-padding or affine grid."""
    if pixels == 0:
        return image

    if image.ndim == 3:
        img_4d = image.unsqueeze(0)
        b = 1
    else:
        img_4d = image
        b = image.shape[0]

    # Create affine transformation matrix for translation
    # normalized tx, ty = 2 * pixels / W, 2 * pixels / H
    _, _, h, w = img_4d.shape
    tx = 2.0 * pixels / w
    ty = 2.0 * pixels / h

    theta = torch.tensor(
        [[[1.0, 0.0, tx], [0.0, 1.0, ty]] for _ in range(b)],
        device=img_4d.device,
        dtype=img_4d.dtype,
    )
    grid = F.affine_grid(theta, img_4d.size(), align_corners=False)
    translated = F.grid_sample(
        img_4d, grid, mode="bilinear", padding_mode="border", align_corners=False
    )

    return translated.squeeze(0) if image.ndim == 3 else translated


def apply_saturation(image: Tensor, factor: float) -> Tensor:
    """Adjust saturation of RGB image tensor [3, H, W] in [0, 1].

    factor: relative scaling (e.g. +0.20 -> 1.20, -0.20 -> 0.80).
    """
    scale = 1.0 + factor if abs(factor) <= 1.0 else factor
    # Luminance weights for RGB
    weights = torch.tensor([0.299, 0.587, 0.114], device=image.device, dtype=image.dtype).view(
        3, 1, 1
    )
    grayscale = (image * weights).sum(dim=0, keepdim=True)
    saturated = grayscale + scale * (image - grayscale)
    return torch.clamp(saturated, 0.0, 1.0)


def apply_perturbation(
    t2_image: Tensor, perturbation_name: str, value: float, seed: int = 42
) -> Tensor:
    """Dispatch perturbation function to T2 image tensor [3, H, W] or [B, 3, H, W]."""
    name = perturbation_name.lower()
    if name == "brightness":
        return apply_brightness(t2_image, float(value))
    elif name == "contrast":
        return apply_contrast(t2_image, float(value))
    elif name == "gaussian_noise":
        return apply_gaussian_noise(t2_image, sigma=float(value), seed=seed)
    elif name == "gaussian_blur":
        return apply_gaussian_blur(t2_image, sigma=float(value))
    elif name == "translation":
        return apply_translation(t2_image, pixels=int(value))
    elif name in ("saturation", "color"):
        return apply_saturation(t2_image, factor=float(value))
    else:
        raise ValueError(f"Unknown perturbation type: {perturbation_name}")
