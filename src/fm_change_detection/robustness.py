import math

import torch
import torch.nn.functional as F
from torch import Tensor


def apply_brightness(image: Tensor, factor: float) -> Tensor:
    if abs(factor) < 1.0:
        return torch.clamp(image + factor, 0.0, 1.0)
    else:
        return torch.clamp(image * factor, 0.0, 1.0)


def apply_contrast(image: Tensor, factor: float) -> Tensor:
    mean = image.mean(dim=(-2, -1), keepdim=True)
    return torch.clamp((image - mean) * factor + mean, 0.0, 1.0)


def apply_gaussian_noise(image: Tensor, sigma: float = 0.03, seed: int = 42) -> Tensor:
    generator = torch.Generator(device=image.device).manual_seed(seed)
    noise = (
        torch.randn(image.shape, generator=generator, device=image.device, dtype=image.dtype)
        * sigma
    )
    return torch.clamp(image + noise, 0.0, 1.0)


def apply_gaussian_blur(image: Tensor, sigma: float = 1.0) -> Tensor:
    kernel_size = int(2 * math.ceil(2 * sigma) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    x = torch.arange(kernel_size, dtype=torch.float32, device=image.device) - (kernel_size // 2)
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

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
    if pixels == 0:
        return image

    if image.ndim == 3:
        img_4d = image.unsqueeze(0)
        b = 1
    else:
        img_4d = image
        b = image.shape[0]

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
    scale = 1.0 + factor if abs(factor) <= 1.0 else factor

    weights = torch.tensor([0.299, 0.587, 0.114], device=image.device, dtype=image.dtype).view(
        3, 1, 1
    )
    grayscale = (image * weights).sum(dim=0, keepdim=True)
    saturated = grayscale + scale * (image - grayscale)
    return torch.clamp(saturated, 0.0, 1.0)


def apply_perturbation(
    t2_image: Tensor, perturbation_name: str, value: float, seed: int = 42
) -> Tensor:
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
