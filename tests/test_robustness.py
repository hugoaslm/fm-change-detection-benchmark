import torch

from fm_change_detection.robustness import (
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_translation,
)


def test_blur_does_not_create_dark_border_on_constant_image():
    image = torch.ones(3, 32, 32)
    assert torch.allclose(apply_gaussian_blur(image, sigma=1.0), image, atol=1e-6)


def test_translation_uses_border_padding():
    image = torch.ones(3, 32, 32)
    assert torch.allclose(apply_translation(image, pixels=4), image, atol=1e-6)


def test_noise_is_reproducible_but_seed_dependent():
    image = torch.zeros(3, 16, 16)
    first = apply_gaussian_noise(image, seed=1)
    repeated = apply_gaussian_noise(image, seed=1)
    different = apply_gaussian_noise(image, seed=2)
    assert torch.equal(first, repeated)
    assert not torch.equal(first, different)
