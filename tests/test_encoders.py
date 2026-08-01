import torch

from fm_change_detection.encoders import get_encoder, registry
from fm_change_detection.encoders.simple import MockEncoder, RGBPixelsEncoder


def test_rgb_pixels_encoder():
    encoder = RGBPixelsEncoder()
    x = torch.rand(2, 3, 32, 32)
    feats = encoder.encode(x)
    assert "input" in feats
    assert feats["input"].shape == (2, 3, 32, 32)
    assert encoder.metadata.name == "rgb_pixels"


def test_mock_encoder_shape_and_freezing():
    encoder = MockEncoder(layers=("layer1", "layer2"))

    for p in encoder.parameters():
        assert not p.requires_grad

    x = torch.rand(2, 3, 32, 32)
    feats = encoder.encode(x)
    assert "layer1" in feats and "layer2" in feats
    assert feats["layer1"].shape[0] == 2
    assert feats["layer1"].ndim == 4
    assert feats["layer2"].ndim == 4


def test_resnet50_imagenet_encoder():
    encoder = get_encoder("resnet50_imagenet", layers=("layer2", "layer3"))
    x = torch.rand(1, 3, 64, 64)
    feats = encoder.encode(x)

    assert "layer2" in feats and "layer3" in feats
    for fmap in feats.values():
        assert fmap.ndim == 4
        assert fmap.shape[0] == 1


def test_dinov2_token_to_spatial_grid_conversion():
    encoder = get_encoder("dinov2_vits14", layers=("block3",))

    x = torch.rand(1, 3, 28, 28)
    feats = encoder.encode(x)

    assert "block3" in feats
    fmap = feats["block3"]
    assert fmap.ndim == 4
    assert fmap.shape[0] == 1

    assert fmap.shape[2:] == (2, 2)


def test_remote_encoder_uses_pinned_torchgeo_checkpoint(monkeypatch):
    captured = {}

    def fake_resnet(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(registry, "_resnet_encoder_class", lambda: fake_resnet)
    registry.get_encoder("resnet50_s2_moco", layers=("layer3",))

    assert "torchgeo/resnet50_sentinel2_rgb_moco" in captured["checkpoint_url"]
    assert captured["weights"] is None
    assert captured["license"] == "CC-BY-4.0"
