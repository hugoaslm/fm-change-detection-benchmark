import torch

from fm_change_detection.cache import FeatureCache, compute_config_hash
from fm_change_detection.features import preprocess_images_for_encoder


def test_preprocess_images_for_encoder():
    imgs = torch.rand(2, 3, 100, 100)
    resized = preprocess_images_for_encoder(imgs, input_size=252)
    assert resized.shape == (2, 3, 252, 252)


def test_cache_hash_uniqueness():
    hash1 = compute_config_hash("levir_cd", "dinov2_vits14", "ckpt1", ("block3",), 252)
    hash2 = compute_config_hash("levir_cd", "dinov2_vits14", "ckpt2", ("block3",), 252)
    hash3 = compute_config_hash("levir_cd", "dinov2_vits14", "ckpt1", ("block3",), 256)

    assert hash1 != hash2
    assert hash1 != hash3


def test_feature_cache_io(tmp_dir):
    cache = FeatureCache(tmp_dir, "test_dataset", "hash123")
    fmap = torch.rand(16, 8, 8)

    cache.save_sample_features("sample_001", {"layer1": fmap})
    assert cache.is_cached("sample_001", "layer1")

    loaded = cache.get("sample_001", "layer1")
    assert loaded is not None
    assert torch.allclose(fmap, loaded)


def test_feature_cache_float16_storage(tmp_dir):
    cache = FeatureCache(tmp_dir, "test_dataset", "hash123")
    fmap = torch.rand(16, 8, 8)
    cache.save_sample_features("sample_002", {"layer1": fmap}, dtype="float16")

    loaded = cache.get("sample_002", "layer1")
    assert loaded is not None
    assert loaded.dtype == torch.float16
