import torch
import torch.nn.functional as F
from torch import Tensor


def cosine_score(t1: Tensor, t2: Tensor, eps: float = 1e-8) -> Tensor:
    norm1 = torch.linalg.vector_norm(t1, dim=1)
    norm2 = torch.linalg.vector_norm(t2, dim=1)
    normalized1 = F.normalize(t1, dim=1, eps=eps)
    normalized2 = F.normalize(t2, dim=1, eps=eps)
    score = 1.0 - (normalized1 * normalized2).sum(dim=1)

    both_zero = (norm1 <= eps) & (norm2 <= eps)
    return torch.where(both_zero, torch.zeros_like(score), score.clamp(0.0, 2.0))


def standardized_euclidean_score(
    t1: Tensor, t2: Tensor, channel_std: Tensor, eps: float = 1e-6
) -> Tensor:
    if channel_std.ndim == 1:
        std = channel_std.view(1, -1, 1, 1).to(device=t1.device, dtype=t1.dtype)
    else:
        std = channel_std.to(device=t1.device, dtype=t1.dtype)

    diff = (t1 - t2) / (std + eps)
    return torch.sqrt(torch.sum(diff**2, dim=1))


def upsample_score_map(score_map: Tensor, target_size: tuple[int, int] = (256, 256)) -> Tensor:
    if score_map.ndim == 3:
        score_4d = score_map.unsqueeze(1)
        upsampled = F.interpolate(score_4d, size=target_size, mode="bilinear", align_corners=False)
        return upsampled.squeeze(1)
    elif score_map.ndim == 4:
        return F.interpolate(score_map, size=target_size, mode="bilinear", align_corners=False)
    else:
        raise ValueError(f"Expected 3D or 4D score map, got ndim={score_map.ndim}")


class FeatureStatsTracker:
    def __init__(self) -> None:
        self.count = 0
        self.mean: Tensor | None = None
        self.M2: Tensor | None = None

    def update(self, features: Tensor) -> None:
        _b, c, _h, _w = features.shape
        flat = features.permute(0, 2, 3, 1).reshape(-1, c).detach().cpu()
        n = flat.shape[0]

        if self.count == 0:
            self.count = n
            self.mean = flat.mean(dim=0)
            self.M2 = ((flat - self.mean) ** 2).sum(dim=0)
        else:
            self.count += n
            delta = flat - self.mean
            self.mean += delta.sum(dim=0) / self.count
            delta2 = flat - self.mean
            self.M2 += (delta * delta2).sum(dim=0)

    def get_std(self, eps: float = 1e-6) -> Tensor:
        if self.count < 2 or self.M2 is None:
            raise RuntimeError("Not enough samples to compute channel std")
        var = self.M2 / (self.count - 1)
        return torch.sqrt(torch.clamp(var, min=eps))
