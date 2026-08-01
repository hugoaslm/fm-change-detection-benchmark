import timm
import torch
from torch import Tensor, nn

from fm_change_detection.encoders.base import (
    EncoderMetadata,
    validate_feature_maps,
)


class DINOv2Encoder(nn.Module):
    def __init__(
        self,
        checkpoint: str = "vit_small_patch14_dinov2.lvd142m",
        layers: tuple[str, ...] = ("block3", "block6", "block9", "block12"),
        patch_size: int = 14,
        normalization_mean: tuple[float, ...] = (0.485, 0.456, 0.406),
        normalization_std: tuple[float, ...] = (0.229, 0.224, 0.225),
        license: str = "Apache-2.0",
    ) -> None:
        super().__init__()
        self.checkpoint_name = checkpoint
        self.requested_layers = layers
        self.patch_size = patch_size

        self.model = timm.create_model(
            checkpoint, pretrained=True, num_classes=0, dynamic_img_size=True
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.layer_to_block_idx = {}
        for l_name in layers:
            if l_name.startswith("block"):
                idx = int(l_name.replace("block", "")) - 1
                self.layer_to_block_idx[l_name] = idx
            else:
                raise ValueError(f"Unknown layer specification for DINOv2: {l_name}")

        param_count = sum(p.numel() for p in self.model.parameters())
        strides = {l: patch_size for l in layers}

        self.metadata = EncoderMetadata(
            name="dinov2_vits14",
            checkpoint=checkpoint,
            layers=layers,
            feature_strides=strides,
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
        b, _c, h, w = images.shape
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(
                f"Image dimensions ({h}x{w}) must be divisible by patch size {self.patch_size}"
            )

        grid_h = h // self.patch_size
        grid_w = w // self.patch_size

        mean = self.mean.to(device=images.device, dtype=images.dtype)
        std = self.std.to(device=images.device, dtype=images.dtype)
        norm_images = (images - mean) / std

        activations: dict[str, Tensor] = {}
        hooks = []

        def make_hook(layer_name: str):
            def hook(module, input, output):
                activations[layer_name] = output

            return hook

        for layer_name, block_idx in self.layer_to_block_idx.items():
            if block_idx < 0 or block_idx >= len(self.model.blocks):
                raise ValueError(
                    f"Block index {block_idx} for layer '{layer_name}' out of range for model with {len(self.model.blocks)} blocks"
                )
            h_ref = self.model.blocks[block_idx].register_forward_hook(make_hook(layer_name))
            hooks.append(h_ref)

        try:
            _ = self.model(norm_images)
        finally:
            for hook in hooks:
                hook.remove()

        outputs: dict[str, Tensor] = {}
        for layer_name in self.requested_layers:
            act = activations[layer_name]
            num_tokens = act.shape[1]
            expected_spatial_tokens = grid_h * grid_w

            if num_tokens == expected_spatial_tokens + 1:
                spatial_tokens = act[:, 1:, :]
            elif num_tokens == expected_spatial_tokens:
                spatial_tokens = act
            else:
                spatial_tokens = act[:, -expected_spatial_tokens:, :]

            feat_map = (
                spatial_tokens.reshape(b, grid_h, grid_w, -1).permute(0, 3, 1, 2).contiguous()
            )
            outputs[layer_name] = feat_map

        validate_feature_maps(outputs, b)
        return outputs
