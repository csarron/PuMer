from collections import OrderedDict

import numpy as np
import torch
from torch import nn


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, x_mask: torch.Tensor):
        if x_mask is not None:
            x_mask = x_mask.to(dtype=torch.bool, device=x.device)
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(
            x,
            x,
            x,
            need_weights=False,
            attn_mask=self.attn_mask,
            key_padding_mask=x_mask,
        )[0]

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        x = x + self.attention(self.ln_1(x), x_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers - 1)])

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        for block in self.resblocks:
            x = block(x, x_mask)
        return x


class VisualTransformer(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        resolution_after: int,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((resolution_after // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)

    def forward(self, x: torch.Tensor, x_mask=None):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        t = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([t, x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, x_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)

        return x


def adapt_position_encoding(model, patch_size=32, after=384, suffix="visual.positional_embedding"):
    keys = [k for k in model if k.endswith(suffix)]
    assert len(keys) == 1
    key = keys[0]
    origin_pos_embed = model[key]
    origin_dim2 = False
    if len(origin_pos_embed.shape) == 2:
        origin_dim2 = True
        origin_pos_embed = origin_pos_embed.unsqueeze(0)
    grid_before = int(np.sqrt(origin_pos_embed.shape[1] - 1))
    before = int(grid_before * patch_size)
    assert (before % patch_size) == 0
    grid_after = after // patch_size
    assert (after % patch_size) == 0
    embed_dim = origin_pos_embed.shape[-1]

    pos_embed = origin_pos_embed[0, 1:, :].reshape((grid_before, grid_before, embed_dim))
    new_size = (grid_after, grid_after)
    pos_embed = torch.nn.functional.interpolate(
        pos_embed.permute((2, 0, 1)).unsqueeze(0), size=new_size, mode="bicubic"
    )
    pos_embed = pos_embed.squeeze(0).permute((1, 2, 0)).reshape((-1, embed_dim))
    pos_embed = torch.cat((origin_pos_embed[0, 0:1, :], pos_embed), dim=0).unsqueeze(0)
    assert pos_embed.shape == (1, grid_after * grid_after + 1, embed_dim)
    if origin_dim2:
        assert pos_embed.shape[0] == 1
        pos_embed = pos_embed.squeeze(0)
    model[key] = pos_embed
    return model


def build_clip(patch_size, image_size):
    model = VisualTransformer(
        input_resolution=224,  # not used!
        patch_size=patch_size,  # 16 for ViT-B-16, 32 for ViT-B-32
        width=768,
        layers=12,
        heads=12,
        output_dim=512,
        resolution_after=image_size,
    )
    return model
