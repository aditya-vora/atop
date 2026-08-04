"""Multi-view *video* diffusion UNet.

The checkpoint in ``checkpoints/imagedream-ipmv-diffusers`` is a pretrained
multi-view *image* diffusion model (ImageDream/MVDream-style): it denoises
``num_views`` camera views of a single static frame, with no notion of time.

To finetune it into the spatially-controllable multi-view *video* model this
repository trains, the UNet needs two things the base checkpoint doesn't have:

  1. A temporal axis, separate from the view axis, with an ``attn_temp``
     attention layer that mixes information across a view's frames.
  2. Conditioning on the part-segmentation mask that specifies *which*
     articulated part the video should move — the "spatially controllable"
     part of ATOP. The mask FiLM-modulates the ``attn_temp`` features before
     attention, so different masks steer the same video toward articulating
     different parts.

``MultiViewUNetModel.from_pretrained_2d`` builds this extended architecture
and inflates it from the base checkpoint: layers that exist in both (spatial
attention, resnets, the camera/IP-Adapter heads, ...) are loaded from the
pretrained weights; the new temporal/mask layers start from their own
(randomly initialized, zero-gated) init and are learned during finetuning.
"""

from __future__ import annotations

import json
import math
import os
from inspect import isfunction
from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import xformers
import xformers.ops
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin, load_state_dict
from einops import rearrange, repeat


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


def timestep_embedding(timesteps, dim, max_period=10000):
    """Sinusoidal timestep embeddings, shape (N, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class InflatedConv3d(nn.Conv2d):
    """A 2D conv applied independently to every frame of a (b, c, f, h, w) tensor."""

    def forward(self, x):
        video_length = x.shape[2]
        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = super().forward(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", f=video_length)
        return x


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )
        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.net(x)


class MemoryEfficientCrossAttention(nn.Module):
    """Spatial self-/cross-attention, with optional IP-Adapter image conditioning."""

    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        ip_dim=0,
        ip_weight=1.0,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head
        self.ip_dim = ip_dim
        self.ip_weight = ip_weight

        if self.ip_dim > 0:
            self.to_k_ip = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_v_ip = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def _split_heads(self, t, b):
        return (
            t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous()
        )

    def forward(self, x, context=None):
        q = self.to_q(x)
        context = default(context, x)

        if self.ip_dim > 0:
            token_len = context.shape[1]
            context_ip = context[:, -self.ip_dim :, :]
            k_ip = self.to_k_ip(context_ip)
            v_ip = self.to_v_ip(context_ip)
            context = context[:, : (token_len - self.ip_dim), :]

        k = self.to_k(context)
        v = self.to_v(context)

        b = q.shape[0]
        q, k, v = (self._split_heads(t, b) for t in (q, k, v))
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        if self.ip_dim > 0:
            k_ip, v_ip = (self._split_heads(t, b) for t in (k_ip, v_ip))
            out_ip = xformers.ops.memory_efficient_attention(
                q, k_ip, v_ip, attn_bias=None, op=self.attention_op
            )
            out = out + self.ip_weight * out_ip

        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


class MaskedTemporalAttention(nn.Module):
    """Attention across a view's frames, FiLM-modulated by the part mask.

    This is the model's spatial-control mechanism: ``masks`` (one binary part
    mask per view) is convolved into a per-frame (scale, shift) pair that
    modulates the token features before temporal self-attention, so the same
    frames attend differently depending on which part the mask highlights.
    """

    def __init__(self, query_dim, latent_size, num_frames, num_views, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.dim_head = dim_head
        self.latent_size = latent_size
        self.num_frames = num_frames
        self.num_views = num_views

        self.mask_conv = nn.Sequential(
            nn.Conv2d(1, inner_dim, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(inner_dim),
            nn.ReLU(),
            nn.Conv2d(inner_dim, 2 * inner_dim, kernel_size=3, padding=1, bias=True),
        )

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def forward(self, x, masks):
        v, f, l = self.num_views, self.num_frames, self.latent_size * self.latent_size

        masks = rearrange(masks, "b v h w -> (b v) h w", v=v).contiguous()
        masks = F.interpolate(masks[:, None], size=(self.latent_size, self.latent_size), mode="nearest")
        masks = masks.to(dtype=x.dtype)
        masks = (masks - masks.mean(dim=(2, 3), keepdim=True)) / (masks.std(dim=(2, 3), keepdim=True) + 1e-5)
        masks = masks[:, None].repeat(1, f, 1, 1, 1)
        masks = rearrange(masks, "(b v) f c h w -> (b v f) c h w", v=v, f=f).contiguous()
        scale, shift = self.mask_conv(masks).chunk(2, dim=1)
        scale = rearrange(scale, "(b v f) c h w -> (b v) (f h w) c", v=v, f=f)
        shift = rearrange(shift, "(b v f) c h w -> (b v) (f h w) c", v=v, f=f)

        x = x * (1.0 + scale) + shift
        x = rearrange(x, "(b v) (f l) c -> (b v l) f c", v=v, f=f)

        q = self.to_q(x)
        k = self.to_k(x)
        v_ = self.to_v(x)

        b = q.shape[0]
        q, k, v_ = (
            t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous()
            for t in (q, k, v_)
        )
        out = xformers.ops.memory_efficient_attention(q, k, v_, attn_bias=None, op=self.attention_op)
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        out = self.to_out(out)
        out = rearrange(out, "(b v l) f c -> (b v) (f l) c", v=self.num_views, l=l)
        return out


class PerceiverAttention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        x = self.norm1(x)
        latents = self.norm2(latents)
        b, l, _ = latents.shape

        q = self.to_q(latents)
        k, v = self.to_kv(torch.cat((x, latents), dim=-2)).chunk(2, dim=-1)
        q, k, v = (
            t.reshape(b, t.shape[1], self.heads, -1).transpose(1, 2).contiguous() for t in (q, k, v)
        )

        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = torch.softmax(((q * scale) @ (k * scale).transpose(-2, -1)).float(), dim=-1).type(q.dtype)
        out = (weight @ v).permute(0, 2, 1, 3).reshape(b, l, -1)
        return self.to_out(out)


class Resampler(nn.Module):
    """IP-Adapter-style resampler: CLIP image features -> ``num_queries`` context tokens."""

    def __init__(self, dim, depth, dim_head, heads, num_queries, embedding_dim, output_dim, ff_mult=4):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)
        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        nn.Sequential(
                            nn.LayerNorm(dim),
                            nn.Linear(dim, dim * ff_mult, bias=False),
                            nn.GELU(),
                            nn.Linear(dim * ff_mult, dim, bias=False),
                        ),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        latents = self.latents.repeat(x.size(0), 1, 1)
        x = self.proj_in(x)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        return self.norm_out(self.proj_out(latents))


class BasicTransformerBlock3D(nn.Module):
    """Spatial self-attn (across views) -> masked temporal attn (across frames) -> cross-attn -> FF."""

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        context_dim,
        latent_size,
        num_frames,
        num_views,
        dropout=0.0,
        gated_ff=True,
        ip_dim=0,
        ip_weight=1.0,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.num_views = num_views

        self.attn1 = MemoryEfficientCrossAttention(
            query_dim=dim, context_dim=None, heads=n_heads, dim_head=d_head, dropout=dropout
        )
        self.attn_temp = MaskedTemporalAttention(
            query_dim=dim,
            latent_size=latent_size,
            num_frames=num_frames,
            num_views=num_views,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )
        zero_module(self.attn_temp.to_out[0])
        self.attn2 = MemoryEfficientCrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            ip_dim=ip_dim,
            ip_weight=ip_weight,
        )
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)

        self.norm1 = nn.LayerNorm(dim)
        self.norm_temp = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, context, num_frames, num_views, masks):
        x = rearrange(x, "(b v f) l c -> (b f) (v l) c", v=num_views, f=num_frames).contiguous()
        x = self.attn1(self.norm1(x)) + x

        x = rearrange(x, "(b f) (v l) c -> (b v) (f l) c", v=num_views, f=num_frames).contiguous()
        x = self.attn_temp(self.norm_temp(x), masks=masks) + x

        x = rearrange(x, "(b v) (f l) c -> (b v f) l c", v=num_views, f=num_frames).contiguous()
        context = rearrange(context, "b v f l c -> (b v f) l c", v=num_views, f=num_frames)
        x = self.attn2(self.norm2(x), context=context) + x

        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer3D(nn.Module):
    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        context_dim,
        latent_size,
        num_frames,
        num_views,
        depth=1,
        dropout=0.0,
        ip_dim=0,
        ip_weight=1.0,
    ):
        super().__init__()
        if not isinstance(context_dim, list):
            context_dim = [context_dim] * depth

        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock3D(
                    inner_dim,
                    n_heads,
                    d_head,
                    context_dim=context_dim[d],
                    latent_size=latent_size,
                    num_frames=num_frames,
                    num_views=num_views,
                    dropout=dropout,
                    ip_dim=ip_dim,
                    ip_weight=ip_weight,
                )
                for d in range(depth)
            ]
        )
        self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))

    def forward(self, x, context, num_frames, num_views, masks):
        if not isinstance(context, list):
            context = [context] * len(self.transformer_blocks)

        b, c, f, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = rearrange(x, "b c f h w -> (b f) (h w) c").contiguous()
        x = self.proj_in(x)

        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i], num_frames=num_frames, num_views=num_views, masks=masks)

        x = self.proj_out(x)
        x = rearrange(x, "(b f) (h w) c -> b c f h w", f=f, h=h, w=w).contiguous()
        return x + x_in


class CondSequential(nn.Sequential):
    """Sequential that forwards timestep/context/mask conditioning to layers that need it."""

    def forward(self, x, emb, context=None, num_frames=1, num_views=1, masks=None):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer3D):
                x = layer(x, context, num_frames=num_frames, num_views=num_views, masks=masks)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = InflatedConv3d(self.channels, self.out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = rearrange(x, "(b f) c h w -> b c f h w", f=f)
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels, use_conv, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.op = InflatedConv3d(
                self.channels, self.out_channels, kernel_size=3, stride=2, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def forward(self, x):
        return self.op(x)


class ResBlock(nn.Module):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
    ):
        super().__init__()
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            InflatedConv3d(channels, self.out_channels, kernel_size=3, padding=1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(InflatedConv3d(self.out_channels, self.out_channels, kernel_size=3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = InflatedConv3d(channels, self.out_channels, kernel_size=3, padding=1)
        else:
            self.skip_connection = InflatedConv3d(channels, self.out_channels, kernel_size=1)

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class MultiViewUNetModel(ModelMixin, ConfigMixin):
    """Multi-view video UNet: spatial (per-frame, cross-view) + temporal (per-view, mask-conditioned) attention."""

    @register_to_config
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        num_frames,
        num_views,
        dropout=0.0,
        channel_mult=(1, 2, 4, 4),
        num_heads=-1,
        num_head_channels=-1,
        transformer_depth=1,
        context_dim=None,
        camera_dim=None,
        ip_dim=0,
        ip_weight=1.0,
        **kwargs,
    ):
        super().__init__()
        assert context_dim is not None
        assert num_heads != -1 or num_head_channels != -1, "set num_heads or num_head_channels"

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_frames = num_frames
        self.num_views = num_views
        self.num_res_blocks = (
            len(channel_mult) * [num_res_blocks] if isinstance(num_res_blocks, int) else num_res_blocks
        )
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        self.ip_dim = ip_dim
        self.ip_weight = ip_weight

        latent_sizes = [image_size // (2**level) for level in range(len(channel_mult))]

        if self.ip_dim > 0:
            self.image_embed = Resampler(
                dim=context_dim,
                depth=4,
                dim_head=64,
                heads=12,
                num_queries=ip_dim,
                embedding_dim=1280,
                output_dim=context_dim,
                ff_mult=4,
            )

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        if camera_dim is not None:
            self.camera_embed = nn.Sequential(
                nn.Linear(camera_dim, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )

        def make_attn(ch, latent_size):
            if num_head_channels == -1:
                heads, dim_head = num_heads, ch // num_heads
            else:
                heads, dim_head = ch // num_head_channels, num_head_channels
            return SpatialTransformer3D(
                ch,
                heads,
                dim_head,
                context_dim=context_dim,
                depth=transformer_depth,
                ip_dim=self.ip_dim,
                ip_weight=self.ip_weight,
                latent_size=latent_size,
                num_frames=num_frames,
                num_views=num_views,
            )

        self.input_blocks = nn.ModuleList(
            [CondSequential(InflatedConv3d(in_channels, model_channels, kernel_size=3, padding=1))]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(self.num_res_blocks[level]):
                layers: List[Any] = [
                    ResBlock(ch, time_embed_dim, dropout, out_channels=mult * model_channels)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(make_attn(ch, latent_sizes[level]))
                self.input_blocks.append(CondSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(CondSequential(Downsample(ch, True, out_channels=ch)))
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = CondSequential(
            ResBlock(ch, time_embed_dim, dropout),
            make_attn(ch, latent_sizes[-1]),
            ResBlock(ch, time_embed_dim, dropout),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [ResBlock(ch + ich, time_embed_dim, dropout, out_channels=model_channels * mult)]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(make_attn(ch, latent_sizes[level]))
                if level and i == self.num_res_blocks[level]:
                    layers.append(Upsample(ch, True, out_channels=ch))
                    ds //= 2
                self.output_blocks.append(CondSequential(*layers))

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            zero_module(InflatedConv3d(model_channels, out_channels, kernel_size=3, padding=1)),
        )

    @classmethod
    def from_pretrained_2d(cls, pretrained_model_path, subfolder, num_frames, num_views):
        """Build the video model and inflate it from a pretrained multi-view *image* checkpoint.

        Layers present in both architectures (everything except the new
        temporal/mask attention) are loaded from the checkpoint; the rest
        keep their freshly initialized (zero-gated) weights.
        """
        path = os.path.join(pretrained_model_path, subfolder) if subfolder else pretrained_model_path
        config_file = os.path.join(path, "config.json")
        if not os.path.isfile(config_file):
            raise FileNotFoundError(config_file)
        with open(config_file, "r") as f:
            config = json.load(f)

        config["_class_name"] = cls.__name__
        config["num_frames"] = num_frames
        config["num_views"] = num_views
        model = cls.from_config(config)

        model_file = os.path.join(path, "diffusion_pytorch_model.safetensors")
        if not os.path.isfile(model_file):
            raise FileNotFoundError(model_file)
        state_dict = load_state_dict(model_file)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        new_params = [k for k in missing if "attn_temp" in k or "norm_temp" in k]
        other_missing = [k for k in missing if "attn_temp" not in k and "norm_temp" not in k]
        print(
            f"[MultiViewUNetModel.from_pretrained_2d] inflated from '{path}': "
            f"{len(state_dict) - len(unexpected)} pretrained tensors loaded, "
            f"{len(new_params)} new temporal/mask parameters randomly initialized."
        )
        if other_missing:
            print(f"  warning: {len(other_missing)} non-temporal params missing from checkpoint: {other_missing[:5]}...")
        if unexpected:
            print(f"  warning: {len(unexpected)} checkpoint tensors unused: {unexpected[:5]}...")
        return model

    def forward(self, x, timesteps, context, camera=None, ip=None, mv_masks=None):
        """
        Args:
            x: (b, v, c, f, h, w) noisy video latents.
            timesteps: (b,) diffusion timesteps, one per sample and shared across its views.
            context: (b, v, f, seq_len, context_dim) text embeddings, broadcast per view/frame.
            camera: (b, v, camera_dim) camera-to-world poses, one per view.
            ip: (b, v, f, num_tokens, context_dim) CLIP image tokens for IP-Adapter conditioning.
            mv_masks: (b, v, h, w) binary part masks, one per view.
        Returns:
            (b, v, out_channels, f, h, w) denoised prediction.
        """
        bsz, num_views = x.shape[0], self.num_views
        num_frames = self.num_frames
        x = rearrange(x, "b v c f h w -> (b v) c f h w", v=num_views).contiguous()

        timesteps = timesteps.unsqueeze(1).expand(bsz, num_views).reshape(-1)
        t_emb = timestep_embedding(timesteps, self.model_channels)
        emb = self.time_embed(t_emb.to(x.dtype))
        if camera is not None:
            camera = rearrange(camera, "b v d -> (b v) d")
            emb = emb + self.camera_embed(camera.to(x.dtype))

        if self.ip_dim > 0 and ip is not None:
            ip = rearrange(ip, "b v f n c -> (b v f) n c")
            ip_emb = self.image_embed(ip)
            ip_emb = rearrange(ip_emb, "(b v f) n c -> b v f n c", b=bsz, v=num_views, f=num_frames)
            context = torch.cat((context, ip_emb), dim=3)

        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb, context, num_frames=num_frames, num_views=num_views, masks=mv_masks)
            hs.append(h)

        h = self.middle_block(h, emb, context, num_frames=num_frames, num_views=num_views, masks=mv_masks)

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context, num_frames=num_frames, num_views=num_views, masks=mv_masks)

        h = self.out(h.type(x.dtype))
        return rearrange(h, "(b v) c f h w -> b v c f h w", v=num_views)
