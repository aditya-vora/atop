"""Sampling pipeline for the ATOP multi-view video diffusion model.

A ``diffusers.DiffusionPipeline`` wired for ``MultiViewUNetModel``: classifier-free
guidance over a text prompt and per-view IP-Adapter image conditioning, with the
camera pose and part mask threaded straight through to the UNet (see
``src/models/mv_unet.py`` for what those condition on).

This mirrors what ``finetune.py`` trains, generating one shape's multi-view
articulation video at a time (batch size 1 -- see ``finetune.py``'s note on
why the architecture is single-example-per-step by convention).
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import torch
from diffusers import AutoencoderKL, DiffusionPipeline
from diffusers.schedulers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from PIL import Image
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModel

from src.models.mv_unet import MultiViewUNetModel
from src.utils.cam_utils import get_camera


class MultiViewVideoPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        unet: MultiViewUNetModel,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        scheduler: DDIMScheduler,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModel,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            unet=unet,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def _encode_prompt(self, prompt: str, negative_prompt: str, do_cfg: bool, device) -> torch.Tensor:
        prompts = [negative_prompt, prompt] if do_cfg else [prompt]
        input_ids = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        return self.text_encoder(input_ids)[0]  # (2, seq, dim) if do_cfg else (1, seq, dim)

    def _encode_image(self, mv_clip_images: torch.Tensor, device) -> torch.Tensor:
        """mv_clip_images: (v, 3, H, W), CLIP-preprocessed -> (v, num_tokens, hidden_dim)."""
        dtype = next(self.image_encoder.parameters()).dtype
        mv_clip_images = mv_clip_images.to(device=device, dtype=dtype)
        return self.image_encoder(mv_clip_images, output_hidden_states=True).hidden_states[-2]

    def prepare_latents(self, num_views, num_frames, height, width, dtype, device, generator):
        shape = (
            1,
            num_views,
            self.unet.config.out_channels,
            num_frames,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents * self.scheduler.init_noise_sigma

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """(b, v, c, f, h, w) latents -> (b, v, f, H, W, 3) float images in [0, 1]."""
        b, v, c, f, h, w = latents.shape
        latents = rearrange(latents, "b v c f h w -> (b v f) c h w")
        images = self.vae.decode(latents.to(self.vae.dtype) / self.vae.config.scaling_factor).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        images = rearrange(images, "(b v f) c h w -> b v f h w c", v=v, f=f)
        return images.cpu().float()

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        mv_clip_images: torch.Tensor,
        mv_masks: torch.Tensor,
        poses: torch.Tensor,
        num_views: int,
        num_frames: int,
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: str = "",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "numpy",
    ):
        """Generate one shape's multi-view articulation video.

        Args:
            prompt: text description of the articulation, e.g.
                "the handle of the bucket is getting opened."
            mv_clip_images: (v, 3, H, W) CLIP-preprocessed conditioning image per view
                (the object's rest-state render, matching training's frame-0 conditioning).
            mv_masks: (v, H, W) binary part mask per view -- the spatial control signal.
            poses: (v, 3) per-view (azimuth, elevation, distance).
            output_type: "numpy" (v, f, H, W, 3) float array, "pil" nested list of
                PIL images, or "latent" the raw denoised latents.
        """
        device = self._execution_device
        do_cfg = guidance_scale > 1.0
        b = 2 if do_cfg else 1

        prompt_embeds = self._encode_prompt(prompt, negative_prompt, do_cfg, device)
        context = prompt_embeds[:, None, None].repeat(1, num_views, num_frames, 1, 1)

        image_embeds = self._encode_image(mv_clip_images, device)  # (v, n, c)
        if do_cfg:
            image_embeds = torch.stack([torch.zeros_like(image_embeds), image_embeds])  # (2, v, n, c)
        else:
            image_embeds = image_embeds[None]
        ip = image_embeds[:, :, None].repeat(1, 1, num_frames, 1, 1)

        camera = get_camera(poses).to(device=device, dtype=context.dtype)
        camera = camera[None].repeat(b, 1, 1)

        masks = mv_masks.to(device=device, dtype=context.dtype)[None].repeat(b, 1, 1, 1)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        latents = self.prepare_latents(num_views, num_frames, height, width, context.dtype, device, generator)

        for t in self.progress_bar(self.scheduler.timesteps):
            latent_input = torch.cat([latents] * b) if do_cfg else latents
            latent_input = self.scheduler.scale_model_input(latent_input, t)
            timesteps = torch.full((b,), t, device=device, dtype=torch.long)

            noise_pred = self.unet(latent_input, timesteps, context, camera=camera, ip=ip, mv_masks=masks)

            if do_cfg:
                noise_uncond, noise_cond = noise_pred.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        if output_type == "latent":
            return latents

        videos = self.decode_latents(latents)[0]  # (v, f, H, W, 3)
        if output_type == "pil":
            frames = (videos.numpy() * 255).astype(np.uint8)
            return [[Image.fromarray(frame) for frame in view] for view in frames]
        return videos.numpy()
