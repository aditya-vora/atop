#!/usr/bin/env python
"""Finetune the ATOP multi-view video diffusion model.

Starts from the pretrained multi-view *image* checkpoint in
``checkpoints/imagedream-ipmv-diffusers`` and finetunes it into a
spatially-controllable multi-view *video* model: given a static object, a
part-segmentation mask, and a text prompt, it generates a synchronized
multi-view video of that part articulating (see arXiv:2502.07278).

Training data is the few-shot split produced by the data-preparation pipeline:
by default, ``datasetv0/train.txt``, a flat ``Category,shape_id`` list. Every
movable part of every listed shape that has a rendered video/mask/pose (for
every view in ``--views``) becomes one training example.

Only a small subset of the UNet is trained (`--trainable_modules`): the
cross-view/IP-Adapter attention projections carried over from the pretrained
image model, and the newly added ``attn_temp`` module (temporal attention +
the part-mask FiLM conditioning) -- see ``src/models/mv_unet.py`` for why the
temporal/mask path has to be new rather than finetuned in place.

Example
-------
    python finetune.py \\
        --pretrained_model_path checkpoints/imagedream-ipmv-diffusers \\
        --data_root datasetv0 \\
        --train_split datasetv0/train.txt \\
        --output_dir output/atop-finetune
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel

from src.data.dataset import FewShotArticulationDataset
from src.models.mv_unet import MultiViewUNetModel
from src.utils.cam_utils import get_camera

logger = get_logger(__name__)

DEFAULT_TRAINABLE_MODULES = (
    "attn1.to_q",
    "attn2.to_q",
    "attn2.to_k_ip",
    "attn2.to_v_ip",
    "attn_temp",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--pretrained_model_path", default="checkpoints/imagedream-ipmv-diffusers",
                         help="Diffusers-format multi-view image checkpoint to finetune from.")
    parser.add_argument("--data_root", default="datasetv0",
                         help="Root of the data-preparation pipeline's output.")
    parser.add_argument("--train_split", default="datasetv0/train.txt",
                         help="Few-shot split file: 'Category,shape_id' per line.")
    parser.add_argument("--output_dir", default="output")

    parser.add_argument("--views", nargs="+", default=["000", "090", "180", "270"],
                         help="Which rendered azimuths to train on (subset of the data-preparation pipeline's azimuths).")
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--frame_start_idx", type=int, default=0)
    parser.add_argument("--frame_rate", type=int, default=2, help="Stride between sampled video frames.")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)

    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_dataloader_workers", type=int, default=4)
    parser.add_argument("--max_train_steps", type=int, default=4000)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--lr_scheduler", default="constant",
                         help="See diffusers.optimization.get_scheduler for choices.")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--guidance_dropout", type=float, default=0.1,
                         help="Probability of replacing the prompt with '' during training, for classifier-free guidance at sampling time.")

    parser.add_argument("--trainable_modules", nargs="+", default=list(DEFAULT_TRAINABLE_MODULES))
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--resume_from_checkpoint", default=None,
                         help="Path to a checkpoint-<step> directory saved by a previous run.")
    parser.add_argument("--mixed_precision", default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def freeze_and_select_trainable(unet: torch.nn.Module, trainable_modules) -> list:
    unet.requires_grad_(False)
    trainable_params = []
    for name, module in unet.named_modules():
        if name.endswith(tuple(trainable_modules)):
            for p in module.parameters():
                p.requires_grad_(True)
                trainable_params.append(p)
    return trainable_params


def save_checkpoint(output_dir, unet, tokenizer, text_encoder, vae, image_encoder, feature_extractor, scheduler):
    """Write a standalone, diffusers-loadable copy of the finetuned model.

    Mirrors the layout of the pretrained checkpoint this was finetuned from
    (component subfolders + model_index.json), so the result can be reloaded
    the same way -- including the custom UNet code, copied in alongside its
    weights per diffusers' custom-pipeline convention.
    """
    os.makedirs(output_dir, exist_ok=True)
    unet.save_pretrained(os.path.join(output_dir, "unet"))
    shutil.copy(
        os.path.join(os.path.dirname(__file__), "src", "models", "mv_unet.py"),
        os.path.join(output_dir, "unet", "mv_unet.py"),
    )
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
    text_encoder.save_pretrained(os.path.join(output_dir, "text_encoder"))
    vae.save_pretrained(os.path.join(output_dir, "vae"))
    image_encoder.save_pretrained(os.path.join(output_dir, "image_encoder"))
    feature_extractor.save_pretrained(os.path.join(output_dir, "feature_extractor"))
    scheduler.save_pretrained(os.path.join(output_dir, "scheduler"))

    model_index = {
        "_class_name": "MVDreamPipeline",
        "feature_extractor": ["transformers", "CLIPImageProcessor"],
        "image_encoder": ["transformers", "CLIPVisionModel"],
        "requires_safety_checker": False,
        "scheduler": ["diffusers", "DDIMScheduler"],
        "text_encoder": ["transformers", "CLIPTextModel"],
        "tokenizer": ["transformers", "CLIPTokenizer"],
        "unet": ["mv_unet", "MultiViewUNetModel"],
        "vae": ["diffusers", "AutoencoderKL"],
    }
    with open(os.path.join(output_dir, "model_index.json"), "w") as f:
        json.dump(model_index, f, indent=2)


def main():
    args = parse_args()
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    logger.info(accelerator.state, main_process_only=False)
    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    num_views = len(args.views)

    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
    image_encoder = CLIPVisionModel.from_pretrained(args.pretrained_model_path, subfolder="image_encoder")
    unet = MultiViewUNetModel.from_pretrained_2d(
        args.pretrained_model_path, subfolder="unet", num_frames=args.num_frames, num_views=num_views
    )

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    trainable_params = freeze_and_select_trainable(unet, args.trainable_modules)
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in unet.parameters())
    logger.info(f"Training {n_trainable:,} / {n_total:,} UNet parameters ({100 * n_trainable / n_total:.1f}%).")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    train_dataset = FewShotArticulationDataset(
        data_root=args.data_root,
        split_file=args.train_split,
        pretrained_model_path=args.pretrained_model_path,
        views=args.views,
        num_frames=args.num_frames,
        frame_start_idx=args.frame_start_idx,
        frame_rate=args.frame_rate,
        width=args.width,
        height=args.height,
    )
    logger.info(f"Few-shot training set: {len(train_dataset)} examples from '{args.train_split}'.")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_dataloader_workers,
        drop_last=True,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint)
        global_step = int(os.path.basename(args.resume_from_checkpoint.rstrip("/")).split("-")[-1])
        logger.info(f"Resumed from {args.resume_from_checkpoint} at step {global_step}")

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("steps")

    for _ in range(num_train_epochs):
        unet.train()
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                bsz = batch["mv_videos"].shape[0]

                prompts = batch["prompt"]
                if args.guidance_dropout > 0:
                    prompts = ["" if torch.rand(()) < args.guidance_dropout else p for p in prompts]
                prompt_ids = tokenizer(
                    prompts,
                    max_length=tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(accelerator.device)
                prompt_embeds = text_encoder(prompt_ids)[0]

                mv_videos = batch["mv_videos"].to(accelerator.device, dtype=weight_dtype)
                mv_clip_images = batch["mv_clip_images"].to(accelerator.device, dtype=weight_dtype)
                mv_masks = batch["mv_masks"].to(accelerator.device, dtype=weight_dtype)
                mv_poses = batch["mv_poses"]

                num_frames = mv_videos.shape[2]

                videos_in = rearrange(mv_videos, "b v f c h w -> (b v f) c h w")
                latents = vae.encode(videos_in).latent_dist.sample() * vae.config.scaling_factor
                latents = rearrange(latents, "(b v f) c h w -> b v c f h w", v=num_views, f=num_frames)

                clip_images_in = rearrange(mv_clip_images, "b v c h w -> (b v) c h w")
                clip_embeds = image_encoder(clip_images_in, output_hidden_states=True).hidden_states[-2]
                clip_embeds = rearrange(clip_embeds, "(b v) n c -> b v n c", v=num_views)
                ip = clip_embeds[:, :, None].repeat(1, 1, num_frames, 1, 1)

                camera = torch.stack([get_camera(mv_poses[i]) for i in range(bsz)])
                camera = camera.to(accelerator.device, dtype=weight_dtype)

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                context = prompt_embeds[:, None, None].repeat(1, num_views, num_frames, 1, 1)

                model_pred = unet(noisy_latents, timesteps, context, camera=camera, ip=ip, mv_masks=mv_masks)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unsupported prediction type: {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                progress_bar.set_postfix(loss=loss.detach().item(), lr=lr_scheduler.get_last_lr()[0])

                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved training state to {save_path}")

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "final_checkpoint")
        save_checkpoint(
            final_dir,
            accelerator.unwrap_model(unet),
            tokenizer,
            text_encoder,
            vae,
            image_encoder,
            train_dataset.clip_image_processor,
            noise_scheduler,
        )
        logger.info(f"Saved finetuned checkpoint to {final_dir}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
