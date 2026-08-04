#!/usr/bin/env python
"""Run inference with a finetuned ATOP multi-view video diffusion checkpoint.

Loads a checkpoint produced by ``finetune.py`` (e.g.
``output/<run>/final_checkpoint``) and, for each shape/part in a data-prep
split (default ``datasetv0/test.txt``), generates a synchronized multi-view
video of that part articulating -- conditioned on the shape's rest-state
per-view renders, its part mask (the spatial control signal), and a text
prompt.

Example
-------
    python infer.py \\
        --checkpoint output/atop-finetune/final_checkpoint \\
        --data_root datasetv0 \\
        --test_split datasetv0/test.txt \\
        --output_dir output/atop-finetune/inference
"""

from __future__ import annotations

import argparse
import logging
import os

import imageio.v2 as imageio
import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from PIL import Image
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModel

from src.data.dataset import FewShotArticulationDataset
from src.models.mv_unet import MultiViewUNetModel
from src.pipelines.mv_video_pipeline import MultiViewVideoPipeline

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

DTYPES = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--checkpoint", required=True,
                         help="Finetuned checkpoint directory (see finetune.py's --output_dir/final_checkpoint).")
    parser.add_argument("--data_root", default="datasetv0")
    parser.add_argument("--test_split", default="datasetv0/test.txt",
                         help="Split file to run inference over: 'Category,shape_id' per line.")
    parser.add_argument("--output_dir", default="output/inference")

    parser.add_argument("--views", nargs="+", default=["000", "090", "180", "270"],
                         help="Must match the views the checkpoint was finetuned with.")
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--frame_start_idx", type=int, default=0)
    parser.add_argument("--frame_rate", type=int, default=2)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)

    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--prompt", default=None,
                         help="Override every example's auto-generated prompt with this one.")
    parser.add_argument("--num_examples", type=int, default=None,
                         help="Only run the first N examples of the split (default: all).")
    parser.add_argument("--fps", type=int, default=5)

    parser.add_argument("--mixed_precision", default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def build_pipeline(checkpoint: str, dtype: torch.dtype) -> MultiViewVideoPipeline:
    tokenizer = CLIPTokenizer.from_pretrained(checkpoint, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(checkpoint, subfolder="text_encoder", torch_dtype=dtype)
    vae = AutoencoderKL.from_pretrained(checkpoint, subfolder="vae", torch_dtype=dtype)
    image_encoder = CLIPVisionModel.from_pretrained(checkpoint, subfolder="image_encoder", torch_dtype=dtype)
    feature_extractor = CLIPImageProcessor.from_pretrained(checkpoint, subfolder="feature_extractor")
    scheduler = DDIMScheduler.from_pretrained(checkpoint, subfolder="scheduler")
    unet = MultiViewUNetModel.from_pretrained(checkpoint, subfolder="unet", torch_dtype=dtype)

    return MultiViewVideoPipeline(
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
        feature_extractor=feature_extractor,
        image_encoder=image_encoder,
    )


def save_multiview_gif(video: np.ndarray, path: str, fps: int) -> None:
    """video: (v, f, H, W, 3) float in [0, 1] -> one gif with views tiled side by side."""
    v, f, h, w, c = video.shape
    frames = (video * 255).clip(0, 255).astype(np.uint8)
    grid = frames.transpose(1, 2, 0, 3, 4).reshape(f, h, v * w, c)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, list(grid), fps=fps)


def save_view_frames(video: np.ndarray, out_dir: str, views) -> None:
    """video: (v, f, H, W, 3) float in [0, 1] -> per-view PNG frame sequences."""
    frames = (video * 255).clip(0, 255).astype(np.uint8)
    for view, per_view in zip(views, frames):
        view_dir = os.path.join(out_dir, f"view_{view}")
        os.makedirs(view_dir, exist_ok=True)
        for i, frame in enumerate(per_view):
            Image.fromarray(frame).save(os.path.join(view_dir, f"frame_{i:03d}.png"))


def save_conditioning(example: dict, out_dir: str, views) -> None:
    """Save the per-view rest-state image and part mask the video was conditioned on."""
    cond_dir = os.path.join(out_dir, "conditioning")
    os.makedirs(cond_dir, exist_ok=True)
    first_frames = ((example["mv_videos"][:, 0].transpose(0, 2, 3, 1) + 1.0) / 2.0 * 255.0)
    first_frames = first_frames.clip(0, 255).astype(np.uint8)
    masks = (example["mv_masks"] * 255).astype(np.uint8)
    for view, image, mask in zip(views, first_frames, masks):
        Image.fromarray(image).save(os.path.join(cond_dir, f"image_{view}.png"))
        Image.fromarray(mask).save(os.path.join(cond_dir, f"mask_{view}.png"))


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = DTYPES[args.mixed_precision]

    logger.info(f"Loading checkpoint from {args.checkpoint} (device={device}, dtype={dtype})")
    pipe = build_pipeline(args.checkpoint, dtype)
    pipe = pipe.to(device)

    dataset = FewShotArticulationDataset(
        data_root=args.data_root,
        split_file=args.test_split,
        pretrained_model_path=args.checkpoint,
        views=args.views,
        num_frames=args.num_frames,
        frame_start_idx=args.frame_start_idx,
        frame_rate=args.frame_rate,
        width=args.width,
        height=args.height,
    )
    num_examples = min(args.num_examples or len(dataset), len(dataset))
    logger.info(f"Running inference on {num_examples}/{len(dataset)} example(s) from '{args.test_split}'.")

    generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed is not None else None

    for i in range(num_examples):
        example = dataset[i]
        prompt = args.prompt or example["prompt"]
        name = f"{example['category']}_{example['shape_id']}_{example['motion_type']}"
        logger.info(f"[{i + 1}/{num_examples}] {name}: '{prompt}'")

        example_out_dir = os.path.join(args.output_dir, name)
        save_conditioning(example, example_out_dir, args.views)

        video = pipe(
            prompt=prompt,
            mv_clip_images=torch.from_numpy(example["mv_clip_images"]),
            mv_masks=torch.from_numpy(example["mv_masks"]),
            poses=torch.from_numpy(example["mv_poses"]),
            num_views=len(args.views),
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            negative_prompt=args.negative_prompt,
            generator=generator,
            output_type="numpy",
        )

        save_multiview_gif(video, os.path.join(example_out_dir, "video.gif"), fps=args.fps)
        save_view_frames(video, os.path.join(example_out_dir, "frames"), args.views)

    logger.info(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
