"""Few-shot multi-view articulation video dataset.

Reads the split files produced alongside ``datasetv0`` (e.g. ``train.txt``):
flat ``Category,shape_id`` lists, one shape per line. For each shape, every
movable part (as annotated in its PartNet-Mobility ``mobility_v2.json``) that
has a rendered video/mask/pose for every requested ``views`` becomes one
training example — the multi-view video of that part articulating, its
per-view part masks (the spatial control signal), and per-view camera poses.

Expects the on-disk layout produced by the data-preparation pipeline:

    <data_root>/<Category>/<shape_id>/
        videos/video_r_<view>_<part_idx>_<joint>.mp4
        masks/mask_r_<view>_<part_idx>_<joint>.png
        poses/pose_r_<view>_<part_idx>_<joint>.txt
        mobility_v2.json
"""

from __future__ import annotations

import json
import os
from typing import List, NamedTuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor


class _Part(NamedTuple):
    idx: int
    joint: str
    name: str


def _movable_parts(shape_dir: str) -> List[_Part]:
    """Movable ``(part_idx, joint_type, part_name)`` triples, in data-prep's ordering.

    Mirrors ``data_prep/render_masks.py::_movable_parts`` so that ``part_idx``
    lines up with the ``{part_idx}_{joint}`` motion-type used in every
    video/mask/pose filename.
    """
    with open(os.path.join(shape_dir, "mobility_v2.json")) as f:
        art_data = json.load(f)

    parts = []
    part_idx = 0
    for entry in art_data:
        if entry.get("joint") and entry.get("jointData"):
            name = entry["parts"][0]["name"]
            parts.append(_Part(part_idx, entry["joint"], name))
            part_idx += 1
    return parts


class FewShotArticulationDataset(Dataset):
    """One example = one (shape, movable part) multi-view articulation video."""

    def __init__(
        self,
        data_root: str,
        split_file: str,
        pretrained_model_path: str,
        views: List[str] = ("000", "090", "180", "270"),
        num_frames: int = 10,
        frame_start_idx: int = 0,
        frame_rate: int = 2,
        width: int = 256,
        height: int = 256,
    ):
        super().__init__()
        self.data_root = data_root
        self.views = list(views)
        self.num_frames = num_frames
        self.frame_start_idx = frame_start_idx
        self.frame_rate = frame_rate
        self.width = width
        self.height = height

        self.clip_image_processor = CLIPImageProcessor.from_pretrained(
            pretrained_model_path, subfolder="feature_extractor"
        )

        with open(split_file) as f:
            shape_refs = [line.strip().split(",") for line in f if line.strip()]

        self.examples = []
        skipped_no_mask = 0
        for category, shape_id in shape_refs:
            shape_dir = os.path.join(data_root, category, shape_id)
            if not os.path.isdir(shape_dir):
                print(f"[FewShotArticulationDataset] warning: {shape_dir} not found, skipping")
                continue
            for part in _movable_parts(shape_dir):
                motion_type = f"{part.idx}_{part.joint}"
                if self._has_all_views(shape_dir, motion_type):
                    self.examples.append((category, shape_id, shape_dir, motion_type, part.name))
                elif self._has_all_videos(shape_dir, motion_type):
                    skipped_no_mask += 1

        if skipped_no_mask:
            print(
                f"[FewShotArticulationDataset] skipped {skipped_no_mask} example(s) with videos "
                "but no rendered part masks -- run the data-preparation pipeline's `render-masks` "
                "stage for them."
            )
        if not self.examples:
            raise RuntimeError(
                f"No usable (video, mask, pose) examples found under '{data_root}' for split "
                f"'{split_file}' and views {self.views}. Make sure the data-preparation pipeline's "
                "`render-masks` stage has been run for these shapes."
            )

    def _view_paths(self, shape_dir: str, motion_type: str, view: str):
        suffix = f"r_{view}_{motion_type}"
        return {
            "video": os.path.join(shape_dir, "videos", f"video_{suffix}.mp4"),
            "mask": os.path.join(shape_dir, "masks", f"mask_{suffix}.png"),
            "pose": os.path.join(shape_dir, "poses", f"pose_{suffix}.txt"),
        }

    def _has_all_videos(self, shape_dir: str, motion_type: str) -> bool:
        return all(
            os.path.isfile(self._view_paths(shape_dir, motion_type, v)["video"]) for v in self.views
        )

    def _has_all_views(self, shape_dir: str, motion_type: str) -> bool:
        return all(
            all(os.path.isfile(p) for p in self._view_paths(shape_dir, motion_type, v).values())
            for v in self.views
        )

    def __len__(self) -> int:
        return len(self.examples)

    def _read_video(self, path: str) -> np.ndarray:
        """(num_frames, H, W, 3) uint8 RGB, resized and evenly subsampled."""
        cap = cv2.VideoCapture(path)
        frames = []
        idx = 0
        sample_indices = set(
            range(self.frame_start_idx, self.frame_start_idx + self.frame_rate * self.num_frames, self.frame_rate)
        )
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in sample_indices:
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            idx += 1
        cap.release()
        if len(frames) < self.num_frames:
            # Repeat the last frame if the clip is shorter than requested (should not
            # happen with data produced by data_prep, but keeps this robust).
            frames += [frames[-1]] * (self.num_frames - len(frames))
        return np.stack(frames[: self.num_frames], axis=0)

    def _read_mask(self, path: str) -> np.ndarray:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        return (mask > 127).astype(np.float32)

    def _read_pose(self, path: str) -> np.ndarray:
        with open(path) as f:
            azim, elev, dist = (float(v) for v in f.read().split())
        return np.array([azim, elev, dist], dtype=np.float32)

    def __getitem__(self, index: int):
        category, shape_id, shape_dir, motion_type, part_name = self.examples[index]

        videos, masks, poses, first_frames = [], [], [], []
        for view in self.views:
            paths = self._view_paths(shape_dir, motion_type, view)
            video = self._read_video(paths["video"])  # (f, h, w, 3) uint8
            videos.append(video)
            first_frames.append(video[0])
            masks.append(self._read_mask(paths["mask"]))
            poses.append(self._read_pose(paths["pose"]))

        # (v, f, h, w, 3) uint8 -> (v, f, 3, h, w) float in [-1, 1]
        mv_videos = np.stack(videos, axis=0).astype(np.float32)
        mv_videos = mv_videos.transpose(0, 1, 4, 2, 3) / 127.5 - 1.0

        # First frame of each view, CLIP-preprocessed for IP-Adapter image conditioning.
        mv_clip_images = self.clip_image_processor(
            np.stack(first_frames, axis=0), return_tensors="pt"
        ).pixel_values  # (v, 3, clip_h, clip_w)

        return {
            "mv_videos": mv_videos,
            "mv_clip_images": mv_clip_images.numpy(),
            "mv_masks": np.stack(masks, axis=0),
            "mv_poses": np.stack(poses, axis=0),
            "prompt": f"the {part_name.replace('_', ' ')} of the {category.lower()} is getting opened.",
            "category": category,
            "shape_id": shape_id,
            "motion_type": motion_type,
        }
