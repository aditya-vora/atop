"""Camera-pose conversion for the multi-view UNet's camera embedding.

The data-prep pipeline stores each view's camera as a single spherical pose
``azimuth elevation distance``. This module converts a batch of those raw
poses into the flattened 4x4 camera-to-world matrices the UNet's
``camera_embed`` head expects.
"""

from __future__ import annotations

import numpy as np
import torch


def _safe_normalize(v: np.ndarray, eps: float = 1e-20) -> np.ndarray:
    return v / max(np.linalg.norm(v), eps)


def _look_at(campos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation matrix (3, 3) for a camera at ``campos`` looking at ``target``."""
    forward = _safe_normalize(campos - target)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = _safe_normalize(np.cross(up, forward))
    up = _safe_normalize(np.cross(forward, right))
    return np.stack([right, up, forward], axis=1)


def _orbit_camera(elevation: float, azimuth: float, radius: float = 1.0) -> np.ndarray:
    """Camera-to-world pose (4, 4) orbiting the origin at (elevation, azimuth)."""
    elevation, azimuth = np.deg2rad(elevation), np.deg2rad(azimuth)
    x = radius * np.cos(elevation) * np.sin(azimuth)
    y = radius * np.sin(elevation)
    z = radius * np.cos(elevation) * np.cos(azimuth)
    campos = np.array([x, y, z], dtype=np.float32)
    target = np.zeros(3, dtype=np.float32)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = _look_at(campos, target)
    pose[:3, 3] = campos
    return pose


def get_camera(poses: torch.Tensor, blender_coord: bool = True) -> torch.Tensor:
    """Convert raw ``(azim, elev, dist)`` poses into flattened camera embeddings.

    Args:
        poses: ``(num_views, 3)`` tensor of ``(azimuth, elevation, distance)``
            in degrees/scene-units, as stored in ``poses/pose_r_*.txt``.
            Azimuths follow the data-prep camera rig's convention: measured
            from Blender's +X axis, offset by 90 degrees below to align with
            this module's orbit-camera parameterization.
        blender_coord: apply the Blender <-> OpenGL axis swap used throughout
            the data-prep renders.

    Returns:
        ``(num_views, 16)`` float tensor: each row is a flattened 4x4
        camera-to-world matrix, ready to feed into the UNet's camera
        embedding head.
    """
    poses_np = poses.detach().cpu().numpy()
    cams = []
    for azim, elev, radius in poses_np:
        if azim <= 0:
            azim = 180 + abs(180 + azim)
        pose = _orbit_camera(elev, azim + 90, radius=radius)
        if blender_coord:
            pose[2] *= -1
            pose[[1, 2]] = pose[[2, 1]]
        cams.append(pose.flatten())
    return torch.from_numpy(np.stack(cams, axis=0)).float()
