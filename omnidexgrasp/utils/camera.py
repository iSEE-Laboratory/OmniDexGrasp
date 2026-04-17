"""📷 Camera utilities for intrinsic parameter computation."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass
class CameraIntrinsics:
    """📐 Camera intrinsic parameters."""

    fx: float
    fy: float
    ppx: float
    ppy: float
    width: int
    height: int

def dynamic_intrinsics(
    base: CameraIntrinsics, target_width: int, target_height: int
) -> CameraIntrinsics:
    """🔄 Scale intrinsics for different image size.

    Args:
        base: Original camera intrinsics.
        target_width: Target image width.
        target_height: Target image height.

    Returns:
        Scaled camera intrinsics for the new image size.
    """
    scale_x = target_width / base.width
    scale_y = target_height / base.height
    return CameraIntrinsics(
        fx=base.fx * scale_x,
        fy=base.fy * scale_y,
        ppx=base.ppx * scale_x,
        ppy=base.ppy * scale_y,
        width=target_width,
        height=target_height,
    )


def compute_focal(intrinsics: CameraIntrinsics) -> float:
    """📐 Compute average focal length.

    Args:
        intrinsics: Camera intrinsic parameters.

    Returns:
        Average of fx and fy as focal length.
    """
    return (intrinsics.fx + intrinsics.fy) / 2


def load_k_from_yaml(camera_yaml: Path) -> np.ndarray:
    """Load 3x3 K matrix from camera.yaml."""
    with open(camera_yaml) as f:
        data = yaml.safe_load(f)
    return np.array(
        [[data["fx"], 0, data["ppx"]], [0, data["fy"], data["ppy"]], [0, 0, 1]],
        dtype=np.float32,
    )


def load_k_from_json(intrinsics_json: Path) -> np.ndarray:
    """Load 3x3 K matrix from intrinsics.json."""
    data = json.loads(intrinsics_json.read_text())
    return np.array(
        [[data["fx"], 0, data["ppx"]], [0, data["fy"], data["ppy"]], [0, 0, 1]],
        dtype=np.float32,
    )


