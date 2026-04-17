"""Shared helpers for viser visualization scripts."""

import json
import math
from pathlib import Path

import trimesh
import torch

_ROT90X = (math.sqrt(2) / 2, math.sqrt(2) / 2, 0.0, 0.0)  # wxyz


def load_tasks(output_dir: Path) -> dict[str, dict]:
    """Load all tasks that have robo.json."""
    tasks = {}
    for d in sorted(output_dir.iterdir()):
        if d.is_dir() and (d / "robo.json").exists():
            with open(d / "robo.json") as f:
                tasks[d.name] = json.load(f)
    return tasks


def parse_entry(entry: dict | list) -> tuple[list, list]:
    """Return (init_pose, final_pose) from a robo.json hand entry."""
    if isinstance(entry, dict):
        return entry["init"], entry["final"]
    return entry, entry  # legacy: entry is already the pose


def get_hand_mesh(hand_type: str, dex_pose: list, assets_root: Path) -> trimesh.Trimesh | None:
    """Get robot hand trimesh from forward kinematics (fingertip link meshes excluded)."""
    try:
        from human2robo.models import HAND_MODELS
        model = HAND_MODELS[hand_type](assets_root=assets_root, device="cpu", use_convex=False)
        out = model(torch.tensor([dex_pose], dtype=torch.float32), include_fingertip_mesh=False)
        return trimesh.Trimesh(
            vertices=out["vertices"][0].detach().numpy(),
            faces=out["faces"].numpy(),
        )
    except Exception as e:
        print(f"⚠️  [{hand_type}] FK failed: {e}")
        return None
