"""📦 Data loading for MANO hand optimization.

Loads client.py flat output + dataset source images.
"""
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image
from mesh_to_sdf import mesh_to_voxels
from pyrender.camera import IntrinsicsCamera

# 🔄 y-down/z-forward -> y-up/z-back
COORD_CORRECTION = torch.tensor([
    [1.0,  0.0,  0.0, 0.0],
    [0.0, -1.0,  0.0, 0.0],
    [0.0,  0.0, -1.0, 0.0],
    [0.0,  0.0,  0.0, 1.0],
])


def compute_sdf(
    mesh: trimesh.Trimesh, cache_dir: Path, device: torch.device
) -> tuple[dict | None, torch.Tensor]:
    """📐 Compute or load cached SDF voxel grid."""
    obj_verts = torch.tensor(mesh.vertices).float().to(device)
    mesh_copy = trimesh.Trimesh(obj_verts.cpu().numpy(), mesh.faces)

    sdf_dir = cache_dir / "SDF"
    sdf_dir.mkdir(parents=True, exist_ok=True)
    sdf_path = sdf_dir / "sdf.npy"

    sdf_voxel = None
    if sdf_path.exists():
        sdf_voxel = np.load(sdf_path, allow_pickle=True)
        if sdf_voxel.size == 1 and sdf_voxel.item() is None:
            sdf_voxel = None

    if sdf_voxel is None:
        logging.info("🔨 Computing object mesh SDF...")
        for attempt in range(1, 6):
            random.seed(attempt)
            np.random.seed(attempt)
            try:
                sdf_voxel = mesh_to_voxels(
                    mesh_copy, voxel_resolution=64, check_result=True,
                    surface_point_method="scan", sign_method="depth",
                    sample_point_count=500000,
                )
                break
            except Exception as e:
                logging.warning(f"  SDF attempt {attempt}/5 failed: {e}")
        if sdf_voxel is not None:
            np.save(sdf_path, sdf_voxel)

    if sdf_voxel is None:
        return None, obj_verts

    return {
        "origin": torch.FloatTensor(mesh_copy.bounding_box.centroid.copy()).to(device),
        "scale": torch.FloatTensor([2.0 / np.max(mesh_copy.bounding_box.extents)]).to(device),
        "voxel": torch.FloatTensor(sdf_voxel).to(device),
    }, obj_verts


def _load_cam(path: Path, device: torch.device) -> dict:
    """📷 Load normalized camera params JSON → dict with tensors."""
    with open(path) as f:
        cam = json.load(f)
    cam["extrinsics"] = torch.tensor(cam["extrinsics"]).to(device)
    cam["intrinsic"] = torch.tensor([
        [cam["fx"], 0, cam["cx"]],
        [0, cam["fy"], cam["cy"]],
        [0, 0, 1],
    ]).to(device)
    return cam


def _get_projection(cam: dict, width: int, height: int) -> np.ndarray:
    """📐 Build projection matrix from normalized camera params."""
    fx, fy = cam["fx"] * width, cam["fy"] * height
    cx, cy = cam["cx"] * width, cam["cy"] * height
    return IntrinsicsCamera(fx, fy, cx, cy).get_projection_matrix(width, height)


class OptimDataLoader:
    """📦 Data loader for MANO optimization.

    Args:
        data_dir: datasets/<task_name>/ — source images
        output_dir: out/<task_name>/ — client.py flat output
        device: torch device
    """

    def __init__(self, data_dir: Path, output_dir: Path, device: str = "cuda"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.device = torch.device(device)

    def load_data(self) -> dict | None:
        """🚀 Load all data for HOI optimization."""
        out = self.output_dir

        # 🖼️ grasp image
        img_path = self.data_dir / "generated_human_grasp.png"
        image = Image.open(img_path)
        origin_h, origin_w = image.height, image.width

        # 🎭 seg_mask.png → hand_mask + obj_mask
        seg = np.array(Image.open(out / "seg_mask.png").convert("L"))
        # rendered inpaint mask from pose_est (grasp pose render), NOT seg obj_mask
        inpaint = np.array(Image.open(out / "obj_mask.png").convert("L"))

        # 📐 object mesh + pose
        mesh_path = out / "scaled_mesh.obj"
        obj_mesh = trimesh.load(str(mesh_path))
        object_colors = getattr(obj_mesh.visual, "vertex_colors", None)

        # 📷 Load normalized camera params (shared between hand and object)
        cam_params = _load_cam(out / "camera_params.json", self.device)

        # 📐 object camera = same intrinsics, pose-derived extrinsics
        # pose_est.json: {"grasp": {"pose": 4x4 T_CO matrix, "score": float}, "scene": {...}}
        with open(out / "pose_est.json") as f:
            object_pose = torch.tensor(json.load(f)["grasp"]["pose"], device=self.device).float()
        obj_extrinsics = torch.linalg.inv(object_pose) @ COORD_CORRECTION.to(self.device)
        obj_cam = {
            "fx": cam_params["fx"], "fy": cam_params["fy"],
            "cx": cam_params["cx"], "cy": cam_params["cy"],
            "extrinsics": obj_extrinsics,
            "intrinsic": cam_params["intrinsic"],
        }
        obj_cam["projection"] = torch.FloatTensor(
            _get_projection(obj_cam, origin_w, origin_h)
        ).to(self.device)

        # 🤚 hand camera = same intrinsics, identity extrinsics (from camera_params.json)
        hand_cam = cam_params
        hand_cam["projection"] = torch.tensor(
            _get_projection(hand_cam, origin_w, origin_h)
        ).float().to(self.device)

        # 🤚 hand params
        hand_info = torch.load(out / "hand_params.pt", weights_only=False)
        mano_params = {
            k: torch.tensor(v[0]).to(self.device).unsqueeze(0)
            for k, v in hand_info["mano_params"].items()
        }

        # 📐 SDF
        sdf_cache_dir = self.output_dir / "data" / "optim"
        obj_sdf, obj_verts = compute_sdf(obj_mesh, sdf_cache_dir, self.device)
        if obj_sdf is None:
            logging.error("❌ SDF computation failed")
            return None

        return {
            "name": self.data_dir.name,
            "img_path": str(img_path),
            "resolution": [origin_h, origin_w],
            "image": image,
            "hand_mask": torch.tensor(seg == 128).to(self.device),
            "obj_mask": torch.tensor(seg == 255).to(self.device),
            "inpaint_mask": torch.tensor(inpaint > 0).to(self.device),
            "hand_cam": hand_cam,
            "obj_cam": obj_cam,
            "mano_params": mano_params,
            "object_verts": obj_verts.unsqueeze(0),
            "object_pose": object_pose,
            "object_faces": torch.LongTensor(obj_mesh.faces).to(self.device),
            "object_colors": object_colors,
            "object_sdf": obj_sdf,
            "cam_transl": torch.tensor(hand_info["cam_transl"][0]).unsqueeze(0).float().to(self.device),
            "is_right": hand_info["is_right"][0],
            "mesh_path": str(mesh_path),
        }
