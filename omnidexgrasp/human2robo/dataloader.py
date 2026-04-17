"""Load optim-stage output for H2R retargeting.

Coordinate system note:
  All data is transformed to obj_cam frame (= mesh-local frame) using the
  full get_hand_transform() pipeline from real_dataset.py (reference):
    Step 1: HaMeR rotation correction (flip y/z axes)
    Step 2: Apply T matrix with scale/transl (-> obj_cam = mesh-local)
  Object point cloud stays in mesh-local (no transform needed).
  Optimization and output are all in obj_cam frame.
"""

from dataclasses import dataclass
from pathlib import Path
import json
import torch
import trimesh

from manotorch.manolayer import ManoLayer
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle


@dataclass
class RetargetData:
    """🤚 Data for one task's retargeting (all in obj_cam/mesh-local frame)."""
    task_name: str
    gt_fingertip: torch.Tensor       # (1, 5, 3)  obj_cam frame
    obj_pc: torch.Tensor             # (1, N, 3)  mesh-local = obj_cam frame
    mano_trans: torch.Tensor         # (1, 3)     wrist in obj_cam frame
    mano_axis_angle: torch.Tensor    # (1, 3)     updated global rotation in obj_cam frame
    mano_pose: torch.Tensor          # (1, 45)    finger joint angles (frame-independent)
    is_right: bool
    mano_verts_obj: torch.Tensor     # (1, V, 3)  MANO mesh vertices in obj_cam frame


class RetargetDataLoader:
    """📂 Load optim stage output for H2R retargeting."""

    def __init__(self, output_dir: Path, n_obj_pts: int, device: str,
                 mano_assets_root: str):
        self.output_dir = output_dir
        self.n_obj_pts = n_obj_pts
        self.device = device
        self._mano = ManoLayer(
            side="right",
            mano_assets_root=mano_assets_root,
            use_pca=False,
        ).to(device)

    @property
    def mano_faces(self) -> torch.Tensor:
        return self._mano.th_faces  # (F, 3) long

    def load(self, task_name: str) -> "RetargetData | None":
        task_out = self.output_dir / task_name
        manopose_path = task_out / "optim_res.json"
        mesh_path = task_out / "scaled_mesh.obj"

        if not manopose_path.exists() or not mesh_path.exists():
            return None

        with open(manopose_path) as f:
            m = json.load(f)

        fullpose   = torch.tensor(m["fullpose"],   dtype=torch.float32).to(self.device)  # (1, 48)
        betas      = torch.tensor(m["betas"],      dtype=torch.float32).to(self.device)  # (1, 10)
        cam_transl = torch.tensor(m["cam_transl"], dtype=torch.float32).to(self.device)  # (1, 3)
        T          = torch.tensor(m["T"],          dtype=torch.float32).to(self.device)  # (4, 4)
        is_right   = bool(m["is_right"])

        scale_raw = m["hand_params"]["scale"]
        scale  = float(scale_raw[0]) if isinstance(scale_raw, list) else float(scale_raw)
        transl = torch.tensor(m["hand_params"]["transl"], dtype=torch.float32).to(self.device)  # (3,)

        # MANO FK -> joints in MANO canonical space
        mano_out = self._mano(fullpose, betas)
        joints_local = mano_out.joints   # (1, J, 3)
        verts_local  = mano_out.verts      # (1, V, 3)

        # ── Replicate real_dataset.py get_hand_transform() ──────────────────
        # Step 1: HaMeR rotation correction (flip y/z; diagonal matrix so .T == self)
        hamer_rot = torch.tensor([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
                                 dtype=torch.float32, device=self.device)  # (3, 3)
        rot   = hamer_rot.clone()
        trans = cam_transl.squeeze(0) @ hamer_rot.T  # (3,)

        # Step 2: Apply T matrix (cam -> obj_cam = mesh-local frame)
        rot   = rot @ (scale * T[:3, :3].T)          # (3, 3)
        trans = trans @ (scale * T[:3, :3].T) + (T[:3, 3] * scale + transl)  # (3,)
        # ────────────────────────────────────────────────────────────────────

        # Apply combined (rot, trans) to MANO joints -> obj_cam frame
        joints_obj = joints_local @ rot + trans       # (1, J, 3)
        verts_obj  = verts_local  @ rot + trans       # (1, V, 3)

        # Fingertips: indices [4, 8, 12, 16, 20] -> thumb/index/middle/ring/pinky
        gt_fingertip = joints_obj[:, [4, 8, 12, 16, 20]]   # (1, 5, 3)

        # Wrist position in obj_cam frame
        mano_trans = joints_obj[:, 0]                       # (1, 3)

        # Update global orient to obj_cam frame: R_new = rot.T @ R_old
        R_old = axis_angle_to_matrix(fullpose[:, :3])       # (1, 3, 3)
        R_new = rot.T.unsqueeze(0) @ R_old                  # (1, 3, 3)
        new_axis_angle = matrix_to_axis_angle(R_new)        # (1, 3)

        # Object point cloud: area-weighted surface sampling (match real_dataset.py:133)
        mesh = trimesh.load(str(mesh_path))
        pts_surface = mesh.sample(self.n_obj_pts)
        pts_obj = torch.tensor(pts_surface, dtype=torch.float32, device=self.device)
        obj_pc = pts_obj.unsqueeze(0)                       # (1, N, 3)

        return RetargetData(
            task_name=task_name,
            gt_fingertip=gt_fingertip,
            obj_pc=obj_pc,
            mano_trans=mano_trans,
            mano_axis_angle=new_axis_angle,
            mano_pose=fullpose[:, 3:],
            is_right=is_right,
            mano_verts_obj=verts_obj,
        )
