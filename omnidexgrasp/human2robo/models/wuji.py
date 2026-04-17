"""Wuji Hand model (20 controllable DOF, 26-dim pose vector)."""

from pathlib import Path
import torch
import pytorch3d.transforms as T
from .base import RobotHandModel


_FINGERTIP_LINKS = [
    "finger1_tip_link", "finger2_tip_link", "finger3_tip_link",
    "finger4_tip_link", "finger5_tip_link",
]

# Coordinate transform: same as Inspire
_ROT1 = torch.tensor([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
_ROT2 = torch.tensor([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
_LOCAL_OFFSET = torch.tensor([0., 0., 0.0])


class WujiModel(RobotHandModel):
    """Wuji Hand (20-DOF controllable joints, 26-dim pose vector)."""

    dof_total = 26  # 3(trans) + 3(rot) + 20(joints)

    def __init__(self, assets_root: Path, device: str = "cuda", use_convex: bool = True):
        urdf_path = Path(assets_root) / "wuji_hand" / "urdf" / "right.urdf"
        mesh_dir  = Path(assets_root) / "wuji_hand" / "meshes" / "right"
        super().__init__(urdf_path, mesh_dir, _FINGERTIP_LINKS, device, use_convex)

    def mano2robot_batch(
        self,
        mano_trans: torch.Tensor,       # (B, 3)
        mano_axis_angle: torch.Tensor,  # (B, 3)
        mano_pose: torch.Tensor,        # (B, 45)
    ) -> torch.Tensor:
        """Map MANO parameters to initial Wuji dex_pose (B, 26)."""
        device = mano_trans.device
        B = mano_trans.shape[0]
        rot1 = _ROT1.to(device)
        rot2 = _ROT2.to(device)
        offset = _LOCAL_OFFSET.to(device)

        global_aa = T.matrix_to_axis_angle(
            T.axis_angle_to_matrix(mano_axis_angle) @ rot1 @ rot2
        )
        R = T.axis_angle_to_matrix(global_aa)
        trans = mano_trans + torch.matmul(R, offset)

        mano_aa = mano_pose.reshape(B, -1, 3)
        euler = T.matrix_to_euler_angles(T.axis_angle_to_matrix(mano_aa), "XYZ")  # (B, 15, 3)

        joints = torch.zeros(B, 20, device=device)
        # Thumb  (MANO 12-13 → Wuji 0-3)
        joints[:, 0] = euler[:, 12, 2]         # MCP flexion
        joints[:, 1] = euler[:, 12, 1]         # MCP abduction
        joints[:, 2] = euler[:, 13, 2] * 0.5   # PIP flexion
        joints[:, 3] = euler[:, 13, 2] * 0.3   # DIP flexion
        # Index  (MANO 0-2 → Wuji 4-7)
        joints[:, 4] = euler[:, 0, 2];  joints[:, 5] = euler[:, 0, 1]
        joints[:, 6] = euler[:, 1, 2];  joints[:, 7] = euler[:, 2, 2]
        # Middle (MANO 3-5 → Wuji 8-11)
        joints[:, 8] = euler[:, 3, 2];  joints[:, 9] = euler[:, 3, 1]
        joints[:, 10] = euler[:, 4, 2]; joints[:, 11] = euler[:, 5, 2]
        # Ring   (MANO 9-11 → Wuji 12-15)
        joints[:, 12] = euler[:, 9, 2];  joints[:, 13] = euler[:, 9, 1]
        joints[:, 14] = euler[:, 10, 2]; joints[:, 15] = euler[:, 11, 2]
        # Pinky  (MANO 6-8 → Wuji 16-19)
        joints[:, 16] = euler[:, 6, 2];  joints[:, 17] = euler[:, 6, 1]
        joints[:, 18] = euler[:, 7, 2];  joints[:, 19] = euler[:, 8, 2]

        return torch.cat([trans, global_aa, joints], dim=-1)  # (B, 26)
