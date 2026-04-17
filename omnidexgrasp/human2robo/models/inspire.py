"""Inspire Hand model (12 controllable DOF, 18 total pose dims)."""

from pathlib import Path
import torch
import pytorch3d.transforms as T
from .base import RobotHandModel


# Fingertip link names (from dex-retargeting config)
_FINGERTIP_LINKS = ["R_thumb_tip", "R_index_tip", "R_middle_tip", "R_ring_tip", "R_little_tip"]

# Mimic joint constraints: mimic_joint = base_joint * multiplier + offset
_MIMIC = {
    "right_thumb_3_joint":  ("right_thumb_2_joint",  0.6,  0.0),
    "right_thumb_4_joint":  ("right_thumb_2_joint",  0.8,  0.0),
    "right_index_2_joint":  ("right_index_1_joint",  1.05, 0.0),
    "right_middle_2_joint": ("right_middle_1_joint", 1.05, 0.0),
    "right_ring_2_joint":   ("right_ring_1_joint",   1.05, 0.0),
    "right_little_2_joint": ("right_little_1_joint", 1.18, 0.0),
}

# Coordinate transform: MANO camera frame → Inspire robot frame
_ROT1 = torch.tensor([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
_ROT2 = torch.tensor([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
_LOCAL_OFFSET = torch.tensor([0., 0., 0.05])


class InspireModel(RobotHandModel):
    """Inspire Hand (12-DOF controllable joints, 18-dim pose vector)."""

    dof_total = 18  # 3(trans) + 3(rot) + 12(joints)

    def __init__(self, assets_root: Path, device: str = "cuda", use_convex: bool = True):
        urdf_path = Path(assets_root) / "inspire_hand_ftp" / "urdf" / "inspire_right.urdf"
        mesh_dir  = Path(assets_root) / "inspire_hand_ftp" / "meshes"
        super().__init__(urdf_path, mesh_dir, _FINGERTIP_LINKS, device, use_convex)

    def mano2robot_batch(
        self,
        mano_trans: torch.Tensor,       # (B, 3) wrist position
        mano_axis_angle: torch.Tensor,  # (B, 3) global rotation
        mano_pose: torch.Tensor,        # (B, 45) finger joints
    ) -> torch.Tensor:
        """Map MANO parameters to initial Inspire dex_pose (B, 18)."""
        device = mano_trans.device
        B = mano_trans.shape[0]
        rot1 = _ROT1.to(device)
        rot2 = _ROT2.to(device)
        offset = _LOCAL_OFFSET.to(device)

        # Transform global rotation
        global_aa = T.matrix_to_axis_angle(
            T.axis_angle_to_matrix(mano_axis_angle) @ rot1 @ rot2
        )
        R = T.axis_angle_to_matrix(global_aa)
        trans = mano_trans + torch.matmul(R, offset)  # (B, 3)

        # Map finger joints: MANO 45-dim → Inspire 12-dim (Z-axis flexion only)
        mano_aa = mano_pose.reshape(B, -1, 3)  # (B, 15, 3)
        euler = T.matrix_to_euler_angles(T.axis_angle_to_matrix(mano_aa), "XYZ")  # (B, 15, 3)
        joints = torch.zeros(B, 12, device=device)
        joints[:, 0]  = 1.0                  # Thumb base fixed
        joints[:, 4]  = euler[:, 0, 2]       # Index  MCP flexion
        joints[:, 6]  = euler[:, 3, 2]       # Middle MCP flexion
        joints[:, 8]  = euler[:, 9, 2]       # Ring   MCP flexion
        joints[:, 10] = euler[:, 6, 2]       # Pinky  MCP flexion

        return torch.cat([trans, global_aa, joints], dim=-1)  # (B, 18)

    def _get_joint_pose(self, joint_pose: torch.Tensor) -> torch.Tensor:
        """Apply mimic joint constraints (Inspire-specific)."""
        joint_names = self.chain.get_joint_parameter_names()
        name2idx = {n: i for i, n in enumerate(joint_names)}
        result = joint_pose.clone()
        for mimic_name, (base_name, mult, offset) in _MIMIC.items():
            if mimic_name in name2idx and base_name in name2idx:
                result[:, name2idx[mimic_name]] = (
                    result[:, name2idx[base_name]] * mult + offset
                )
        return result
