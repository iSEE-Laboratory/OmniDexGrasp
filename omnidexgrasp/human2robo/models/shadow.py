"""Shadow Hand model (22 controllable DOF, 28-dim pose vector)."""

from pathlib import Path
import torch
import pytorch3d.transforms as T
from .base import RobotHandModel


_FINGERTIP_LINKS = [
    "robot0:thdistal", "robot0:ffdistal", "robot0:mfdistal",
    "robot0:rfdistal", "robot0:lfdistal",
]

# Shadow: identity coordinate transform
_ROT1 = torch.eye(3)
_ROT2 = torch.eye(3)
_LOCAL_OFFSET = torch.tensor([0., 0., 0.])


class ShadowModel(RobotHandModel):
    """Shadow Hand (22-DOF controllable joints, 28-dim pose vector)."""

    dof_total = 28  # 3(trans) + 3(rot) + 22(joints)

    def __init__(self, assets_root: Path, device: str = "cuda", use_convex: bool = True):
        urdf_path = Path(assets_root) / "shadow_hand" / "shadowhand.urdf"
        mesh_dir  = Path(assets_root) / "shadow_hand"
        super().__init__(urdf_path, mesh_dir, _FINGERTIP_LINKS, device, use_convex)

    def mano2robot_batch(
        self,
        mano_trans: torch.Tensor,       # (B, 3)
        mano_axis_angle: torch.Tensor,  # (B, 3)
        mano_pose: torch.Tensor,        # (B, 45)
    ) -> torch.Tensor:
        """Map MANO parameters to initial Shadow dex_pose (B, 28)."""
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

        joints = torch.zeros(B, 22, device=device)
        # Index  (FFJ, MANO 0-2 → Shadow 0-3)
        joints[:, 0] = euler[:, 0, 1]   # FFJ3: abduction
        joints[:, 1] = euler[:, 0, 2]   # FFJ2: MCP flexion
        joints[:, 2] = euler[:, 1, 2]   # FFJ1: PIP flexion
        joints[:, 3] = euler[:, 2, 2]   # FFJ0: DIP flexion
        # Middle (MFJ, MANO 3-5 → Shadow 4-7)
        joints[:, 4] = euler[:, 3, 1]   # MFJ3: abduction
        joints[:, 5] = euler[:, 3, 2]   # MFJ2: MCP flexion
        joints[:, 6] = euler[:, 4, 2]   # MFJ1: PIP flexion
        joints[:, 7] = euler[:, 5, 2]   # MFJ0: DIP flexion
        # Ring   (RFJ, MANO 9-11 → Shadow 8-11)
        joints[:, 8]  = euler[:, 9, 1]   # RFJ3: abduction
        joints[:, 9]  = euler[:, 9, 2]   # RFJ2: MCP flexion
        joints[:, 10] = euler[:, 10, 2]  # RFJ1: PIP flexion
        joints[:, 11] = euler[:, 11, 2]  # RFJ0: DIP flexion
        # Pinky  (LFJ, MANO 6-8 → Shadow 12-16)
        joints[:, 12] = euler[:, 6, 1] * 0.5 + euler[:, 6, 2] * 0.3  # LFJ4: metacarpal
        joints[:, 13] = euler[:, 6, 1]   # LFJ3: abduction
        joints[:, 14] = euler[:, 6, 2]   # LFJ2: MCP flexion
        joints[:, 15] = euler[:, 7, 2]   # LFJ1: PIP flexion
        joints[:, 16] = euler[:, 8, 2]   # LFJ0: DIP flexion
        # Thumb  (THJ, MANO 12-13 → Shadow 17-21)
        joints[:, 18] = euler[:, 12, 1]         # THJ3: abduction
        joints[:, 19] = euler[:, 12, 2]         # THJ2: rotation
        joints[:, 20] = euler[:, 13, 2] * 0.5   # THJ1: rotation
        joints[:, 21] = euler[:, 13, 2] * 0.3   # THJ0: rotation

        return torch.cat([trans, global_aa, joints], dim=-1)  # (B, 28)
