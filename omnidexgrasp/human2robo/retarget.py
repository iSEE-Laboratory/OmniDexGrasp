"""Two-stage MANO -> robot hand retargeting optimization.

All inputs are in obj_cam frame (= mesh-local frame).
Stage 1: Adam (lr=0.01, iter=40)   -- fingertip alignment only
Stage 2: SGD  (lr=0.001, iter=100) -- fingertip + penetration, freeze translation
Output: dex_pose already in obj_cam frame, no post-transform needed.
"""

from dataclasses import dataclass
import torch
import logging
from omegaconf import DictConfig

from .models.base import RobotHandModel
from .loss import compute_loss, LossWeights


@dataclass
class RetargetResult:
    """📤 Retargeting result for one hand type (pose in obj_cam/mesh-local frame)."""
    hand_type: str
    init_dex_pose_obj: torch.Tensor  # (1, DOF) initial pose (kinematic mapping, before optim)
    dex_pose_obj: torch.Tensor       # (1, DOF) optimized pose in obj_cam frame


def retarget_pose(
    model: RobotHandModel,
    gt_fingertip: torch.Tensor,    # (1, 5, 3) -- obj_cam frame
    obj_pc: torch.Tensor,          # (1, N, 3) -- obj_cam frame (mesh-local)
    mano_trans: torch.Tensor,      # (1, 3)    -- obj_cam frame
    mano_axis_angle: torch.Tensor, # (1, 3)    -- obj_cam frame
    mano_pose: torch.Tensor,       # (1, 45)
    cfg: DictConfig,
    hand_type: str,
) -> RetargetResult:
    """🎯 Two-stage MANO -> robot hand retargeting, output in obj_cam frame."""
    # Initial pose from kinematic mapping (obj_cam frame)
    dex_pose = model.mano2robot_batch(mano_trans, mano_axis_angle, mano_pose)
    init_dex_pose = dex_pose.detach().clone()   # save before optimization
    dex_pose = dex_pose.detach().requires_grad_(True)
    
    # ── Stage 1: Fingertip alignment (Adam, obj_cam frame) ───────────────────
    opt = torch.optim.Adam([dex_pose], lr=cfg.optim.stage1.lr, weight_decay=0)
    w1 = LossWeights(finger=cfg.optim.weights.finger, pen=0.0)

    for t in range(cfg.optim.stage1.iters):
        out = model(dex_pose)
        loss, info = compute_loss(out["fingertip_keypoints"], gt_fingertip, None, w1)
        if t % 10 == 0:
            logging.info(f"  [{hand_type}] S1 iter={t:3d}  finger={info.fingertip:.4f}")
        opt.zero_grad(); loss.backward(); opt.step()

    # ── Stage 2: Fingertip + penetration (SGD, freeze translation) ───────────
    opt = torch.optim.SGD([dex_pose], lr=cfg.optim.stage2.lr, weight_decay=0)
    w2 = LossWeights(finger=cfg.optim.weights.finger, pen=cfg.optim.weights.pen)

    freeze_mask = torch.ones_like(dex_pose)
    freeze_mask[:, :3] = 0.0

    for t in range(cfg.optim.stage2.iters):
        out = model(dex_pose, object_pc=obj_pc, with_penetration=True)
        loss, info = compute_loss(out["fingertip_keypoints"], gt_fingertip,
                                  out.get("penetration"), w2)
        if t % 20 == 0:
            logging.info(f"  [{hand_type}] S2 iter={t:3d}  loss={info.total:.4f} "
                         f"finger={info.fingertip:.4f} pen={info.penetration:.4f}")
        opt.zero_grad(); loss.backward()
        dex_pose.grad.data *= freeze_mask
        opt.step()

    logging.info(f"  [{hand_type}] ✅ final loss={info.total:.4f} "
                 f"finger={info.fingertip:.4f} pen={info.penetration:.4f}")

    return RetargetResult(hand_type=hand_type,
                          init_dex_pose_obj=init_dex_pose,
                          dex_pose_obj=dex_pose.detach())
