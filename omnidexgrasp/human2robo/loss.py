"""Fingertip alignment loss + penetration loss for H2R retargeting."""

from dataclasses import dataclass
import torch


@dataclass
class LossWeights:
    """⚖️ Loss weights for two-stage optimization."""
    finger: float = 1.0
    pen: float = 1.0


@dataclass
class LossResult:
    """📊 Per-component loss values for logging."""
    total: float
    fingertip: float
    penetration: float


def compute_loss(
    pred_fingertip: torch.Tensor,      # (B, 5, 3)
    gt_fingertip: torch.Tensor,        # (B, 5, 3)
    penetration: torch.Tensor | None,  # (B, N_pts) negative = penetrating
    weights: LossWeights,
) -> tuple[torch.Tensor, LossResult]:
    """🎯 Compute weighted fingertip + penetration loss."""
    # Fingertip Loss: per-finger weighted L2 (thumb weight = 2)
    finger_w = torch.tensor([2., 1., 1., 1., 1.], device=pred_fingertip.device)
    diff = (pred_fingertip - gt_fingertip) * 100.0  # scale to cm-range
    l2 = torch.norm(diff, dim=2)                    # (B, 5)
    loss_finger = (l2 * finger_w).sum(dim=1).mean()

    # Penetration Loss: sum of interpenetration depths
    loss_pen = pred_fingertip.new_tensor(0.0)
    if penetration is not None and weights.pen > 0:
        pen = penetration.clone()
        pen[pen < 0] = 0.0    # zero outside points (negative SDF), keep inside (positive = penetrating)
        loss_pen = pen.sum()

    total = weights.finger * loss_finger + weights.pen * loss_pen
    return total, LossResult(
        total=total.item(),
        fingertip=loss_finger.item(),
        penetration=loss_pen.item(),
    )
