"""Binary segmentation losses."""

from __future__ import annotations

import torch
import torch.nn as nn


class DiceLossBinary(nn.Module):
    """Soft Dice loss for binary segmentation logits."""

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(0, 2, 3))
        denominator = probs.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class CombinedBinaryLoss(nn.Module):
    """Weighted sum of BCEWithLogitsLoss and binary Dice loss."""

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        weight = None if pos_weight is None else torch.tensor([pos_weight], dtype=torch.float32)
        self.register_buffer("pos_weight", weight)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.dice = DiceLossBinary()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        total = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return total, {
            "bce_loss": float(bce_loss.detach().cpu()),
            "dice_loss": float(dice_loss.detach().cpu()),
        }
