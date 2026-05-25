"""Loss functions for multiclass kurgan segmentation."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

__path__ = [str(Path(__file__).with_name("losses"))]


class DiceLoss(nn.Module):
    """Soft multiclass Dice loss with optional background exclusion."""

    def __init__(
        self,
        num_classes: int = 3,
        smooth: float = 1e-6,
        include_background: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = (probs * targets_one_hot).sum(dim=dims)
        denominator = probs.sum(dim=dims) + targets_one_hot.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)

        if not self.include_background:
            dice = dice[1:]
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """Weighted sum of CrossEntropyLoss and DiceLoss."""

    def __init__(
        self,
        num_classes: int = 3,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        weights = None
        if class_weights is not None:
            weights = torch.tensor(class_weights, dtype=torch.float32)
        self.register_buffer("class_weights", weights)
        self.ce = nn.CrossEntropyLoss(weight=self.class_weights)
        self.dice = DiceLoss(num_classes=num_classes)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        total = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        return total, {
            "ce_loss": float(ce_loss.detach().cpu()),
            "dice_loss": float(dice_loss.detach().cpu()),
        }
