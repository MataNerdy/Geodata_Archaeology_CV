"""Dice losses for binary and multiclass segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryDiceLoss(nn.Module):
    """Soft Dice loss for binary logits."""

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


class MulticlassDiceLoss(nn.Module):
    """Soft Dice loss for multiclass logits."""

    def __init__(
        self,
        num_classes: int,
        smooth: float = 1e-6,
        include_background: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        intersection = (probs * one_hot).sum(dim=(0, 2, 3))
        denominator = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        if not self.include_background:
            dice = dice[1:]
        return 1.0 - dice.mean()

