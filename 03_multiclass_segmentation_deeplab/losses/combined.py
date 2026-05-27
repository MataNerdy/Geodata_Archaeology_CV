"""Combined segmentation losses."""

from __future__ import annotations

import torch
import torch.nn as nn

from losses.dice import BinaryDiceLoss, MulticlassDiceLoss


class CombinedBinaryLoss(nn.Module):
    """BCEWithLogitsLoss plus optional binary Dice."""

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
        self.dice = BinaryDiceLoss()

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


class CombinedMulticlassLoss(nn.Module):
    """CrossEntropyLoss plus optional multiclass Dice."""

    def __init__(
        self,
        num_classes: int,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        weights = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32)
        self.register_buffer("class_weights", weights)
        self.ce = nn.CrossEntropyLoss(weight=self.class_weights)
        self.dice = MulticlassDiceLoss(num_classes=num_classes)

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

