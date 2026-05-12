import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: [B, 1, H, W]
        targets: [B, 1, H, W], float in {0, 1}
        """
        targets = targets.float()

        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)

        focal_weight = self.alpha * (1 - pt).pow(self.gamma)
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1).float()

        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class CombinedSegLoss(nn.Module):
    def __init__(
        self,
        dice_weight=0.50,
        bce_weight=0.25,
        focal_weight=0.25,
        focal_alpha=0.75,
        focal_gamma=2.0,
    ):
        super().__init__()

        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            reduction="mean"
        )

    def forward(self, logits, targets):
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        focal = self.focal_loss(logits, targets)

        total = (
            self.dice_weight * dice +
            self.bce_weight * bce +
            self.focal_weight * focal
        )

        return total, {
            "bce": bce.item(),
            "dice": dice.item(),
            "focal": focal.item(),
        }
