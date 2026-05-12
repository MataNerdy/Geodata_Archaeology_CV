
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiClassDiceLoss(nn.Module):
    def __init__(self, num_classes=3, smooth=1e-6, ignore_background=True):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)  # [B, C, H, W]
        targets_oh = F.one_hot(targets, num_classes=self.num_classes)  # [B, H, W, C]
        targets_oh = targets_oh.permute(0, 3, 1, 2).float()            # [B, C, H, W]

        dims = (0, 2, 3)
        intersection = (probs * targets_oh).sum(dims)
        union = probs.sum(dims) + targets_oh.sum(dims)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        if self.ignore_background:
            dice = dice[1:]

        return 1.0 - dice.mean()


class CombinedMultiClassLoss(nn.Module):
    def __init__(
        self,
        num_classes=3,
        ce_weight=0.5,
        dice_weight=0.5,
        class_weights=None,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else None
        )

        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)
        self.dice_loss = MultiClassDiceLoss(num_classes=num_classes)

    def forward(self, logits, targets):
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        total = self.ce_weight * ce + self.dice_weight * dice
        return total, {
            "ce": ce.item(),
            "dice": dice.item(),
        }
