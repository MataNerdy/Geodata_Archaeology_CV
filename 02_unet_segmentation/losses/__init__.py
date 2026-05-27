"""Loss functions for UNet segmentation experiments."""

from losses.binary_losses import CombinedBinaryLoss, DiceLossBinary
from losses.multiclass_losses import CombinedLoss, DiceLoss

__all__ = [
    "CombinedBinaryLoss",
    "CombinedLoss",
    "DiceLoss",
    "DiceLossBinary",
]
