"""Loss package for DeepLab archaeology segmentation."""

from losses.combined import CombinedBinaryLoss, CombinedMulticlassLoss
from losses.dice import BinaryDiceLoss, MulticlassDiceLoss

__all__ = [
    "BinaryDiceLoss",
    "CombinedBinaryLoss",
    "CombinedMulticlassLoss",
    "MulticlassDiceLoss",
]

