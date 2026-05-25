"""Model factory for segmentation experiments."""

from __future__ import annotations

import torch.nn as nn

from models.unet_small import UNetSmall


def build_model(
    name: str = "unet_small",
    in_channels: int = 1,
    num_classes: int = 3,
) -> nn.Module:
    """Create a segmentation model by name."""

    if name == "unet_small":
        return UNetSmall(in_channels=in_channels, out_channels=num_classes)
    raise ValueError(f"Unknown model '{name}'. Available: unet_small")
