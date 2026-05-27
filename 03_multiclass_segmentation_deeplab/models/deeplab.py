"""DeepLab model factory."""

from __future__ import annotations

import torch
import torch.nn as nn


class DeepLabSegmentationModel(nn.Module):
    """Thin wrapper around segmentation_models_pytorch DeepLabV3+."""

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = None,
        in_channels: int = 1,
        classes: int = 6,
    ) -> None:
        super().__init__()
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise ImportError(
                "segmentation_models_pytorch is required for DeepLab experiments. "
                "Install requirements.txt before training."
            ) from exc

        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return segmentation logits."""

        return self.model(images)


def build_model(
    encoder_name: str = "resnet34",
    encoder_weights: str | None = None,
    in_channels: int = 1,
    classes: int = 6,
) -> DeepLabSegmentationModel:
    """Build a DeepLabV3+ model."""

    return DeepLabSegmentationModel(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )

