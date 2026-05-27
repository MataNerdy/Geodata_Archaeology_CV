"""Visualization helpers for segmentation predictions."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.archaeology_dataset import class_names_for_task
from utils.metrics import logits_to_predictions


COLORS = np.array(
    [
        [0, 0, 0],
        [0, 190, 80],
        [230, 45, 45],
        [0, 180, 220],
        [230, 210, 40],
        [210, 70, 220],
    ],
    dtype=np.float32,
) / 255.0


def save_prediction_grid(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_path: str | Path,
    task: str,
    max_samples: int = 6,
    threshold: float = 0.5,
) -> None:
    """Save Image | GT | Prediction | Overlay grid."""

    model.eval()
    batch = next(iter(loader))
    images = batch["image"].to(device)
    masks = batch["mask"]
    with torch.no_grad():
        preds = logits_to_predictions(model(images), task, threshold=threshold).cpu()

    n = min(max_samples, images.shape[0])
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.4 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)
    for idx in range(n):
        image = stretch(images[idx, 0].detach().cpu().numpy())
        gt = masks[idx].numpy()
        pred = preds[idx].numpy()
        axes[idx, 0].imshow(image, cmap="gray")
        axes[idx, 0].set_title(f"{batch['sample_id'][idx]} | {batch['modality'][idx]}")
        axes[idx, 1].imshow(colorize_mask(gt, task))
        axes[idx, 1].set_title("Ground Truth")
        axes[idx, 2].imshow(colorize_mask(pred, task))
        axes[idx, 2].set_title("Prediction")
        axes[idx, 3].imshow(image, cmap="gray")
        axes[idx, 3].imshow(mask_overlay(pred, task), alpha=0.65)
        axes[idx, 3].set_title(str(batch["region"][idx]))
        for axis in axes[idx]:
            axis.axis("off")
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    matrix: torch.Tensor,
    class_names: dict[int, str],
    save_path: str | Path,
) -> None:
    """Save normalized confusion matrix heatmap."""

    values = matrix.float()
    row_sum = values.sum(dim=1, keepdim=True).clamp_min(1)
    norm = (values / row_sum).numpy()
    labels = [class_names[idx] for idx in sorted(class_names)]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_sweep(rows: list[dict[str, float]], save_path: str | Path) -> None:
    """Plot threshold sweep metrics."""

    thresholds = [row["threshold"] for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(thresholds, [row["fg_iou"] for row in rows], marker="o")
    axes[0].set_title("Threshold vs fg_iou")
    axes[1].plot(thresholds, [row["fg_dice"] for row in rows], marker="o")
    axes[1].set_title("Threshold vs fg_dice")
    axes[2].plot(thresholds, [row["precision"] for row in rows], label="precision", marker="o")
    axes[2].plot(thresholds, [row["recall"] for row in rows], label="recall", marker="o")
    axes[2].legend()
    for axis in axes:
        axis.set_xlabel("threshold")
        axis.grid(alpha=0.3)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def stretch(image: np.ndarray) -> np.ndarray:
    """Robust min-max stretch for display."""

    lo, hi = np.percentile(image, [2, 98])
    if hi - lo < 1e-6:
        return np.zeros_like(image)
    return np.clip((image - lo) / (hi - lo), 0, 1)


def colorize_mask(mask: np.ndarray, task: str) -> np.ndarray:
    """Convert integer mask to RGB image."""

    max_id = max(class_names_for_task(task))
    return COLORS[np.clip(mask.astype(np.int64), 0, max_id)]


def mask_overlay(mask: np.ndarray, task: str) -> np.ndarray:
    """Create RGBA overlay for non-background classes."""

    rgb = colorize_mask(mask, task)
    alpha = (mask > 0).astype(np.float32) * 0.7
    return np.dstack([rgb, alpha])

