import numpy as np # linear algebra# data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import random
import segmentation_models_pytorch as smp

DATA_PATH = "/Users/Di/Documents/Новая папка/Geodata/the_most_important_things/baseline_kurgan/dataset_multi_full_non_binary_сrop/metadata.csv"
IMAGES_DIR = "/Users/Di/Documents/Новая папка/Geodata/the_most_important_things/baseline_kurgan/dataset_multi_full_non_binary_сrop/images"
MASKS_DIR = "/Users/Di/Documents/Новая папка/Geodata/the_most_important_things/baseline_kurgan/dataset_multi_full_non_binary_сrop/masks"

class DeepLabBinary(nn.Module):
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=1,
        classes=1,
    ):
        super().__init__()

        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )

    def forward(self, x):
        return self.model(x)

class KurganDatasetBinary(Dataset):
    def __init__(
        self,
        meta_df,
        images_dir,
        masks_dir,
        target_size=256,
        normalize="zscore",
        transform=None,
    ):
        self.meta = meta_df.reset_index(drop=True).copy()
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.target_size = target_size
        self.normalize = normalize
        self.transform = transform

    def __len__(self):
        return len(self.meta)

    def _normalize_patch(self, patch: np.ndarray) -> np.ndarray:
        patch = patch.astype(np.float32)

        if self.normalize == "zscore":
            mean = patch.mean()
            std = patch.std()
            if std < 1e-6:
                std = 1.0
            patch = (patch - mean) / std

        elif self.normalize == "minmax":
            pmin = patch.min()
            pmax = patch.max()
            if pmax - pmin < 1e-6:
                patch = np.zeros_like(patch, dtype=np.float32)
            else:
                patch = (patch - pmin) / (pmax - pmin)

        elif self.normalize is None:
            pass
        else:
            raise ValueError(f"Unknown normalize mode: {self.normalize}")

        return patch

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        sample_id = f"{int(row['sample_id']):06d}"

        patch = np.load(self.images_dir / f"{sample_id}.npy")
        mask = np.load(self.masks_dir / f"{sample_id}.npy")

        mask = (mask > 0).astype(np.uint8)

        orig_h, orig_w = patch.shape

        if self.target_size is not None:
            if patch.shape != (self.target_size, self.target_size):
                patch = cv2.resize(
                    patch,
                    (self.target_size, self.target_size),
                    interpolation=cv2.INTER_LINEAR
                )

            if mask.shape != (self.target_size, self.target_size):
                mask = cv2.resize(
                    mask,
                    (self.target_size, self.target_size),
                    interpolation=cv2.INTER_NEAREST
                )

        mask = (mask > 0).astype(np.uint8)

        if self.transform is not None:
            patch, mask = self.transform(patch, mask)

        patch = np.asarray(patch, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)

        unique_vals = np.unique(mask)
        if not np.all(np.isin(unique_vals, [0, 1])):
            raise ValueError(f"Unexpected binary mask values for sample {sample_id}: {unique_vals}")

        patch = self._normalize_patch(patch)

        patch = torch.from_numpy(patch).float().unsqueeze(0)
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        sample = {
            "image": patch,
            "mask": mask,
            "sample_id": sample_id,
            "region": row["region"],
            "modality": row["modality"],
            "raster_file": row["raster_file"],
            "kurgan_type": row["kurgan_type"],
            "n_objects_in_patch": int(row["n_objects_in_patch"]),
            "orig_height": int(orig_h),
            "orig_width": int(orig_w),
            "input_height": int(patch.shape[-2]),
            "input_width": int(patch.shape[-1]),
        }

        optional_cols = [
            "used_crs_fallback",
            "mask_bg_pixels",
            "mask_whole_pixels",
            "mask_damaged_pixels",
            "has_whole",
            "has_damaged",
            "touches_border",
            "crop_size",
        ]
        for col in optional_cols:
            if col in row.index:
                sample[col] = row[col]

        return sample

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetSmall(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class CombinedBinaryLoss(nn.Module):
    def __init__(self, bce_weight=0.7, dice_weight=0.3, pos_weight=None):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        if pos_weight is not None:
            pos_weight = torch.tensor([pos_weight], dtype=torch.float32)
        else:
            pos_weight = None

        self.register_buffer("pos_weight", pos_weight)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.dice = BinaryDiceLoss()

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        dice = self.dice(logits, targets)

        total = self.bce_weight * bce + self.dice_weight * dice
        return total, {
            "bce": float(bce.item()),
            "dice": float(dice.item()),
        }


def binary_dice_score(logits, targets, threshold=0.5, smooth=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)

    dice = (2 * intersection + smooth) / (union + smooth)
    return dice  # vector [B]


def binary_iou_score(logits, targets, threshold=0.5, smooth=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    total = preds.sum(dim=1) + targets.sum(dim=1) - intersection

    iou = (intersection + smooth) / (total + smooth)
    return iou  # vector [B]

def evaluate_binary(model, loader, criterion, device, threshold=0.5):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            loss, _ = criterion(logits, masks)

            batch_size = images.size(0)
            dice_batch = binary_dice_score(logits, masks, threshold=threshold)
            iou_batch = binary_iou_score(logits, masks, threshold=threshold)

            total_loss += loss.item() * batch_size
            total_dice += dice_batch.sum().item()
            total_iou += iou_batch.sum().item()
            total_samples += batch_size

    if total_samples == 0:
        return {
            "loss": float("nan"),
            "dice": float("nan"),
            "iou": float("nan"),
        }

    return {
        "loss": total_loss / total_samples,
        "dice": total_dice / total_samples,
        "iou": total_iou / total_samples,
    }

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)