import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import random
from tqdm import tqdm


DATA_PATH = "/Users/Di/Documents/Новая папка/Geodata/the_most_important_things/baseline_kurgan/dataset_multi_full_non_binary_сrop/metadata.csv"
IMAGES_DIR = "/Users/Di/Documents/Новая папка/Geodata/the_most_important_things/baseline_kurgan/dataset_multi_full_non_binary_сrop/images"
MASKS_DIR = "/Users/Di/Documents/Новая папка/Geodata/the_most_important_things/baseline_kurgan/dataset_multi_full_non_binary_сrop/masks"

meta = pd.read_csv(DATA_PATH)

print(meta["modality"].value_counts())
meta

meta.groupby("modality")["crop_size"].describe()

print(meta["touches_border"].value_counts())
print(meta["modality"].value_counts())
print(meta["kurgan_type"].value_counts())
print(meta["region"].value_counts())

def filter_kurgan_metadata(
    meta: pd.DataFrame,
    allowed_modalities=("Li", "Ae", "SpOr"),
    max_crop_size=2048,
    max_objects_in_patch=40,
    touches_border=False,
):
    meta = meta.copy()

    meta = meta[meta["mask_whole_pixels"] + meta["mask_damaged_pixels"] > 0].copy()

    if "modality" in meta.columns and allowed_modalities is not None:
        meta = meta[meta["modality"].isin(allowed_modalities)].copy()

    if "crop_size" in meta.columns:
        meta = meta[meta["crop_size"] <= max_crop_size].copy()

    if "n_objects_in_patch" in meta.columns:
        meta = meta[meta["n_objects_in_patch"] <= max_objects_in_patch].copy()

    if "touches_border" in meta.columns:
        meta = meta[meta["touches_border"] == touches_border].copy()

    meta = meta.reset_index(drop=True)
    return meta

df_start = filter_kurgan_metadata(
                meta,
                allowed_modalities=("Li", "Ae", "SpOr"),
                max_crop_size=2048,
                max_objects_in_patch=40,
                touches_border=False,
            )
print(df_start["kurgan_type"].value_counts())
df_start

print(df_start["modality"].value_counts())
print(df_start["kurgan_type"].value_counts())
print(df_start["region"].value_counts())

df_start.groupby("modality")["crop_size"].describe()

df_tim = df_start[df_start["region"] == "027_ТИМЕРЕВО"]
df_tim.groupby("modality")["crop_size"].describe()

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


dataset = KurganDatasetBinary(
    meta_df=df_start,
    images_dir=IMAGES_DIR,
    masks_dir=MASKS_DIR,
    target_size=256,
    normalize="zscore",
)

print("Dataset size:", len(dataset))

sample = dataset[4]
print(sample["image"].shape)   # [1, 256, 256]
print(sample["mask"].shape)    # [1, 256, 256]
print(torch.unique(sample["mask"]))  # должно быть tensor([0., 1.])

loader = DataLoader(dataset, batch_size=16, shuffle=True)

batch = next(iter(loader))
print("batch image shape:", batch["image"].shape)   # [B, 1, 256, 256]
print("batch mask shape:", batch["mask"].shape)     # [B, 1, 256, 256]
print("mask unique values:", torch.unique(batch["mask"]))

# visualize one sample
img = batch["image"][0, 0].numpy()
msk = batch["mask"][0, 0].numpy()

overlay = np.zeros((*msk.shape, 4), dtype=np.float32)
overlay[msk == 1] = [1.0, 0.0, 0.0, 0.35]

plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.title("Image")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Mask")
plt.imshow(msk, vmin=0, vmax=1)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Overlay")
plt.imshow(img, cmap="gray")
plt.imshow(overlay)
plt.axis("off")

plt.tight_layout()
plt.show()


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

meta = df_start

print(meta["kurgan_type"].value_counts())
print(meta["region"].value_counts())

val_regions = [
    "027_ТИМЕРЕВО",
]

train_df = meta[~meta["region"].isin(val_regions)].copy()
val_df = meta[meta["region"].isin(val_regions)].copy()

print(len(train_df), len(val_df))
print(train_df["kurgan_type"].value_counts())
print(val_df["kurgan_type"].value_counts())
print(val_df["region"].value_counts())

print("Train ", train_df["modality"].value_counts())
print(pd.crosstab(train_df["modality"], train_df["kurgan_type"]))
print("Val ", val_df["modality"].value_counts())
print(pd.crosstab(val_df["modality"], val_df["kurgan_type"]))

print("Train ", train_df["has_whole"].value_counts(dropna=False))
print("Train ", train_df["has_damaged"].value_counts(dropna=False))
print(((train_df["has_whole"]) & (train_df["has_damaged"])).sum())
print("Val ", val_df["has_whole"].value_counts(dropna=False))
print("Val ", val_df["has_damaged"].value_counts(dropna=False))
print(((val_df["has_whole"]) & (val_df["has_damaged"])).sum())

set_seed(42)

val_regions = ["027_ТИМЕРЕВO"]

train_df = meta[~meta["region"].isin(val_regions)].copy()
val_df = meta[meta["region"].isin(val_regions)].copy()

print("Train:", len(train_df))
print(train_df["modality"].value_counts())

print("Val:", len(val_df))
print(val_df["modality"].value_counts())
print(val_df["region"].value_counts())

train_dataset = KurganDatasetBinary(
    meta_df=train_df,
    images_dir=IMAGES_DIR,
    masks_dir=MASKS_DIR,
    target_size=256,
    normalize="zscore",
)

val_dataset = KurganDatasetBinary(
    meta_df=val_df,
    images_dir=IMAGES_DIR,
    masks_dir=MASKS_DIR,
    target_size=256,
    normalize="zscore",
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = UNetSmall(in_channels=1, out_channels=1).to(device)

criterion = CombinedBinaryLoss(
    bce_weight=0.7,
    dice_weight=0.3,
    pos_weight=2.0
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=5,
)

n_epochs = 80
patience = 12
epochs_no_improve = 0

best_val_iou = -1.0
best_epoch = -1
ckpt_path = "unet_kurgan_binary_timerevo_holdout_best.pth"

history = []

for epoch in range(1, n_epochs + 1):
    model.train()
    train_loss = 0.0
    train_bce = 0.0
    train_dice_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}")
    for batch in pbar:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss, loss_dict = criterion(logits, masks)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        train_bce += loss_dict["bce"]
        train_dice_loss += loss_dict["dice"]

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            bce=f"{loss_dict['bce']:.4f}",
            dice=f"{loss_dict['dice']:.4f}",
        )

    train_loss /= len(train_loader)
    train_bce /= len(train_loader)
    train_dice_loss /= len(train_loader)

    val_metrics = evaluate_binary(model, val_loader, criterion, device)

    print(
        f"Epoch {epoch:02d} | "
        f"train_loss={train_loss:.4f} | "
        f"train_bce={train_bce:.4f} | "
        f"train_dice_loss={train_dice_loss:.4f} | "
        f"val_loss={val_metrics['loss']:.4f} | "
        f"val_dice={val_metrics['dice']:.4f} | "
        f"val_iou={val_metrics['iou']:.4f}"
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_bce": train_bce,
        "train_dice_loss": train_dice_loss,
        **val_metrics,
    })

    if val_metrics["iou"] > best_val_iou:
        best_val_iou = val_metrics["iou"]
        best_epoch = epoch
        epochs_no_improve = 0
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved best model to {ckpt_path}")
    else:
        epochs_no_improve += 1

    scheduler.step(val_metrics["iou"])
    current_lr = optimizer.param_groups[0]["lr"]
    print(f"           lr={current_lr:.6f}")

    if epochs_no_improve >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

history_df = pd.DataFrame(history)
history_df.to_csv("train_history_kurgan_binary_timerevo_holdout.csv", index=False)

print("Best epoch:", best_epoch)
print("Best val IoU:", best_val_iou)

# %% [code] {"execution":{"iopub.status.busy":"2026-04-17T22:16:01.950813Z","iopub.execute_input":"2026-04-17T22:16:01.951544Z","iopub.status.idle":"2026-04-17T22:16:01.963528Z","shell.execute_reply.started":"2026-04-17T22:16:01.951509Z","shell.execute_reply":"2026-04-17T22:16:01.962599Z"}}
history_df

# %% [code] {"execution":{"iopub.status.busy":"2026-04-17T22:16:09.848677Z","iopub.execute_input":"2026-04-17T22:16:09.849067Z","iopub.status.idle":"2026-04-17T22:17:07.859802Z","shell.execute_reply.started":"2026-04-17T22:16:09.849039Z","shell.execute_reply":"2026-04-17T22:17:07.858525Z"},"jupyter":{"outputs_hidden":false}}
def denormalize_for_display(img):
    img = img.copy()
    img_min = img.min()
    img_max = img.max()
    if img_max - img_min < 1e-6:
        return img * 0.0
    return (img - img_min) / (img_max - img_min)


def mask_to_overlay_binary(mask):
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[mask == 1] = [1.0, 0.0, 0.0, 0.35]  # курган = красный
    return overlay


def visualize_batch_binary(model, loader, device, max_samples=8, threshold=0.5, save_path=None):
    model.eval()

    batch = next(iter(loader))
    images = batch["image"].to(device)
    masks = batch["mask"].to(device)

    with torch.no_grad():
        logits = model(images)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

    images = images.cpu()
    masks = masks.cpu()
    probs = probs.cpu()
    preds = preds.cpu()

    sample_ids = batch["sample_id"]
    modalities = batch["modality"]

    n = min(max_samples, images.shape[0])
    fig, axes = plt.subplots(n, 5, figsize=(20, 4 * n))

    if n == 1:
        axes = [axes]

    for i in range(n):
        img = images[i, 0].numpy()
        gt = masks[i, 0].numpy()
        pr = preds[i, 0].numpy()
        prob = probs[i, 0].numpy()

        img_show = denormalize_for_display(img)

        # IoU per sample
        intersection = (gt * pr).sum()
        union = gt.sum() + pr.sum() - intersection
        iou = intersection / (union + 1e-6)

        axes[i][0].imshow(img_show, cmap="gray")
        axes[i][0].set_title(f"Image\n{modalities[i]} | id={sample_ids[i]}")
        axes[i][0].axis("off")

        axes[i][1].imshow(gt, vmin=0, vmax=1, cmap="gray")
        axes[i][1].set_title("GT")
        axes[i][1].axis("off")

        axes[i][2].imshow(prob, cmap="viridis")
        axes[i][2].set_title("Prob")
        axes[i][2].axis("off")

        axes[i][3].imshow(pr, vmin=0, vmax=1, cmap="gray")
        axes[i][3].set_title(f"Pred (thr={threshold})")
        axes[i][3].axis("off")

        overlay = mask_to_overlay_binary(pr)

        axes[i][4].imshow(img_show, cmap="gray")
        axes[i][4].imshow(overlay)
        axes[i][4].set_title(f"Overlay\nIoU={iou:.3f}")
        axes[i][4].axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")

    plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

val_dataset = KurganDatasetBinary(
    meta_df=val_df,
    images_dir=IMAGES_DIR,
    masks_dir=MASKS_DIR,
    target_size=256,
    normalize="zscore",
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0,
)

model = UNetSmall(in_channels=1, out_channels=1).to(device)
model.load_state_dict(torch.load("/kaggle/working/unet_kurgan_binary_timerevo_holdout_best.pth", map_location=device))

for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    visualize_batch_binary(
            model=model,
            loader=val_loader,
            device=device,
            max_samples=8,
            threshold=t,
            save_path=f"val_predictions_binary_{t}.png",
        )


def binary_dice_iou_per_sample(logits, targets, threshold=0.5, smooth=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    pred_sum = preds.sum(dim=1)
    target_sum = targets.sum(dim=1)
    union = pred_sum + target_sum
    total = pred_sum + target_sum - intersection

    dice = (2 * intersection + smooth) / (union + smooth)
    iou = (intersection + smooth) / (total + smooth)

    return dice, iou


def threshold_sweep_binary(
    model,
    loader,
    device,
    thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
    save_csv_path=None,
):
    model.eval()

    rows = []

    for thr in thresholds:
        total_dice = 0.0
        total_iou = 0.0
        total_samples = 0

        modality_dice = {}
        modality_iou = {}
        modality_count = {}

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                modalities = batch["modality"]

                logits = model(images)

                dice_vec, iou_vec = binary_dice_iou_per_sample(
                    logits, masks, threshold=thr
                )

                bs = images.size(0)

                total_dice += dice_vec.sum().item()
                total_iou += iou_vec.sum().item()
                total_samples += bs

                for i in range(bs):
                    mod = modalities[i]

                    if mod not in modality_dice:
                        modality_dice[mod] = 0.0
                        modality_iou[mod] = 0.0
                        modality_count[mod] = 0

                    modality_dice[mod] += dice_vec[i].item()
                    modality_iou[mod] += iou_vec[i].item()
                    modality_count[mod] += 1

        row = {
            "threshold": thr,
            "dice": total_dice / total_samples if total_samples > 0 else np.nan,
            "iou": total_iou / total_samples if total_samples > 0 else np.nan,
            "n_samples": total_samples,
        }

        for mod in ["Li", "Ae", "SpOr"]:
            if modality_count.get(mod, 0) > 0:
                row[f"{mod}_dice"] = modality_dice[mod] / modality_count[mod]
                row[f"{mod}_iou"] = modality_iou[mod] / modality_count[mod]
                row[f"{mod}_n"] = modality_count[mod]
            else:
                row[f"{mod}_dice"] = np.nan
                row[f"{mod}_iou"] = np.nan
                row[f"{mod}_n"] = 0

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)

    if save_csv_path is not None:
        df.to_csv(save_csv_path, index=False)
        print(f"Saved threshold sweep to {save_csv_path}")

    return df

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNetSmall(in_channels=1, out_channels=1).to(device)
model.load_state_dict(
    torch.load(
        "/kaggle/working/unet_kurgan_binary_timerevo_holdout_best.pth",
        map_location=device
    )
)

sweep_df = threshold_sweep_binary(
    model=model,
    loader=val_loader,
    device=device,
    thresholds=np.arange(0.1, 0.8, 0.1),
    save_csv_path="threshold_sweep_binary_timerevo.csv",
)

print(sweep_df)
print()
print("Best by IoU:")
print(sweep_df.loc[sweep_df["iou"].idxmax()])
print()
print("Best by Dice:")
print(sweep_df.loc[sweep_df["dice"].idxmax()])

plt.figure(figsize=(8, 5))
plt.plot(sweep_df["threshold"], sweep_df["dice"], marker="o", label="Dice")
plt.plot(sweep_df["threshold"], sweep_df["iou"], marker="o", label="IoU")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Binary threshold sweep on val")
plt.grid(True)
plt.legend()
plt.show()


def ensemble_threshold_sweep_binary(
    model_pw1,
    model_pw2,
    model_pw3,
    loader,
    device,
    thresholds=(0.1, 0.2, 0.3, 0.4, 0.5),
    weights=(1/3, 1/3, 1/3),
):
    model_pw1.eval()
    model_pw2.eval()
    model_pw3.eval()

    rows = []
    w1, w2, w3 = weights

    for thr in thresholds:
        total_iou = 0.0
        total_dice = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)

                prob1 = torch.sigmoid(model_pw1(images))
                prob2 = torch.sigmoid(model_pw2(images))
                prob3 = torch.sigmoid(model_pw3(images))

                prob = w1 * prob1 + w2 * prob2 + w3 * prob3
                pred = (prob > thr).float()

                pred = pred.view(pred.size(0), -1)
                target = masks.view(masks.size(0), -1)

                intersection = (pred * target).sum(dim=1)
                pred_sum = pred.sum(dim=1)
                target_sum = target.sum(dim=1)

                dice = (2 * intersection + 1e-6) / (pred_sum + target_sum + 1e-6)
                iou = (intersection + 1e-6) / (pred_sum + target_sum - intersection + 1e-6)

                total_dice += dice.sum().item()
                total_iou += iou.sum().item()
                total_samples += images.size(0)

        rows.append({
            "threshold": thr,
            "dice": total_dice / total_samples,
            "iou": total_iou / total_samples,
            "w1": w1,
            "w2": w2,
            "w3": w3,
        })

    return pd.DataFrame(rows)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_pw1 = UNetSmall().to(device)
model_pw1.load_state_dict(torch.load("/Users/Di/Documents/Новая папка/Geodata/the_most_important_things/baseline_kurgan/unet_kurgan_binary_pos_1_best.pth", map_location=device))

model_pw2 = UNetSmall().to(device)
model_pw2.load_state_dict(torch.load("/Users/Di/Documents/Новая папка/Geodata/the_most_important_things/baseline_kurgan/unet_kurgan_binary_pos_2_best.pth", map_location=device))

model_pw3 = UNetSmall().to(device)
model_pw3.load_state_dict(torch.load("/Users/Di/Documents/Новая папка/Geodata/the_most_important_things/baseline_kurgan/unet_kurgan_binary_pos_3_best.pth", map_location=device))

df_ens_mean = ensemble_threshold_sweep_binary(
    model_pw1, model_pw2, model_pw3,
    loader=val_loader,
    device=device,
    thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    weights=(1/3, 1/3, 1/3),
)

df_ens_pw2 = ensemble_threshold_sweep_binary(
    model_pw1, model_pw2, model_pw3,
    loader=val_loader,
    device=device,
    thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    weights=(0.25, 0.5, 0.25),
)

df_ens_12 = ensemble_threshold_sweep_binary(
    model_pw1, model_pw2, model_pw3,
    loader=val_loader,
    device=device,
    thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    weights=(0.3, 0.7, 0.0),
)

print(df_ens_mean)
print(df_ens_pw2)
print(df_ens_12)