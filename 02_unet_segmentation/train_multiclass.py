import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.kurgan_dataset import KurganDataset
from models.unet_small import UNetSmall
from utils.metrics import evaluate
from losses.multiclass_losses import CombinedMultiClassLoss


IMAGES_DIR = "datasets/kurgans_dataset/images"
MASKS_DIR = "datasets/kurgans_dataset/masks"
DATA_PATH = pd.read_csv("datasets/kurgans_dataset/metadata.csv")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

meta = pd.read_csv(DATA_PATH)

val_regions = ["005_ЛУБНО",
               "019_ОСЕЧКИ_1",
               "025_ШУМГОРА",
               "030_КОПАНСКОЕ",
               "037_КЧР",
               "056_ПЕРЫНЬ",
               "057_ШИШКИНО",
               "072_Каменка",
               "075_Сары_Булун",
               "076_Скупая_Полудань",
               "078_Архангельское",
               "081_Тоссор",
               "082_Солдатское",
               "083_Мостище",
               "088_Верхний_Карабут",
               "090_Артпозиции_Иссык_Куль",
               "118_Иссык_Куль_курганы_и_постройки_1",
               "120_Курганы_7",
               "150_Постройки_1"]

train_df = meta[~meta["region"].isin(val_regions)].copy()
val_df = meta[meta["region"].isin(val_regions)].copy()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

train_dataset = KurganDataset(
        meta_df=train_df,
        images_dir=IMAGES_DIR,
        masks_dir=MASKS_DIR,
        target_size=256,
        normalize="zscore",
        transform=None,
    )

val_dataset = KurganDataset(
        meta_df=val_df,
        images_dir=IMAGES_DIR,
        masks_dir=MASKS_DIR,
        target_size=256,
        normalize="zscore",
        transform=None,
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

model = UNetSmall(in_channels=1, out_channels=3).to(device)

criterion = CombinedMultiClassLoss(
    num_classes=3,
    ce_weight=1,
    dice_weight=0,
    class_weights=[0.2, 1.0, 3.0],
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = 3e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=5
)

n_epochs = 80
patience = 12
epochs_no_improve = 0

best_val_iou = -1.0
best_epoch = -1
ckpt_path = "/kaggle/working/unet_multiclass_best.pth"

history = []

for epoch in range(1, n_epochs + 1):
    model.train()
    train_loss = 0.0
    train_ce = 0.0
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
        train_ce += loss_dict["ce"]
        train_dice_loss += loss_dict["dice"]

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            ce=f"{loss_dict['ce']:.4f}",
            dice=f"{loss_dict['dice']:.4f}",
        )

    train_loss /= len(train_loader)
    train_ce /= len(train_loader)
    train_dice_loss /= len(train_loader)

    val_metrics = evaluate(model, val_loader, criterion, device)

    print(
        f"Epoch {epoch:02d} | "
        f"train_loss={train_loss:.4f} | "
        f"train_ce={train_ce:.4f} | "
        f"train_dice_loss={train_dice_loss:.4f} | "
        f"val_loss={val_metrics['loss']:.4f} | "
        f"val_fg_dice={val_metrics['mean_fg_dice']:.4f} | "
        f"val_fg_iou={val_metrics['mean_fg_iou']:.4f}"
    )
    print(
        f"           "
        f"val_dice_bg={val_metrics['dice_bg']:.4f} | "
        f"val_dice_whole={val_metrics['dice_whole']:.4f} | "
        f"val_dice_damaged={val_metrics['dice_damaged']:.4f}"
    )
    print(
        f"           "
        f"val_iou_bg={val_metrics['iou_bg']:.4f} | "
        f"val_iou_whole={val_metrics['iou_whole']:.4f} | "
        f"val_iou_damaged={val_metrics['iou_damaged']:.4f}"
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_ce": train_ce,
        "train_dice_loss": train_dice_loss,
        **val_metrics,
    })

    if val_metrics["mean_fg_iou"] > best_val_iou:
        best_val_iou = val_metrics["mean_fg_iou"]
        best_epoch = epoch
        epochs_no_improve = 0
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved best model to {ckpt_path}")
    else:
        epochs_no_improve += 1

    scheduler.step(val_metrics["mean_fg_iou"])
    current_lr = optimizer.param_groups[0]["lr"]
    print(f"           lr={current_lr:.6f}")

    if epochs_no_improve >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

history_df = pd.DataFrame(history)
history_df.to_csv("/kaggle/working/train_history_multiclass.csv", index=False)

print("Best epoch:", best_epoch)
print("Best val mean FG IoU:", best_val_iou)
