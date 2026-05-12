import torch

def per_class_iou(logits, targets, num_classes=3, smooth=1e-6):
    preds = torch.argmax(logits, dim=1)
    out = {}

    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()

        intersection = (pred_cls * target_cls).sum(dim=(1, 2))
        union = pred_cls.sum(dim=(1, 2)) + target_cls.sum(dim=(1, 2)) - intersection
        iou = (intersection + smooth) / (union + smooth)
        out[cls] = iou.mean().item()

    return out

def per_class_dice(logits, targets, num_classes=3, smooth=1e-6):
    preds = torch.argmax(logits, dim=1)
    out = {}

    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()

        intersection = (pred_cls * target_cls).sum(dim=(1, 2))
        union = pred_cls.sum(dim=(1, 2)) + target_cls.sum(dim=(1, 2))
        dice = (2 * intersection + smooth) / (union + smooth)
        out[cls] = dice.mean().item()

    return out

def mean_fg_iou(logits, targets):
    cls_iou = per_class_iou(logits, targets, num_classes=3)
    return (cls_iou[1] + cls_iou[2]) / 2.0


def mean_fg_dice(logits, targets):
    cls_dice = per_class_dice(logits, targets, num_classes=3)
    return (cls_dice[1] + cls_dice[2]) / 2.0

def dice_score(logits, targets, threshold=0.5, smooth=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)

    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.mean().item()

def iou_score(logits, targets, threshold=0.5, smooth=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    total = preds.sum(dim=1) + targets.sum(dim=1) - intersection

    iou = (intersection + smooth) / (total + smooth)
    return iou.mean().item()


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    n_batches = 0

    per_cls_dice_sum = {0: 0.0, 1: 0.0, 2: 0.0}
    per_cls_iou_sum = {0: 0.0, 1: 0.0, 2: 0.0}

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)   # [B, 1, H, W]
            masks = batch["mask"].to(device)     # [B, H, W]

            logits = model(images)
            loss, _ = criterion(logits, masks)

            total_loss += loss.item()
            total_dice += mean_fg_dice(logits, masks)
            total_iou += mean_fg_iou(logits, masks)

            cls_dice = per_class_dice(logits, masks, num_classes=3)
            cls_iou = per_class_iou(logits, masks, num_classes=3)

            for k in per_cls_dice_sum:
                per_cls_dice_sum[k] += cls_dice[k]
                per_cls_iou_sum[k] += cls_iou[k]

            n_batches += 1

    metrics = {
        "loss": total_loss / n_batches,
        "mean_fg_dice": total_dice / n_batches,
        "mean_fg_iou": total_iou / n_batches,
        "dice_bg": per_cls_dice_sum[0] / n_batches,
        "dice_whole": per_cls_dice_sum[1] / n_batches,
        "dice_damaged": per_cls_dice_sum[2] / n_batches,
        "iou_bg": per_cls_iou_sum[0] / n_batches,
        "iou_whole": per_cls_iou_sum[1] / n_batches,
        "iou_damaged": per_cls_iou_sum[2] / n_batches,
    }
    return metrics