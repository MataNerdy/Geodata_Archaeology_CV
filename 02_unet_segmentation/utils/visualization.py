import torch
import matplotlib.pyplot as plt
import numpy as np
def denormalize_for_display(img):
    """
    img: numpy array after zscore normalization
    just stretch to [0, 1] for visualization
    """
    img = img.copy()
    img_min = img.min()
    img_max = img.max()
    if img_max - img_min < 1e-6:
        return img * 0.0
    return (img - img_min) / (img_max - img_min)

def mask_to_overlay(mask):
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)

    # whole = зелёный
    overlay[mask == 1] = [0.0, 1.0, 0.0, 0.35]

    # damaged = красный
    overlay[mask == 2] = [1.0, 0.0, 0.0, 0.35]

    return overlay

def visualize_batch(model, loader, device, max_samples=8, save_path=None):
    model.eval()

    batch = next(iter(loader))
    images = batch["image"].to(device)
    masks = batch["mask"].to(device)

    with torch.no_grad():
        logits = model(images)
        preds = torch.argmax(logits, dim=1)  # [B, H, W]

    images = images.cpu()
    masks = masks.cpu()
    preds = preds.cpu()

    sample_ids = batch["sample_id"]
    regions = batch["region"]

    n = min(max_samples, images.shape[0])
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))

    if n == 1:
        axes = [axes]

    for i in range(n):
        img = images[i, 0].numpy()
        gt = masks[i].numpy()
        pr = preds[i].numpy()

        img_show = denormalize_for_display(img)

        pr_overlay = mask_to_overlay(pr)

        axes[i][0].imshow(img_show, cmap="gray")
        axes[i][0].set_title(f"Image\nid={sample_ids[i]}")
        axes[i][0].axis("off")

        axes[i][1].imshow(gt, vmin=0, vmax=2)
        axes[i][1].set_title("GT mask")
        axes[i][1].axis("off")

        axes[i][2].imshow(pr, vmin=0, vmax=2)
        axes[i][2].set_title("Pred mask")
        axes[i][2].axis("off")

        axes[i][3].imshow(img_show, cmap="gray")
        axes[i][3].imshow(pr_overlay)
        axes[i][3].set_title(f"Pred overlay\nregion={regions[i]}")
        axes[i][3].axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")

    plt.show()