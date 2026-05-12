import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset_loader import KurganDataset
import pandas as pd

meta = pd.read_csv("dataset_multi/metadata.csv")

dataset = KurganDataset(
    meta_df=meta,
    images_dir="dataset_multi/images",
    masks_dir="dataset_multi/masks",
    target_size=256,
    normalize="zscore",
)

print("Dataset size:", len(dataset))

sample = dataset[0]
print("image shape:", sample["image"].shape)
print("mask shape:", sample["mask"].shape)
print("sample_id:", sample["sample_id"])
print("region:", sample["region"])

loader = DataLoader(dataset, batch_size=8, shuffle=True)

batch = next(iter(loader))
print("batch image shape:", batch["image"].shape)   # [B, 1, 256, 256]
print("batch mask shape:", batch["mask"].shape)     # [B, 1, 256, 256]

# visualize one sample from batch
img = batch["image"][0, 0].numpy()
msk = batch["mask"][0, 0].numpy()

plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.title("Image")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Mask")
plt.imshow(msk, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Overlay")
plt.imshow(img, cmap="gray")
plt.imshow(msk, cmap="Reds", alpha=0.35)
plt.axis("off")

plt.tight_layout()
plt.show()