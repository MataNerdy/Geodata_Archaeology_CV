import numpy as np
import pandas as pd
from pathlib import Path

root = Path("/Users/Di/Documents/GitHub/My projects/Geodata_Archaeology_CV/datasets/segmentation_dataset")
sid = "000541"

mask = np.load(root / "masks" / f"{sid}.npy")

binary = np.isin(mask, [1, 2]).astype("uint8")

print("raw unique:", np.unique(mask, return_counts=True))
print("binary unique:", np.unique(binary, return_counts=True))
print("binary sum:", binary.sum())