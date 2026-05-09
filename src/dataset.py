import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ButterflyDataset(Dataset):
    """
    Dataset for the ACA Butterfly competition.

    Accepts the raw CSV dataframe (with a 'label' string column).
    The class→index mapping is built internally, exactly like the
    professors' reference implementation in TP2-students.ipynb.
    """

    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.img_labels = df.reset_index(drop=True)
        self.img_dir    = img_dir
        self.transform  = transform
        self.is_test    = is_test

        # Build class ↔ index mapping from the label column (if present)
        if not is_test and "label" in self.img_labels.columns:
            self.classes      = sorted(self.img_labels["label"].unique())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        else:
            self.classes      = []
            self.class_to_idx = {}

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        row = self.img_labels.iloc[idx]

        img_path = os.path.join(self.img_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image, row["filename"]

        # Use pre-computed label_idx if available, otherwise map from string label
        if "label_idx" in row.index:
            label = int(row["label_idx"])
        else:
            label = self.class_to_idx[row["label"]]

        return image, torch.tensor(label, dtype=torch.long)