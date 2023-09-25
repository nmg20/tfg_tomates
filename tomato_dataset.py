import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from pathlib import Path
import pandas as pd

transforms = 

class TomatoDataset(Dataset):
    # def __init__(self, data_dir, df, transform=None):
    def __init__(self, data_dir, df, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images_dir = os.path.join(data_dir, "JPEGImages/")
        self.df = pd.read_csv(df)
        self.images = self.df.image.unique().tolist()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.images_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        bboxes = self.df[self.df.image == image_name][
            ["xmin","ymin","xmax","ymax"]
        ].values
        class_labels = np.ones(len(bboxes))

        if self.transform:
            image = self.transform(image)

        return image, bboxes, class_labels


