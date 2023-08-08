# Tutorial modificado de (https://www.aicrowd.com/showcase/tutorial-with-pytorch-torchvision-and-pytorch-lightning)
import math
import sys
import time
from tqdm.notebook import tqdm
import numpy as np
from pathlib import Path
import pandas as pd
import random
import cv2
import matplotlib.pyplot as plt

from PIL import Image

# Torch imports 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.ops.boxes import box_iou
import torch.nn.functional as F

# Albumentations is used for the Data Augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

class TomatoDataset(Dataset):
	def __init__(self, csv_file, images_dir, transform=None):
		self.images_dir = Path(images_dir)
		annotations = pd.read_csv(csv_file)
		self.image_list = annotations["image"].unique()
		self.bboxes = [annotations[annotations.image==x][["xmin", "ymin", "xmax", "ymax"]].values for x in self.image_list]
		self.transform = transform

	def __len__(self):
		return len(self.image_list)

	def __getitem__(self, idx):
		imgp = str(self.images_dir / 
			(self.image_list[idx]))
		bboxes = self.bboxes[idx]
		# img = cv2.imread(imgp)
		image = Image.open(imgp)
		# No usamos labels pero ciertas cabeceras de funciones fuerzan a introducir algún valor
		labels = np.ones(len(bboxes))

		if self.transform:
			transformed = self.transform(image=image, bboxes=bboxes)
			image = transformed["image"]
			bboxes = transformed["bboxes"]

		if len(bboxes)>0:
			bboxes = torch.stack([torch.tensor(item) for item in bboxes])
		else:
			bboxes = torch.zeros((0,4))

		return image, bboxes, labels

	def collate_fn(batch):
		images, targets, metadatas = list(), list(), list()
		for image, target, metadata in batch:
			images.append(image)
			targets.append(target)
			metadatas.append(metadata)
		images = torch.stack(images, dim=0)
		return images, targets, metadatas


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def get_train_transforms(target_img_size=512):
    return A.Compose(
        [
            A.Resize(height=target_img_size, width=target_img_size, p=1),
            A.Normalize(mean, std),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )

def get_valid_transforms(target_img_size=512):
    return A.Compose(
        [
            A.Resize(height=target_img_size, width=target_img_size, p=1),
            A.Normalize(mean, std),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )

def get_test_transforms(target_img_size=512):
    return A.Compose(
        [
            A.Resize(height=target_img_size, width=target_img_size, p=1),
            A.Normalize(mean, std),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )