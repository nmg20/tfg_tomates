import os
import pandas as pd
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from lightning.pytorch import LightningDataModule
from pathlib import Path
import config
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class EffDetDatasetAdaptor:
    def __init__(self, images_path, df_path):
        self.images_path = images_path
        self.df = pd.read_csv(df_path)
        self.images = self.df.image.unique().tolist()

    def __len__(self) -> int:
        return len(self.images)

    def get_image_and_labels_by_idx(self, index):
        image_name = self.images[index]
        image = Image.open(self.images_path + image_name)
        bboxes = self.df[self.df.image == image_name][
            ["xmin", "ymin", "xmax", "ymax"]
        ].values
        class_labels = np.ones(len(bboxes))
        return image, bboxes, class_labels, index

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def denormalize(tensor):
    z = tensor * torch.tensor(std).view(3,1,1)
    z = z + torch.tensor(mean).view(3,1,1)
    return tfs.ToPILImage(mode="RGB")(z.squeeze(0))

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

class EffDetDataset(Dataset):
    def __init__(
        self, dataset_adaptor, transforms=get_valid_transforms()
    ):
        self.ds = dataset_adaptor
        self.transforms = transforms

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        (
            image,
            bboxes,
            class_labels,
            image_id,
        ) = self.ds.get_image_and_labels_by_idx(index)

        sample = {
            "image": np.array(image, dtype=np.float32),
            "bboxes": bboxes,
            "labels": class_labels,
        }

        sample = self.transforms(**sample)
        sample["bboxes"] = np.array(sample["bboxes"])
        image = sample["image"]
        bboxes = sample["bboxes"]
        labels = sample["labels"]

        _, new_h, new_w = image.shape
        sample["bboxes"][:, [0, 1, 2, 3]] = [sample["bboxes"][
            :, [1, 0, 3, 2]]][0]  # convert to yxyx

        target = {
            "bboxes": torch.as_tensor(sample["bboxes"], dtype=torch.float32),
            "boxes": torch.as_tensor(sample["bboxes"], dtype=torch.float32),
            "labels": torch.as_tensor(labels),
            "image_id": torch.tensor([image_id]),
            "img_size": (new_h, new_w),
            "img_scale": torch.tensor([1.0]),
        }
        return image, target, image_id

class EffDetDataModule(LightningDataModule):
    
    def __init__(self,
                dfs_path=config.MAIN_DS,
                images_path=config.IMAGES_DIR,
                num_workers=config.NUM_WORKERS,
                batch_size=config.BATCH_SIZE):
        self.train_df_path = f"{dfs_path}labelstrain.csv"
        self.val_df_path = f"{dfs_path}labelsval.csv"
        self.test_df_path = f"{dfs_path}labelstest.csv"
        self.images_path = images_path
        self.num_workers = num_workers
        self.batch_size = batch_size
        super().__init__()

    def train_dataset(self) -> EffDetDataset:
        return EffDetDataset(
            dataset_adaptor = EffDetDatasetAdaptor(
                self.images_path, self.train_df_path
            )
        )

    def train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset()
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        return train_loader

    def val_dataset(self) -> EffDetDataset:
        return EffDetDataset(
            dataset_adaptor = EffDetDatasetAdaptor(
                self.images_path, self.val_df_path
            )
        )

    def val_dataloader(self) -> DataLoader:
        val_dataset = self.train_dataset()
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        return val_loader

    def test_dataset(self) -> EffDetDataset:
        return EffDetDataset(
            dataset_adaptor = EffDetDatasetAdaptor(
                self.images_path, self.test_df_path
            )
        )

    def test_dataloader(self) -> DataLoader:
        test_dataset = self.train_dataset()
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        return test_loader
    
    @staticmethod
    def collate_fn(batch):
        images, targets, image_ids = tuple(zip(*batch))
        images = torch.stack(images)
        images = images.float()

        boxes = [target["bboxes"].float() for target in targets]
        labels = [target["labels"].float() for target in targets]
        img_size = torch.tensor([target["img_size"] for target in targets]).float()
        img_scale = torch.tensor([target["img_scale"] for target in targets]).float()

        annotations = {
            "bbox": boxes,
            "cls": labels,
            "img_size": img_size,
            "img_scale": img_scale,
        }

        return images, annotations, targets, image_ids