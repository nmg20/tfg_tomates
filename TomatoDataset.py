import os
import pandas as pd
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as T
from torchvision.io import read_image
from lightning.pytorch import LightningDataModule
from pathlib import Path

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

main_ds = "../datasets/T1280x720/"
images_dir = main_ds+"JPEGImages/"
imagesets_dir = main_ds+"ImageSets/"

class TomatoDatasetAdaptor:
    """
    Adaptador: recibe directorio con las imágenes y la ruta al dataframe
    """
    def __init__(self, images_path, df_path):
        self.images_path = images_path
        self.df = pd.read_csv(df_path)
        self.images = self.df.image.unique().tolist()

    def __len__(self) -> int:
        return len(self.images)

    def img_and_labels_by_idx(self, idx):
        image_name = self.images[idx]
        image = Image.open(f"{self.images_path}{image_name}")
        bboxes = self.df[self.df.image == image_name][
            ["xmin","ymin","xmax","ymax"]
        ].values
        class_labels = np.ones(len(bboxes))
        return image, bboxes, class_labels, idx
        # return image, bboxes, class_labels

    # def 

#plt.imshow(image_tensor.permute(1,2,0))

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def denormalize(tensor):
    z = tensor * torch.tensor(std).view(3,1,1)
    z = z + torch.tensor(mean).view(3,1,1)
    return z

def get_basic_transform():
    """
    Transformaciones mínimas para aplicar a las imágenes y anotaciones.
    Base para añadir augmentations.
    """
    return A.Compose([
        A.Normalize(mean, std),
        ToTensorV2(p=1),
    ],
    p=1.0,
    # bbox_params=A.BboxParams(
    #     format="pascal_voc", min_area=0, min_visibility=0,
    #     label_fields=["labels"])
    )
    # return T.ToTensor()

class TomatoDataset(Dataset):
    """
    Dataset compuesto: obtiene del adaptador las imágenes y las anotaciones
    y aplica las transformaciones a los datos.
    """
    def __init__(self, adaptor, transforms=get_basic_transform()):
        self.ds = adaptor 
        self.transforms = transforms

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        (
            image, 
            bboxes, 
            class_labels,
            image_id,
        ) = self.ds.img_and_labels_by_idx(idx)
        sample = {
            "image": np.array(image, dtype=np.float32),
            "bboxes": bboxes,
            "labels": class_labels,
        }
        sample = self.transforms(**sample)
        sample["bboxes"] = np.array(sample["bboxes"])
        image = sample["image"]
        pascal_bboxes = sample["bboxes"]
        labels = sample["labels"]

        # sample["bboxes"][:,[0, 1, 2, 3]] = [sample["bboxes"][
        #     :, [1, 0, 3, 2]]][0]

        target = {
            "boxes": torch.as_tensor(sample["bboxes"], dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.long),
            "image_id": torch.as_tensor([image_id])
        }
        return image, target, image_id
        # return image, target

class TomatoDataModule(LightningDataModule):
    """
    Datamodule: recibe la ruta a las imágenes, a cada dataframe y crea
    un dataset de entrenamiento y otro de validacion con sus respectivos
    dataloaders. 
        -> opcional: cambiar ruta del dataframe al dataframe directamente(?)
    """
    # def __init__(self, train_dataframe, val_dataframe, image_dir, batch_size, num_workers):
    def __init__(self, dfs_path, images_path, batch_size, num_workers):
        self.train_df_path = f"{dfs_path}labelstrain.csv"
        self.val_df_path = f"{dfs_path}labelsval.csv"
        self.test_df_path = f"{dfs_path}labelstest.csv"
        self.images_path = images_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        super().__init__()

    def train_dataset(self) -> TomatoDataset:
        return TomatoDataset(
            adaptor = TomatoDatasetAdaptor(
                self.images_path, self.train_df_path)
            )

    def train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset()
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size = self.batch_size,
            shuffle = True,
            pin_memory = False,
            num_workers = self.num_workers,
            collate_fn = self.collate_fn,
        )
        return train_loader

    def val_dataset(self) -> TomatoDataset:
        return TomatoDataset(
            adaptor = TomatoDatasetAdaptor(
                self.images_path, self.val_df_path)
            )

    def val_dataloader(self) -> DataLoader:
        val_dataset = self.train_dataset()
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            pin_memory = False,
            num_workers = self.num_workers,
            collate_fn = self.collate_fn,
        )
        return val_loader

    def test_dataset(self) -> TomatoDataset:
        return TomatoDataset(
            adaptor = TomatoDatasetAdaptor(
                self.images_path, self.test_df_path)
            )

    def test_dataloader(self) -> DataLoader:
        test_dataset = self.test_dataset()
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            pin_memory = False,
            num_workers = self.num_workers,
            collate_fn = self.collate_fn,
        )
        return test_loader

    @staticmethod
    def collate_fn(batch):
        images, targets, image_ids = tuple(zip(*batch))
        # images, targets = tuple(zip(*batch))
        images = torch.stack(images)
        images = images.float()

        boxes = [target["boxes"].float() for target in targets]
        labels = [target["labels"].float() for target in targets]
       
        annotations = {
            "bbox": boxes,
            "cls": labels,
        }
        # return images, annotations, targets, image_ids
        return images, targets, image_ids

