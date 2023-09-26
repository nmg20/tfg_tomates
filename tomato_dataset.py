import os
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as T
from torchvision.io import read_image
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pathlib import Path

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

main_ds = "/media/rtx3090/Disco2TB/cvazquez/nico/datasets/T1280x720/"
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
        return image, bboxes, class_labels

def get_basic_transform():
    """
    Transformaciones mínimas para aplicar a las imágenes y anotaciones.
    Base para añadir augmentations.
    """
    return A.Compose([
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2(p=1),
    ],
    p=1.0,
    bbox_params=A.BboxParams(
        format="pascal_voc", min_area=0, min_visibility=0,
        label_fields=["labels"])
    )

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
            class_labels
        ) = self.ds.img_and_labels_by_idx(idx)
        sample = {
            "image": np.array(image, dtype=np.float32),
            "bboxes": bboxes,
            "labels": class_labels,
        }
        sample = self.transforms(**sample)
        image = torch.as_tensor(sample["image"], dtype=torch.float32)
        sample["bboxes"][:, [0,1,2,3]] = [sample["bboxes"][
            :, [1, 0, 3, 2]]][0] #intercambiar xxyy a yxyx
        bboxes = torch.as_tensor(np.array(sample["bboxes"]))
        labels = torch.as_tensor(sample["labels"])
        return image, bboxes, labels

class TomatoDataModule(pl.LightningDataModule):
    """
    Datamodule: recibe la ruta a las imágenes, a cada dataframe y crea
    un dataset de entrenamiento y otro de validacion con sus respectivos
    dataloaders. 
        -> opcional: cambiar ruta del dataframe al dataframe directamente(?)
    """
    # def __init__(self, train_dataframe, val_dataframe, image_dir, batch_size, num_workers):
    def __init__(self, dfs_path, images_path, batch_size, num_workers):
        self.train_df_path = f"{dfs_path}/labelstrain.csv"
        self.val_dataframe = f"{dfs_path}/labelsval.csv"
        self.images_path = images_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        super().__init__()

    def train_dataset(self) -> TomatoDataset:
        return TomatoDataset(
            adaptor = TomatoDatasetAdaptor(
                self.images_path,self.train_df_path)
            )

    def train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset()
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = self.num_workers,
            # collate_fn = self.collate_fn,
        )
        return train_loader

    def val_dataset(self) -> TomatoDataset:
        return TomatoDataset(
            adaptor = TomatoDatasetAdaptor(
                self.images_path,self.val_df_path)
            )

    def val_dataloader(self) -> DataLoader:
        val_dataset = self.train_dataset()
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = self.num_workers,
            # collate_fn = self.collate_fn,
        )
        return val_loader

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def create_model(self, num_classes=1):
    """
    Función para crear el modelo FasterRCNN con los pesos por defecto,
    adaptando las características del predictor del módulo al número de clases.
    """
    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features, num_classes=1)
    return model

class FasterRCNNThres(FasterRCNN):
    """
    Modificación de FasterRCNN con una variable entrenable a modo de 
    umbral de resultados.
    """
    # def __init__(self, backbone, num_classes, init_threshold=0.5):
    #     super().__init__(backbone, num_classes)
    #     self.model = create_model(num_classes)
    #     self.threshold = nn.Parameter(torch.tensor(init_threshold), requires_grad=True)

    def __init__(self, backbone, num_classes, thres=0.5):
        super().__init__(backbone, num_classes)
        self.model = FasterRCNN(backbone, num_classes)
        self.threshold = nn.Parameter(torch.tensor(thres), requires_grad=True)

    def forward(self, images, targets=None):
        if self.training:
            return self.model(images, targets)
        else:
            outputs, scores = self.mode(images)
            thresholded = self.threshold + scores.shape[0]*0.5
            indices = torch.nonzero(scores > thresholded).squeeze()
            return outputs[indices]

class FasterRCNNModule(pl.LightningModule):
    def __init__(
        self,
        num_classes=1,
        lr=0.0002,
        # model_backbone="fasterrcnn",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.model = FasterRCNNThres(num_classes)
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    # def training_step(self, batch, batch_idx):
    #     images, 