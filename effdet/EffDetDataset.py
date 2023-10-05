import numpy as np
import cv2 as cv
import torch
import os
torch.manual_seed(17)
import pandas as pd
from utils import dataset_to_csv
from Dataset_Analysis import read_imageset_names, Image
from pathlib import Path
from config import *

NUM_WORKERS = 8
# NUM_WORKERS = 1
BATCH_SIZE = 4

def read_anots(image):
    df = pd.read_csv(imagesets_dir+"all_annotations.csv")
    return df[df.image==image.filename.split("/")[-1]][["xmin", "ymin", "xmax", "ymax"]].values

def get_dir_imgs_names(imgs_dir=data_dir):
    return [x for x in os.listdir(imgs_dir) if x[len(x)-4::]=='.jpg']

class TomatoDatasetAdaptor:
    def __init__(self, images_dir_path, annotations_dataframe):
        self.images_dir_path = Path(images_dir_path)
        self.annotations_df = annotations_dataframe
        self.images = self.annotations_df.image.unique().tolist()

    def __len__(self) -> int:
        return len(self.images)

    def get_image_and_labels_by_idx(self, index):
        image_name = self.images[index]
        image = Image.open(self.images_dir_path / image_name)
        pascal_bboxes = self.annotations_df[self.annotations_df.image == image_name][
            ["xmin", "ymin", "xmax", "ymax"]
        ].values
        class_labels = np.ones(len(pascal_bboxes))

        return image, pascal_bboxes, class_labels, index
    def get_imgs_and_anots(self):
        return [self.get_image_and_labels_by_idx(i) for i in range(len(self.images))]    

    def show_image(self, index):
        image, bboxes, class_labels, image_id = self.get_image_and_labels_by_idx(index)
        print(f"image_id: {image_id}")
        show_image(image, bboxes.tolist())
        print(class_labels)

# class TomatoDatasetPredAdaptor:
#     def __init__(self, images_dir_path):
#         self.images_dir_path = Path(images_dir_path)

#     def get_image_by_idx(self, index):
#         image_name = self.images[index]
#         image = Image.open(self.images_dir_path / image_name)
#         return image


from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations import transforms
import torchvision.transforms as tfs

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def denormalize(tensor):
    z = tensor * torch.tensor(std).view(3,1,1)
    z = z + torch.tensor(mean).view(3,1,1)
    return tfs.ToPILImage(mode="RGB")(z.squeeze(0))

def get_train_transforms(target_img_size=512):
    return A.Compose(
        [
            # A.Blur(blur_limit=3, p=0.5),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            # A.Transpose(p=0.5),
            # A.RandomScale(p=0.3),
            # A.SafeRotate(p=0.5),
            # # A.augmentations.crops.transforms.BBoxSafeRandomCrop(p=0.5),
            # transforms.ColorJitter(brightness=0.2),
            # transforms.ColorJitter(contrast=0.2),
            # transforms.ColorJitter(saturation=0.3),
            # transforms.Equalize(mode='pil',by_channels=True),
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
            # A.Blur(blur_limit=3, p=0.5),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            # A.Transpose(p=0.5),
            # A.RandomScale(p=0.3),
            # A.SafeRotate(p=0.5),
            # transforms.ColorJitter(brightness=0.2),
            # transforms.ColorJitter(contrast=0.2),
            # transforms.ColorJitter(saturation=0.3),
            # transforms.Equalize(mode='pil',by_channels=True),
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

def get_pred_transforms(target_img_size=512):
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

# def get_pred_transforms(target_img_size=512):
#     return A.Compose(
#         [
#             A.Blur(blur_limit=3, p=0.5),
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.5),
#             A.Transpose(p=0.5),
#             A.RandomScale(p=0.3),
#             A.SafeRotate(p=0.5),
#             A.Resize(height=target_img_size,width=target_img_size,p=1),
#             A.Normalize(mean, std),
#             ToTensorV2(p=1),
#         ],
#         p=1.0,
#         bbox_params=A.BboxParams(
#             format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
#         ),
#     )


class EfficientDetDataset(Dataset):
    def __init__(
        self, dataset_adaptor, transforms=get_valid_transforms()
    ):
        self.ds = dataset_adaptor
        self.transforms = transforms

    def __getitem__(self, index):
        (
            image,
            pascal_bboxes,
            class_labels,
            image_id,
        ) = self.ds.get_image_and_labels_by_idx(index)

        sample = {
            "image": np.array(image, dtype=np.float32),
            "bboxes": pascal_bboxes,
            "labels": class_labels,
        }

        sample = self.transforms(**sample)
        sample["bboxes"] = np.array(sample["bboxes"])
        image = sample["image"]
        pascal_bboxes = sample["bboxes"]
        labels = sample["labels"]

        _, new_h, new_w = image.shape
        # print(f"\n{sample['bboxes']}\n")
        sample["bboxes"][:, [0, 1, 2, 3]] = [sample["bboxes"][
            :, [1, 0, 3, 2]]][0]  # convert to yxyx

        target = {
            "bboxes": torch.as_tensor(sample["bboxes"], dtype=torch.float32),
            "labels": torch.as_tensor(labels),
            "image_id": torch.tensor([image_id]),
            "img_size": (new_h, new_w),
            "img_scale": torch.tensor([1.0]),
        }

        return image, target, image_id

    def __len__(self):
        return len(self.ds)

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

class EfficientDetDataModule(LightningDataModule):
    
    def __init__(self,
                train_dataset_adaptor,
                validation_dataset_adaptor,
                test_dataset_adaptor,
                pred_dataset_adaptor,
                train_transforms=get_train_transforms(target_img_size=512),
                valid_transforms=get_valid_transforms(target_img_size=512),
                test_transforms=get_test_transforms(target_img_size=512),
                pred_transforms=get_pred_transforms(target_img_size=512),
                num_workers=NUM_WORKERS,
                batch_size=BATCH_SIZE):
        
        self.train_ds = train_dataset_adaptor
        self.valid_ds = validation_dataset_adaptor
        self.test_ds = test_dataset_adaptor
        self.pred_ds = pred_dataset_adaptor
        self.train_tfms = train_transforms
        self.valid_tfms = valid_transforms
        self.test_tfms = test_transforms
        self.pred_tfms = pred_transforms
        self.num_workers = num_workers
        self.batch_size = batch_size
        super().__init__()

    def train_dataset(self) -> EfficientDetDataset:
        return EfficientDetDataset(
            dataset_adaptor=self.train_ds, transforms=self.train_tfms
        )

    def train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset()
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        return train_loader

    def val_dataset(self) -> EfficientDetDataset:
        return EfficientDetDataset(
            dataset_adaptor=self.valid_ds, transforms=self.valid_tfms
        )

    def val_dataloader(self) -> DataLoader:
        valid_dataset = self.val_dataset()
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        return valid_loader

    def test_dataset(self) -> EfficientDetDataset:
        return EfficientDetDataset(
            dataset_adaptor=self.test_ds, transforms=self.test_tfms
        )

    def test_dataloader(self) -> DataLoader:
        test_dataset = self.test_dataset()
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            # batch_size=1,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        return test_loader

    def pred_dataset(self) -> EfficientDetDataset:
        return EfficientDetDataset(
            dataset_adaptor=self.pred_ds, transforms=self.pred_tfms
        )

    def pred_dataloader(self) -> DataLoader:
        pred_dataset = self.pred_dataset()
        pred_loader = torch.utils.data.DataLoader(
            pred_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        return pred_loader
    
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

def load_dss(name,path=main_ds):
    dataset_path = Path(path)
    images_path = dataset_path/"JPEGImages"
    dfs_path = dataset_path/"ImageSets"/name
    # print(dfs_path)
    df_tr = pd.read_csv(dfs_path/"labelstrain.csv")
    df_ts = pd.read_csv(dfs_path/"labelstest.csv")
    df_vl = pd.read_csv(dfs_path/"labelsval.csv")

    train_ds = TomatoDatasetAdaptor(images_path, df_tr)
    test_ds = TomatoDatasetAdaptor(images_path, df_ts)
    val_ds = TomatoDatasetAdaptor(images_path, df_vl)
    return train_ds, test_ds, val_ds

def get_data(ds='d801010',file="test"):
    if len(os.listdir(data_dir))==0:
        return get_data_ds(read_imageset_names(ds,file))
    else:
        return get_data_ds(get_dir_imgs_names(data_dir))

def get_dm(name="d801010", data_file="test", batch_size=BATCH_SIZE):
    """
    Carga primero los Datasets con anotaciones y luego lee las imágenes de una carpeta
    para crear el dataset para predecir.
    """
    train, test, val = load_dss(name,main_ds)
    pred = get_data(name,data_file)
    return EfficientDetDataModule(train_dataset_adaptor=train, 
        validation_dataset_adaptor=val,
        test_dataset_adaptor=test,
        pred_dataset_adaptor=pred,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE)

def get_main_df(path=main_ds):
    dataset_path = Path(path)
    return pd.read_csv(dataset_path/"ImageSets/all_annotations.csv")

def get_data_df(df, names):
    """
    Dado un dataframe maestro con todas las anotaciones y una lista de nombres,
    (los de la carpeta /data), crea un dataframe con las anotaciones
    correspondientes a las imágenes de la carpeta.
    -> es necesario que las imágenes de la carpeta estén anotadas en el .csv maestro.
    """
    dfs=[]
    for name in names:
        dfs.append(df[df.image==name])
    return pd.concat(dfs)

def get_data_ds(names, path=main_ds):
    """
    Dada la ruta de las imágenes y los nombres, crea el df a partir del .csv maestro
    y genera el dataset.
    """
    # Lee del .csv completo por defecto
    dataset_path = Path(path)
    main_df = get_main_df()
    df = get_data_df(main_df,names)
    return TomatoDatasetAdaptor(dataset_path/"JPEGImages",df)