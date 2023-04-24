import numpy as np
import cv2 as cv
import torch
torch.manual_seed(17)
import pandas as pd
from utils.dataset_to_csv import *
from utils.Visualize import *

datasets_dir = "/media/rtx3090/Disco2TB/cvazquez/nico/datasets/"
main_ds = "/media/rtx3090/Disco2TB/cvazquez/nico/datasets/Tomato_1280x720/"

from pathlib import Path

def read_anots(image):
    df = pd.read_csv(main_ds+"/ImageSets/all_annotations.csv")
    return df[df.image==image.filename.split("/")[-1]][["xmin", "ymin", "xmax", "ymax"]].values

class TomatoDatasetAdaptor:
    def __init__(self, images_dir_path, annotations_dataframe):
        self.images_dir_path = Path(images_dir_path)
        self.annotations_df = annotations_dataframe
        self.images = self.annotations_df.image.unique().tolist()

    def __len__(self) -> int:
        return len(self.images)

    def get_image_and_labels_by_idx(self, index):
        image_name = self.images[index]
        # image = Image.open(self.images_dir_path / image_name)
        image = cv.imread(f"{str(self.images_dir_path)}/{image_name}")
        pascal_bboxes = self.annotations_df[self.annotations_df.image == image_name][
            ["xmin", "ymin", "xmax", "ymax"]
        ].values
        class_labels = np.ones(len(pascal_bboxes))

        return image, pascal_bboxes, class_labels, index

    # def get_annots_of_img(self, image):
    #     return self.annotations_df[self.annotations_df.image==image][
    #         ["xmin","ymin","xmax","ymax"]].values

    # def get_anots_array(self):
    #     return self.annotations_df[["xmin","ymin","xmax","ymax"]].values

    def get_imgs_and_anots(self):
        return [self.get_image_and_labels_by_idx(i) for i in range(len(self.images))]    

    # def get_imgs(self):
    #     return [Image.open(Path(self.images_dir_path/x)) for x in self.images]

    def show_image(self, index):
        image, bboxes, class_labels, image_id = self.get_image_and_labels_by_idx(index)
        print(f"image_id: {image_id}")
        show_image(image, bboxes.tolist())
        print(class_labels)


from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations import transforms

def get_train_transforms(target_img_size=512):
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2),
            # transforms.ColorJitter(contrast=0.2),
            # transforms.ColorJitter(saturation=0.3),
            # transforms.Equalize(mode='pil',by_channels=True),
            A.Resize(height=target_img_size, width=target_img_size, p=1),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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
            # transforms.ColorJitter(brightness=0.2),
            # transforms.ColorJitter(contrast=0.2),
            # transforms.ColorJitter(saturation=0.3),
            # transforms.Equalize(mode='pil',by_channels=True),
            A.Resize(height=target_img_size, width=target_img_size, p=1),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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
            # transforms.ColorJitter(brightness=0.2),
            # transforms.ColorJitter(contrast=0.2),
            # transforms.ColorJitter(saturation=0.3),
            # transforms.Equalize(mode='pil',by_channels=True),
            A.Resize(height=target_img_size, width=target_img_size, p=1),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )

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
        sample["bboxes"][:, [0, 1, 2, 3]] = sample["bboxes"][
            :, [1, 0, 3, 2]
        ]  # convert to yxyx

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

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

class EfficientDetDataModule(LightningDataModule):
    
    def __init__(self,
                train_dataset_adaptor,
                validation_dataset_adaptor,
                test_dataset_adaptor,
                train_transforms=get_train_transforms(target_img_size=512),
                valid_transforms=get_valid_transforms(target_img_size=512),
                test_transforms=get_test_transforms(target_img_size=512),
                num_workers=8,
                batch_size=4):
        
        self.train_ds = train_dataset_adaptor
        self.valid_ds = validation_dataset_adaptor
        self.test_ds = test_dataset_adaptor
        self.train_tfms = train_transforms
        self.valid_tfms = valid_transforms
        self.test_tfms = test_transforms
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
            shuffle=True,
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
            # batch_size=self.batch_size,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
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

def load_dss(path,name):
    dataset_path = Path(path)
    images_path = dataset_path/"JPEGImages"
    dfs_path = dataset_path/"ImageSets"/name
    df_tr = pd.read_csv(dfs_path/"labelstrain.csv")
    df_ts = pd.read_csv(dfs_path/"labelstest.csv")
    df_vl = pd.read_csv(dfs_path/"labelsval.csv")

    train_ds = TomatoDatasetAdaptor(images_path, df_tr)
    test_ds = TomatoDatasetAdaptor(images_path, df_ts)
    val_ds = TomatoDatasetAdaptor(images_path, df_vl)
    return train_ds, test_ds, val_ds

def get_dm(train,val,test):
    return EfficientDetDataModule(train_dataset_adaptor=train, 
        validation_dataset_adaptor=val,
        test_dataset_adaptor=test,
        num_workers=4,
        batch_size=2)

def get_dm_standalone(path,name):
    train, test, val = load_dss(path,name)
    return get_dm(train,val,test)

def get_dm2(name):
    train,test,val=load_dss(main_ds,name)
    return get_dm(train,val,test)

def get_dms_dss(dm):
    return dm.train_ds,dm.valid_ds,dm.test_ds

def get_main_df(path=main_ds):
    dataset_path = Path(path)
    return pd.read_csv(dataset_path/"ImageSets/all_annotations.csv")

# def get_main_ds(path):
#     dataset_path = Path(path)
#     df = get_main_df(path)
#     return TomatoDatasetAdaptor(dataset_path/"JPEGImages",df)

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
