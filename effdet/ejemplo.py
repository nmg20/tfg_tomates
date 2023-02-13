from numbers import Number
from typing import List
from functools import singledispatch

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import numpy as np
import torch

from fastcore.dispatch import typedispatch
from pytorch_lightning import LightningModule
# from pytorch_lightning.core.decorators import auto_move_data

from ensemble_boxes import ensemble_boxes_wbf

from effdet.config.model_config import efficientdet_model_param_dict
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet
from effdet.config.model_config import efficientdet_model_param_dict

import timm

from pathlib import Path
import PIL
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches

from pycocotools import coco, cocoeval, mask

dataset_path = Path("../../tomates512/")
train_data_path = dataset_path/"images/train/"
test_data_path = dataset_path/"images/test/"
val_data_path = dataset_path/"images/val/"

df_tr = pd.read_csv(dataset_path/"annotations/labelstrain.csv")
df_ts = pd.read_csv(dataset_path/"annotations/labelstest.csv")
df_vl = pd.read_csv(dataset_path/"annotations/labelsval.csv")

##################
#DatasetAdaptor
##################

def get_rectangle_edges_from_pascal_bbox(bbox):
    xmin_top_left, ymin_top_left, xmax_bottom_right, ymax_bottom_right = bbox

    bottom_left = (xmin_top_left, ymax_bottom_right)
    width = xmax_bottom_right - xmin_top_left
    height = ymin_top_left - ymax_bottom_right

    return bottom_left, width, height

def draw_pascal_voc_bboxes(
    plot_ax,
    bboxes,
    get_rectangle_corners_fn=get_rectangle_edges_from_pascal_bbox,
):
    for bbox in bboxes:
        bottom_left, width, height = get_rectangle_corners_fn(bbox)

        rect_1 = patches.Rectangle(
            bottom_left,
            width,
            height,
            linewidth=1,
            edgecolor="red",
            fill=False,
        )
        # rect_2 = patches.Rectangle(
        #     bottom_left,
        #     width,
        #     height,
        #     linewidth=1,
        #     edgecolor="white",
        #     fill=False,
        # )

        # Add the patch to the Axes
        plot_ax.add_patch(rect_1)
        # plot_ax.add_patch(rect_2)

def show_image(
    image, bboxes=None, draw_bboxes_fn=draw_pascal_voc_bboxes, figsize=(10, 10)
):
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)

    if bboxes is not None:
        draw_bboxes_fn(ax, bboxes)

    plt.show()


######################
# TomatoDatasetAdaptor
######################

class TomatoDatasetAdaptor:
    def __init__(self, images_dir_path, annotations_dataframe):
        self.images_dir_path = Path(images_dir_path)
        self.annotations_df = annotations_dataframe
        self.images = self.annotations_df.image.unique().tolist()

    def __len__(self) -> int:
        return len(self.images)

    def get_image_and_labels_by_idx(self, index):
        image_name = self.images[index]
        image = PIL.Image.open(self.images_dir_path / image_name)
        pascal_bboxes = self.annotations_df[self.annotations_df.image == image_name][
            ["xmin", "ymin", "xmax", "ymax"]
        ].values
        class_labels = np.ones(len(pascal_bboxes))

        return image, pascal_bboxes, class_labels, index
    
    def show_image(self, index):
        image, bboxes, class_labels, image_id = self.get_image_and_labels_by_idx(index)
        print(f"image_id: {image_id}")
        show_image(image, bboxes.tolist())
        # print(class_labels)


tomato_train_ds = TomatoDatasetAdaptor(train_data_path, df_tr)
tomato_test_ds = TomatoDatasetAdaptor(test_data_path, df_ts)
tomato_val_ds = TomatoDatasetAdaptor(val_data_path, df_vl)

def get_train_transforms(target_img_size=512):
    # random angle, random blur, flip, random noise
    # random combination of 3, random scale, random translation
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
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
    def __init__(self, dataset_adaptor, transforms=get_valid_transforms(), test=True):
        self.ds = dataset_adaptor
        self.transforms = transforms
        self.test = test
        self.num_imgs = len(self.ds)

    def __getitem__(self, index):
        (
            image,
            pascal_bboxes,
            class_labels,
            image_id,
        ) = self.ds.get_image_and_labels_by_idx(index)

        orig_sample = {
            "image": image,
            "bboxes": pascal_bboxes,
            "labels": class_labels,
        }

        if not self.test:
            for i in range(10):
                sample = self.transforms(**orig_sample)
                
                if len(sample["bboxes"]) > 0:
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
        else:
            sample = self.transforms(**orig_sample)
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


class EfficientDetDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset_adaptor,
        validation_dataset_adaptor,
        train_transforms=get_train_transforms(target_img_size=512),
        valid_transforms=get_valid_transforms(target_img_size=512),
        num_workers=8,
        batch_size=4,
    ):

        self.train_ds = train_dataset_adaptor
        self.valid_ds = validation_dataset_adaptor
        self.train_tfms = train_transforms
        self.valid_tfms = valid_transforms
        self.num_workers = num_workers
        self.batch_size = batch_size
        super().__init__()

    def train_dataset(self) -> EfficientDetDataset:
        return EfficientDetDataset(
            dataset_adaptor=self.train_ds, transforms=self.train_tfms, test=False
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
            dataset_adaptor=self.valid_ds, transforms=self.valid_tfms, test=True
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

class EfficientDetModel(LightningModule):
    def __init__(
        self,
        num_classes=2,
        img_size=512,
        prediction_confidence_threshold=0.1,
        learning_rate=1e-3,
        wbf_iou_threshold=0.4,
        inference_transforms=get_valid_transforms(target_img_size=512),
        model_architecture="tf_efficientnetv2_b0",
        val_imgs=None
    ):
        super().__init__()
        self.img_size = img_size
        self.model = create_model(
            num_classes, img_size, architecture=model_architecture
        )
        self.prediction_confidence_threshold = prediction_confidence_threshold
        self.lr = learning_rate
        self.wbf_iou_threshold = wbf_iou_threshold
        self.inference_tfms = inference_transforms
        self.val_imgs = val_imgs

    def forward(self, images, targets):
        return self.model(images, targets)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        images, annotations, _, image_ids = batch

        losses = self.model(images, annotations)

        logging_losses = {
            "class_loss": losses["class_loss"].detach(),
            "box_loss": losses["box_loss"].detach(),
        }

        self.log(
            "train_loss",
            losses["loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_class_loss",
            losses["class_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_box_loss",
            losses["box_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return losses["loss"]

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, annotations, targets, image_ids = batch
        outputs = self.model(images, annotations)

        detections = outputs["detections"]

        batch_predictions = {
            "predictions": detections,
            "targets": targets,
            "image_ids": image_ids,
        }

        logging_losses = {
            "class_loss": outputs["class_loss"].detach(),
            "box_loss": outputs["box_loss"].detach(),
        }

        self.log(
            "valid_loss",
            outputs["loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "valid_class_loss",
            logging_losses["class_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,

        )
        self.log(
            "valid_box_loss",
            logging_losses["box_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        
        # if batch_idx == 0:
        #     images = []
            # for i in range(2):
            #     original_image = batch_images[i].permute(1, 2, 0).detach().cpu()
            #     reconstructed_image = reconstructed_images[i].permute(1, 2, 0).detach().cpu()
            #     image = torch.cat((original_image, reconstructed_image), dim=1)
            #     images.append(image.numpy())
            # self.logger.log_image(key="reconstructions", images=images)

        return {"loss": outputs["loss"], "batch_predictions": batch_predictions}

    @typedispatch
    def predict(self, images: List):
        """
        For making predictions from images
        Args:
            images: a list of PIL images

        Returns: a tuple of lists containing bboxes, predicted_class_labels, predicted_class_confidences

        """
        image_sizes = [(image.size[1], image.size[0]) for image in images]
        images_tensor = torch.stack(
            [
                self.inference_tfms(
                    image=np.array(image, dtype=np.float32),
                    labels=np.ones(1),
                    bboxes=np.array([[0, 0, 1, 1]]),
                )["image"]
                for image in images
            ]
        )

        return self._run_inference(images_tensor, image_sizes)

    def validation_epoch_end(self, outputs):
        """Compute and log training loss and accuracy at the epoch level."""

        validation_loss_mean = torch.stack(
            [output["loss"] for output in outputs]
        ).mean()

        (
            predicted_class_labels,
            image_ids,
            predicted_bboxes,
            predicted_class_confidences,
            targets,
        ) = self.aggregate_prediction_outputs(outputs)
        
                
        # print('######outputs########')
        # print(outputs)
        # print(predicted_class_labels)
        # print('######image_ids########')
        # print(image_ids)
        # print('######predicted_bboxes########')
        # print(predicted_bboxes)
        # print('######predicted_class_confidences########')
        # print(predicted_class_confidences)
        # print('######targets########')
        # print(targets)

        truth_image_ids = [target["image_id"].detach().item() for target in targets]
        truth_boxes = [
            target["bboxes"].detach()[:, [1, 0, 3, 2]].tolist() for target in targets
        ]  # convert to xyxy for evaluation
        truth_labels = [target["labels"].detach().tolist() for target in targets]

        
        if self.val_imgs:
            validation_imgs = []
            img_samples = min(len(image_ids), 8)
            
            for i in range(img_samples):
                img_id = image_ids[i]
                img_path = self.val_imgs[img_id]
                
                image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                pred_bbox = predicted_bboxes[i]
                pred_class = predicted_class_labels[i]
                
                # Predictions
                for j in range(len(pred_bbox)):
                    pt1 = (int(pred_bbox[j][0]), int(pred_bbox[j][1]))
                    pt2 = (int(pred_bbox[j][2]), int(pred_bbox[j][3]))
                    
                    color = PREDS_CLASS_COLORS[int(pred_class[j])]
                    cv2.rectangle(image, pt1, pt2, color, thickness=1)
                
                # Targets
                bbox_targets = truth_boxes[i]
                labels = truth_labels[i]
                
                for j in range(len(bbox_targets)):
                    pt1 = (int(bbox_targets[j][0]), int(bbox_targets[j][1]))
                    pt2 = (int(bbox_targets[j][2]), int(bbox_targets[j][3]))
                    
                    color = LABELS_CLASS_COLORS[int(labels[j])]
                    cv2.rectangle(image, pt1, pt2, color, thickness=2)
                    
                validation_imgs.append(image)
            self.logger.log_image(key="validation_samples", images=validation_imgs)

        stats = get_coco_stats(
            prediction_image_ids=image_ids,
            predicted_class_confidences=predicted_class_confidences,
            predicted_bboxes=predicted_bboxes,
            predicted_class_labels=predicted_class_labels,
            target_image_ids=truth_image_ids,
            target_bboxes=truth_boxes,
            target_class_labels=truth_labels,
        )["All"]

        self.log("val_loss", validation_loss_mean, on_epoch=True, logger=True)

        self.log("mAP_0_50", stats["AP_all_IOU_0_50"], on_epoch=True, logger=True)

        self.log("mAP_0_50_95", stats["AP_all"], on_epoch=True, logger=True)

        return {"val_loss": validation_loss_mean, "metrics": stats}

    @typedispatch
    def predict(self, images_tensor: torch.Tensor):
        """
        For making predictions from tensors returned from the model's dataloader
        Args:
            images_tensor: the images tensor returned from the dataloader

        Returns: a tuple of lists containing bboxes, predicted_class_labels, predicted_class_confidences

        """
        if images_tensor.ndim == 3:
            images_tensor = images_tensor.unsqueeze(0)
        if (
            images_tensor.shape[-1] != self.img_size
            or images_tensor.shape[-2] != self.img_size
        ):
            raise ValueError(
                f"Input tensors must be of shape (N, 3, {self.img_size}, {self.img_size})"
            )

        num_images = images_tensor.shape[0]
        image_sizes = [(self.img_size, self.img_size)] * num_images

        return self._run_inference(images_tensor, image_sizes)

    def aggregate_prediction_outputs(self, outputs):
        detections = torch.cat(
            [output["batch_predictions"]["predictions"] for output in outputs]
        )

        image_ids = []
        targets = []
        for output in outputs:
            batch_predictions = output["batch_predictions"]
            image_ids.extend(batch_predictions["image_ids"])
            targets.extend(batch_predictions["targets"])

        (
            predicted_bboxes,
            predicted_class_confidences,
            predicted_class_labels,
        ) = self.post_process_detections(detections)

        return (
            predicted_class_labels,
            image_ids,
            predicted_bboxes,
            predicted_class_confidences,
            targets,
        )

    def _run_inference(self, images_tensor, image_sizes):
        dummy_targets = self._create_dummy_inference_targets(
            num_images=images_tensor.shape[0]
        )

        detections = self.model(images_tensor.to(self.device), dummy_targets)[
            "detections"
        ]
        (
            predicted_bboxes,
            predicted_class_confidences,
            predicted_class_labels,
        ) = self.post_process_detections(detections)

        scaled_bboxes = self.__rescale_bboxes(
            predicted_bboxes=predicted_bboxes, image_sizes=image_sizes
        )

        return scaled_bboxes, predicted_class_labels, predicted_class_confidences

    def _create_dummy_inference_targets(self, num_images):
        dummy_targets = {
            "bbox": [
                torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=self.device)
                for i in range(num_images)
            ],
            "cls": [torch.tensor([1.0], device=self.device) for i in range(num_images)],
            "img_size": torch.tensor(
                [(self.img_size, self.img_size)] * num_images, device=self.device
            ).float(),
            "img_scale": torch.ones(num_images, device=self.device).float(),
        }

        return dummy_targets

    def post_process_detections(self, detections):
        predictions = []
        for i in range(detections.shape[0]):
            predictions.append(
                self._postprocess_single_prediction_detections(detections[i])
            )

        predicted_bboxes, predicted_class_confidences, predicted_class_labels = run_wbf(
            predictions, image_size=self.img_size, iou_thr=self.wbf_iou_threshold
        )

        return predicted_bboxes, predicted_class_confidences, predicted_class_labels

    def _postprocess_single_prediction_detections(self, detections):
        boxes = detections.detach().cpu().numpy()[:, :4]
        scores = detections.detach().cpu().numpy()[:, 4]
        classes = detections.detach().cpu().numpy()[:, 5]
        indexes = np.where(scores > self.prediction_confidence_threshold)[0]
        boxes = boxes[indexes]

        return {"boxes": boxes, "scores": scores[indexes], "classes": classes[indexes]}

    def __rescale_bboxes(self, predicted_bboxes, image_sizes):
        scaled_bboxes = []
        for bboxes, img_dims in zip(predicted_bboxes, image_sizes):
            im_h, im_w = img_dims

            if len(bboxes) > 0:
                scaled_bboxes.append(
                    (
                        np.array(bboxes)
                        * [
                            im_w / self.img_size,
                            im_h / self.img_size,
                            im_w / self.img_size,
                            im_h / self.img_size,
                        ]
                    ).tolist()
                )
            else:
                scaled_bboxes.append(bboxes)

        return scaled_bbox

dm = EfficientDetDataModule(train_dataset_adaptor=tomato_train_ds, 
        validation_dataset_adaptor=tomato_train_ds,
        num_workers=4,
        batch_size=2)

model = EfficientDetModel(
    num_classes=1,
    img_size=512
    )

from pytorch_lightning import Trainer 

trainer = Trainer(
        gpus=1, max_epochs=5, num_sanity_val_steps=1,
    )
trainer.fit(model,dm)

# ######################

img1, truth_bboxes1, _, _ = tomato_train_ds.get_image_and_labels_by_idx(0)
img2, truth_bboxes2, _, _ = tomato_train_ds.get_image_and_labels_by_idx(1)

images = [img1, img2]

predicted_bboxes, predicted_class_confidences, predicted_class_labels = model.predict(images)

def compare_bboxes_for_image(
    image,
    predicted_bboxes,
    actual_bboxes,
    draw_bboxes_fn=draw_pascal_voc_bboxes,
    figsize=(20, 20),
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.imshow(image)
    ax1.set_title("Prediction")
    ax2.imshow(image)
    ax2.set_title("Actual")

    draw_bboxes_fn(ax1, predicted_bboxes)
    draw_bboxes_fn(ax2, actual_bboxes)

    plt.show()

compare_bboxes_for_image(img1, predicted_bboxes=predicted_bboxes[0],actual_bboxes=truth_bboxes1.tolist())
