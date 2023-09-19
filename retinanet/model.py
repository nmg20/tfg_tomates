from torchvision.io.image import read_image
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import os
import torch
import torch.nn.functional as F

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations import transforms
import torchvision.transforms as tfs

from fastcore.dispatch import typedispatch
from pytorch_lightning import LightningModule

import timm

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

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

class RetinaNetModel(LightningModule):
    def __init__(
        self,
        num_classes=1,
        image_mean=None,
        image_std=None,
        score_thresh=0.05,
        # inference_transforms=get_pred_transforms(target_img_size=512),
    ):
        super().__init__()
        self.model = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT)

    def forward(self, images, targets=None):
        return self.model(images)

    def training_step(self, batch, batch_idx):
        images, bboxes, labels = batch
        outputs = self.model(images)
        loss = F.cross_entropy(outputs, bboxes)
        self.training_step_outputs.append(outputs)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # @torch.no_grad()
    # def validation_step(self, batch, batch_idx):
    #     images, bboxes, labels, _ = batch
    #     outputs = self.model(images)
    #     loss = F.cross_entropy(outputs, bboxes)
    #     self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

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
        self.log("valid_loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True,
                 logger=True, sync_dist=True)
        self.log("valid_box_loss", logging_losses["box_loss"], on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        return {'loss': outputs["loss"], 'batch_predictions': batch_predictions}
    