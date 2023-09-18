from torchvision.io.image import read_image
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import os
import torch
import torch.nn.functional as F

from fastcore.dispatch import typedispatch
from pytorch_lightning import LightningModule

import timm

class RetinaNetModel(LightningModule):
    def __init__(
        self,
        num_classes=1,
        image_mean=None,
        image_std=None,
        score_thresh=0.05,inference_transforms=get_pred_transforms(target_img_size=512),
    ):
        super().__init__()
        self.model = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT)

    def forward(self, images):
        return self.model(images)

    def training_step(self, batch, batch_idx):
        images, bboxes, labels = batch
        outputs = self.model(images)
        loss = F.cross_entropy(outputs, bboxes)
        self.training_step_outputs.append(outputs)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, bboxes, labels = batch
        outputs = self.model(images)
        loss = F.cross_entropy(outputs, bboxes)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)