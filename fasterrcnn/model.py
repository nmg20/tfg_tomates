from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import os
import torch
import torch.nn.functional as F

from fastcore.dispatch import typedispatch
from pytorch_lightning import LightningModule

import timm

class FasterRCNNModel(LightningModule):
    def __init__(
        self,
        num_classes=1,
        image_mean=None,
        image_std=None,
        score_thresh=0.05,
        learning_rate=0.0002,
        # inference_transforms=get_pred_transforms(target_img_size=512),
    ):
        super().__init__()
        model = create_model(num_classes)
        self.lr = learning_rate

    def create_model(self, num_classes=1):
        model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features=in_features, num_classes=1)
        return model

    def forward(self, images):
        self.model.eval()
        return self.model(images)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
            lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, annotations, labels, _ = batch
        loss_dict = self.model(images, )
        outputs = self.detector(images, annotations)
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
    