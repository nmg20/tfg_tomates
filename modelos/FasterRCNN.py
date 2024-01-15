import torch
import torch.nn as n
from lightning.pytorch import LightningModule

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

from modelos.utils import image_sizes, compute_loss, compute_single_loss, threshold_fusion
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from fastcore.basics import patch

import sys
sys.path.append("..")
import config

class FasterRCNNLightning(LightningModule):
    """
    Clase que contiene el funcionamiento básico de un modelo a efectos de entrenamiento/validación/test.
    Crea un modelo en concreto en base al backbone específicado.
    Versión del modelo simple sin umbralización.
    """
    def __init__(
        self,
        num_classes=config.NUM_CLASSES,
        lr=config.LR,
        threshold=0.2, #Threshold de scores de las bounding boxes
        iou_thr=config.IOU_THR, #Threshold de IoU para considerarse la misma bounding box
    ):
        super().__init__()
        self.lr = lr
        self.model = fasterrcnn_resnet50_fpn(
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
            weights_backbone = ResNet50_Weights.IMAGENET1K_V1,
        )
        self.threshold = threshold
        self.iou_thr = iou_thr
        self.loss_fn = compute_loss
        self.mean_ap = MeanAveragePrecision()
        self.mean_ap.warn_on_many_detections=False
        self.val_step_outputs = []
        self.val_step_targets = []
        
    def forward(self, images : torch.Tensor, targets=None):
        outputs = self.model(images, targets)
        if not self.model.training or targets is None:
            outputs = threshold_fusion(
                outputs,
                images,
                iou_thr=self.iou_thr,
                skip_box_thr=self.threshold
            )
        return outputs

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        images, targets, ids = batch
        loss = self.forward(images, targets)
        # Registramos el error de clasificación y regresión de bbox
        self.log('train_class_loss', loss['loss_classifier'].detach())
        self.log('train_box_loss', loss['loss_box_reg'].detach())
        self.log('train_objectness', loss['loss_objectness'].detach())
        self.log('train_loss_rpn_box_reg', loss['loss_rpn_box_reg'].detach())
        # self.log('train_class_loss', loss['classification'].detach())
        # self.log('train_box_loss', loss['bbox_regression'].detach())
        # loss = loss['classification'] + loss['bbox_regression']
        loss = sum(list(loss.values()))
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, targets, ids = batch
        outputs = self.forward(images, targets)
        
        loss = self.loss_fn(outputs, targets)
        self.log('val_class_loss', loss['class'])
        self.log('val_box_loss', loss['box'])
        mean_ap = self.mean_ap(outputs, targets)
        # for k in config.KEYS:
        #     self.log("val_"+k, mean_ap[k], logger=True)
        self.val_step_outputs.extend(outputs)
        self.val_step_targets.extend(targets)
        return loss['total']

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        images, targets, ids = batch
        outputs = self.forward(images, targets)

        loss = self.loss_fn(outputs, targets)
        mean_ap = self.mean_ap(outputs, targets)
        self.log('test_class_loss', loss['class'])
        self.log('test_box_loss', loss['box'])
        for k in config.KEYS:
            self.log("test_"+k, mean_ap[k], logger=True)
        return loss['total']

    def on_validation_epoch_end(self):
        val_all_outputs = self.val_step_outputs
        val_all_targets = self.val_step_targets
        mean_ap = self.mean_ap(val_all_outputs, val_all_targets)
        for k in config.KEYS:
            self.log("val_"+k, mean_ap[k], logger=True)
        self.val_step_outputs.clear()
        self.val_step_targets.clear()
