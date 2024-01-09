import numpy as np
import torch
import torch.nn as n
from lightning.pytorch import LightningModule
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models.vgg import VGG16_Weights
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from modelos.utils import image_sizes, compute_loss, compute_single_loss, threshold_fusion

from fastcore.basics import patch
from neptune.types import File

import sys
sys.path.append("..")
import config

class SSDLightning(LightningModule):
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
        self.model = ssd300_vgg16(
            weights = SSD300_VGG16_Weights.DEFAULT,
            weights_backbone = VGG16_Weights.IMAGENET1K_V1
        )
        self.threshold = threshold
        self.iou_thr = iou_thr
        self.loss_fn = compute_loss
        self.mean_ap = MeanAveragePrecision()
        self.mean_ap.warn_on_many_detections=False
        
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
        self.log('train_class_loss', loss['classification'].detach())
        self.log('train_box_loss', loss['bbox_regression'].detach())
        loss = loss['classification'] + loss['bbox_regression']
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, targets, ids = batch
        outputs = self.forward(images, targets)
        batch_predictions = {
            'predictions' : outputs,
            'targets' : targets,
            'image_ids' : ids,
        }
        loss = self.loss_fn(outputs, targets)
        mean_ap = self.mean_ap(outputs, targets)
        self.log('val_class_loss', loss['class'])
        self.log('val_box_loss', loss['box'])
        for k in config.KEYS:
            self.log("val_"+k, mean_ap[k], logger=True)
        return {'loss' : loss['total'], 'batch_predictions' : batch_predictions}

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        images, targets, ids = batch
        outputs = self.forward(images, targets)
        batch_predictions = {
            # 'predictions' : [output['boxes'] for output in outputs],
            'predictions' : outputs,
            'targets' : targets,
            'image_ids' : ids,
        }
        loss = self.loss_fn(outputs, targets)
        mean_ap = self.mean_ap(outputs, targets)
        # for image, image_id in zip(images,ids):
        #     self.logger.experiment['test/images'].upload(
        #         File.as_image(image),
        #         description=f"image:{image_id}")
        self.log('test_class_loss', loss['class'])
        self.log('test_box_loss', loss['box'])
        for k in config.KEYS:
            self.log("test_"+k, mean_ap[k], logger=True)
        return {'loss' : loss['total'], 'batch_predictions' : batch_predictions}

@patch
def add_pred_outputs(self : SSDLightning, outputs):
    boxes, scores, labels, image_ids, targets = [],[],[],[],[]
    for i in range(len(outputs['batch_predictions']['predictions'])):
        preds = outputs['batch_predictions']['predictions'][i]
        boxes.append(preds['boxes'].detach().cpu().numpy())
        scores.append(preds['scores'].detach().cpu().numpy())
        labels.append(preds['labels'].detach().cpu().numpy())
        image_ids.append(outputs['batch_predictions']['image_ids'][i].detach().cpu().numpy())
        targets.append(outputs['batch_predictions']['targets'][i].detach().cpu().numpy())

    return (labels, image_ids, boxes, scores, targets)

    @patch 
    def validation_epoch_end(self, outputs=outputs):
        """
        Añadido a cada etapa de validación en el que se evalúan los resultados
        del modelo con las estadísticas de COCO.
        """
        (labels, image_ids, boxes, scores, targets) = self.add_pred_outputs(outputs)
        truth_ids, truth_boxes, truth_labels = zip(
            *[
                (
                    target['image_id'].detach().item(),
                    target['boxes'].detach().tolist(),
                    target["labels"].detach().tolist()
                ) for target in targets
            ]
        )
        stats = model.mean_ap(outputs)
        for k in config.KEYS:
            self.log("mean_val_"+k, stats[k], logger=True)
        return {'mean_epoch_val_loss': outputs['loss'], 'metrics': stats}