import torch
import torch.nn as n
from lightning.pytorch import LightningModule

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights

from modelos.utils import image_sizes, compute_loss, compute_single_loss, threshold_fusion
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from fastcore.basics import patch
# from torchvision.models.detection.anchor_utils import AnchorGenerator
# from objdetecteval.metrics.coco_metrics import get_coco_stats
# from objdetecteval.metrics.image_metrics import get_inference_metrics

import sys
sys.path.append("..")
import config

class RetinaNetLightning(LightningModule):
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
        self.model = retinanet_resnet50_fpn(
            weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT,
            weights_backbone = ResNet50_Weights.IMAGENET1K_V1
        )
        self.threshold = threshold
        self.iou_thr = iou_thr
        self.loss_fn = compute_loss
        self.mean_ap = MeanAveragePrecision()
        self.mean_ap.warn_on_many_detections=False
        self.validation_outputs = []
        
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
        self.validation_outputs.append(outputs)
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
        self.log('test_class_loss', loss['class'])
        self.log('test_box_loss', loss['box'])
        for k in config.KEYS:
            self.log("test_"+k, mean_ap[k], logger=True)
        return {'loss' : loss['total'], 'batch_predictions' : batch_predictions}

    # def on_validation_epoch_end(self):
    #     """
    #     Añadido a cada etapa de validación en el que se evalúan los resultados
    #     del modelo con las estadísticas de COCO.
    #     """
    #     outputs = torch.stack(self.validation_outputs)
    #     stats = model.mean_ap(outputs)
    #     for k in config.KEYS:
    #         self.log("mean_val_"+k, stats[k], logger=True)
    #     return {'mean_epoch_val_loss': outputs['loss'], 'metrics': stats}