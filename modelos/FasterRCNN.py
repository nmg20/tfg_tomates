import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection.anchor_utils import AnchorGenerator

import Visualize
from modelos.utils import image_sizes, compute_loss, threshold_fusion

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium') 

models_dir = "./pths/"

class FasterRCNNLightning(LightningModule):
    """
    Clase que contiene el funcionamiento básico de un modelo a efectos de entrenamiento/validación/test.
    Crea un modelo en concreto en base al backbone específicado.
    Versión del modelo simple sin umbralización.
    """
    def __init__(
        self,
        num_classes=1,
        lr=0.0002,
        threshold=0.1, #Threshold de scores de las bounding boxes
        iou_thr=0.3, #Threshold de IoU para considerarse la misma bounding box
    ):
        super().__init__()
        self.lr = lr
        self.model = fasterrcnn_resnet50_fpn(
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
            weights_backbone = ResNet50_Weights.IMAGENET1K_V1,
        )
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.threshold = threshold
        self.iou_thr = iou_thr
        self.loss_fn = compute_loss
        self.model.num_classes = num_classes
    
    def forward(self, images, targets=None):
        outputs = self.model(images, targets)
        if not self.model.training or targets is None:
            outputs = threshold_fusion(
                outputs,
                image_sizes(images),
                iou_thr=self.iou_thr,
                skip_box_thr=self.threshold
            )
        return outputs

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss = self(images, targets)
        # Registramos el error de clasificación y regresión de bbox
        self.log('train_class_loss', loss['loss_classifier'].detach())
        self.log('train_box_loss', loss['loss_box_reg'].detach())
        self.log('train_objectness', loss['loss_objectness'].detach())
        self.log('train_rpn_box_loss', loss['loss_rpn_box_reg'].detach())
        return {'loss' : loss['bbox_regression']}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images, targets)
        batch_predictions = {
            'predictions' : [output['boxes'] for output in outputs],
            'targets' : targets,
            # 'image_ids' : ids,
        }
        loss = self.loss_fn(outputs, targets)
        self.log('val_loss', loss[1])
        return {'loss' : loss[1], 'batch_predictions' : batch_predictions}

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images, targets)
        batch_predictions = {
            'predictions' : [output['boxes'] for output in outputs],
            'targets' : targets,
            # 'image_ids' : ids,
        }
        loss = self.loss_fn(outputs, targets)
        self.log('test_loss', loss[1])
        return {'loss' : loss[1], 'batch_predictions' : batch_predictions}
