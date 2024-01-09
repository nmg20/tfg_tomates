import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection.anchor_utils import AnchorGenerator

from modelos.utils import image_sizes, compute_loss, threshold_fusion

from objdetecteval.metrics.coco_metrics import get_coco_stats
from objdetecteval.metrics.image_metrics import get_inference_metrics

from fastcore.basics import patch

import sys
sys.path.append("..")
import config

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
        self.log('test_class_loss', loss['class'])
        self.log('test_box_loss', loss['box'])
        for k in config.KEYS:
            self.log("test_"+k, mean_ap[k], logger=True)
        return {'loss' : loss['total'], 'batch_predictions' : batch_predictions}

@patch
def add_pred_outputs(self : FasterRCNNLightning, outputs):
    boxes, scores, labels, image_ids, targets = [],[],[],[],[]
    for i in range(len(outputs['batch_predictions']['predictions'])):
        preds = outputs['batch_predictions']['predictions'][i]
        boxes.append(preds['boxes'])
        scores.append(preds['scores'])
        labels.append(preds['labels'])
        image_ids.append(outputs['batch_predictions']['image_ids'][i])
        targets.append(outputs['batch_predictions']['targets'][i])

    return (labels, image_ids, boxes, scores, targets)

@patch
def on_validation_epoch_end(self : FasterRCNNLightning, outputs):
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
    stats = get_coco_stats(
        prediction_image_ids = image_ids,
        predicted_class_confidences = scores,
        predicted_bboxes = boxes,
        predicted_class_labels = labels,
        target_image_ids = truth_ids,
        target_bboxes = truth_boxes,
        target_class_labels = truth_labels,
    )['All']
    for k in stats.keys():
        self.log(k, stats[k], on_step=False, on_epoch=True, logger=True)
    return {'val_loss': outputs['loss'], 'metrics': stats}