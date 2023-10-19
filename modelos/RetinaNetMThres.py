import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
import numpy as np

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights

from torchvision.ops import box_iou, sigmoid_focal_loss, boxes as box_ops

import Visualize

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium') 

models_dir = "./pths/"
    
def retinanet_loss(predictions, targets, reduction="sum"):
    """
    Dados dos listas con diccionarios a tensores de bboxes, scores y labels respectivamente
    obtiene los mejores iou de las predicciones y las anotaciones (targets)
    y aplica sigmoid_focal_loss a cada predicción y las suma.
    """
    loss = 0.0
    for prediction, target in zip(predictions, targets):
        p_box, p_scores, p_labels = prediction['boxes'], prediction['scores'], prediction['labels']
        t_box, t_labels = target['boxes'], target['labels']
        # Tomamos las coincidencias de las predicciones en base a los targets
        iou = box_iou(p_box, t_box)
        best_iou, best_targets = iou.max(dim=1)
        loss += sigmoid_focal_loss(p_box, t_box[best_targets], reduction=reduction)
        # loss += nn.SmoothL1Loss(p_box[best_preds], t_box)
    return loss

class RetinaMThresTomatoLightning(LightningModule):
    """
    Clase que contiene el funcionamiento básico de un modelo a efectos de entrenamiento/validación/test.
    Crea un modelo en concreto en base al backbone específicado.
    Versión del modelo simple sin umbralización.
    """
    def __init__(
        self,
        num_classes=1,
        lr=0.0002,
        threshold=0.23,
        # nms_threshold=0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lr = lr
        self.model = retinanet_resnet50_fpn(
            weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT,
            weights_backbone = ResNet50_Weights.DEFAULT
        )
        self.threshold = threshold
        # self.nms_thres = nms_threshold
        self.loss_fn = retinanet_loss

    def forward(self, images, targets=None):
        outputs = self.model(images, targets)
        if not self.model.training or targets is None:
            outputs = self.threshold_dets(outputs)
        return outputs

    def threshold_dets(self, detections):
        """
        Filtra los resultados del modelo umbralizándolos en base
        a la variable 'threshold' del mismo. 
        """
        thresholded_detections = []
        for detection in detections:
            boxes, scores, labels = detection['boxes'], detection['scores'], detection['labels']
            indexes = np.where(scores.detach().cpu().numpy() > self.threshold)
            # indexes = boxes.batched_nms(boxes, scores, labels, self.nms_thres)
            thresholded_detections.append(
                {
                    'boxes': boxes[indexes],
                    'scores': scores[indexes],
                    'labels': labels[indexes],
                }
            )
        return thresholded_detections

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        images, targets, ids = batch
        loss = self(images, targets)
        # Registramos el error de clasificación y regresión de bbox
        self.log('train_loss', loss['bbox_regression'].detach())
        return {'loss' : loss['bbox_regression']}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, targets, ids = batch
        outputs = self(images, targets)
        batch_predictions = {
            'predictions' : [output['boxes'] for output in outputs],
            'targets' : targets,
            'image_ids' : ids,
        }
        loss = self.loss_fn(outputs, targets)
        self.log('val_loss', loss)
        return {'loss' : loss, 'batch_predictions' : batch_predictions}

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        images, targets, ids = batch
        outputs = self(images, targets)
        batch_predictions = {
            'predictions' : [output['boxes'] for output in outputs],
            'targets' : targets,
            'image_ids' : ids,
        }
        loss = self.loss_fn(outputs, targets)
        self.log('test_loss', loss)
        return {'loss' : loss, 'batch_predictions' : batch_predictions}
    
    @torch.no_grad()
    def predict(self, batch, batch_idx):
        #Asumimos que estas imágenes residen en un dataloader (test)
        images, targets, ids = batch
        outputs = self(images, targets)
        # loss = self.loss_fn(outputs, targets)
        Visualize.compare(
            images,
            [o['boxes'] for o in outputs],
            [t['boxes'] for t in targets])

