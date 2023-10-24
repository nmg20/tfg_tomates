import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
import numpy as np

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights

from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import box_iou, sigmoid_focal_loss, boxes as box_ops
from ensemble_boxes import ensemble_boxes_wbf

import Visualize

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium') 

models_dir = "./pths/"

# def list_to_tensor(l):
#     return torch.tensor(np.array(l))

# def tensor_to_array(t):
#     return t.detach().cpu().numpy()

def compute_loss(predictions, targets):
    losses = []
    total = 0.
    for prediction, target in zip(predictions, targets):
        p_box, p_scores, p_labels = prediction['boxes'], prediction['scores'], prediction['labels']
        t_box, t_labels = target['boxes'], target['labels']
        iou = box_iou(t_box, p_box)
        best_iou, indexes = iou.max(dim=1)
        class_loss = sigmoid_focal_loss(p_box[indexes],t_box,reduction="sum")
        box_loss = F.l1_loss(p_box[indexes],t_box, reduction="sum")
        total += class_loss + box_loss
        losses.append((class_loss, box_loss))
    return losses, total

def resize_boxes(boxes, sizes):
    new_boxes = []
    for box in boxes:
        new_boxes.append(
            [
                box[0]/sizes[1],
                box[1]/sizes[0],
                box[2]/sizes[1],
                box[3]/sizes[0]
            ]
        )
    return new_boxes

def upsize(boxes, sizes):
    new_boxes = []
    for box in boxes:
        new_boxes.append(
            [
                box[0]*sizes[1],
                box[1]*sizes[0],
                box[2]*sizes[1],
                box[3]*sizes[0]
            ]
        )
    return np.array(new_boxes)

def run_wbf(predictions, sizes, iou_thr=0.23, skip_box_thr=0.22):
    results = []
    for prediction, size in zip(predictions, sizes):
        boxes = [(resize_boxes(prediction['boxes'], size))]
        scores = [prediction['scores'].tolist()]
        labels = [prediction['labels'].tolist()]
        boxes, scores, labels = ensemble_boxes_wbf.weighted_boxes_fusion(
            boxes, scores, labels, weights=None,
            iou_thr=iou_thr, skip_box_thr=skip_box_thr,)
        d = {
            'boxes': torch.tensor(upsize(boxes, size)),
            'scores': torch.tensor(np.array(scores)),
            'labels': torch.tensor(np.array([int(x) for x in labels]))
        }
        results.append(d)
    return results

def image_sizes(images):
    sizes = []
    for image in images:
        w, h = image.shape[1:]
        sizes.append((w,h))
    return sizes

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
        # threshold=0.2,
        iou_thr=0.3, #Threshold de IoU para considerarse la misma bounding box
        skip_box_thr=0.1, #Threshold de scores de las bounding boxes
        # nms_threshold=0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lr = lr
        self.model = retinanet_resnet50_fpn(
            weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT,
            weights_backbone = ResNet50_Weights.DEFAULT
        )
        # self.threshold = threshold
        self.iou_thr = iou_thr
        self.skip_box_thr = skip_box_thr
        # self.nms_thres = nms_threshold
        self.loss_fn = compute_loss
    
    def forward(self, images, targets=None):
        outputs = self.model(images, targets)
        if not self.model.training or targets is None:
            outputs = self.threshold_fusion(outputs, image_sizes(images))
        return outputs

    def threshold_fusion(self, outputs, sizes):
        detections = []
        for output, size in zip(outputs, sizes):
            #Paso a arrays
            boxes = output['boxes'].detach().cpu().numpy()
            scores = output['scores'].detach().cpu().numpy()
            labels = output['labels'].detach().cpu().numpy()
            #Escoger indices que pasen el umbral
            indexes = np.where(labels == 1)
            #Aplicar fusion ponderada
            boxes, scores, labels = ensemble_boxes_wbf.weighted_boxes_fusion(
                [(resize_boxes(boxes[indexes], size))],
                # [(resize_boxes(boxes, size))],
                [scores[indexes].tolist()],
                # [scores.tolist()],
                [labels[indexes].tolist()],
                # [labels.tolist()],
                iou_thr=self.iou_thr, skip_box_thr=self.skip_box_thr,)
            detections.append({
                'boxes': torch.tensor(upsize(boxes, size)),
                'scores': torch.tensor(np.array(scores)),
                'labels': torch.tensor(np.array([int(x) for x in labels]))
            })
        return detections

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
        self.log('val_loss', loss[1])
        # self.log('val_class_loss', loss[0][0])
        # self.log('val_box_loss', loss[0][1])
        return {'loss' : loss[1], 'batch_predictions' : batch_predictions}

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
        self.log('test_loss', loss[1])
        # self.log('test_class_loss', loss[0][0])
        # self.log('test_box_loss', loss[0][1])
        return {'loss' : loss[1], 'batch_predictions' : batch_predictions}
    
    @torch.no_grad()
    def predict(self, batch, batch_idx):
        #Asumimos que estas imágenes residen en un dataloader (test)
        images, targets, ids = batch
        outputs = self(images, targets)
        loss = self.loss_fn(outputs, targets)
        Visualize.inference(images, outputs, targets, loss[0])