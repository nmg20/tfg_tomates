import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
import numpy as np

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection import fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights

from torchvision.ops import box_iou, sigmoid_focal_loss, boxes as box_ops
from ensemble_boxes import ensemble_boxes_wbf

import Visualize

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium') 

models_dir = "./pths/"

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

def upsize_boxes(boxes, sizes):
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

def image_sizes(images):
    sizes = []
    for image in images:
        w, h = image.shape[1:]
        sizes.append((w,h))
    return sizes

class FCOSTomatoLightning(LightningModule):
    def __init__(
        self,
        num_classes=1,
        lr=0.0002,
        threshold=0.1, #Threshold de scores de las bounding boxes
        iou_thr=0.3, #Threshold de IoU para considerarse la misma bounding box
    ):
        super().__init__()
        self.lr = lr
        self.model = fcos_resnet50_fpn(
            weights = FCOS_ResNet50_FPN_Weights.DEFAULT,
            weights_backbone = ResNet50_Weights.IMAGENET1K_V1
        )
        self.threshold = threshold
        self.iou_thr = iou_thr
        self.loss_fn = compute_loss
        self.model.num_classes = num_classes
        
    def forward(self, images, targets=None):
        outputs = self.model(images, targets)
        if not self.model.training or targets is None:
            outputs = self.threshold_fusion(outputs, image_sizes(images))
        return outputs

    def threshold_fusion(self, outputs, sizes):
        #Dados los resultados del modelo, los divide en bboxes, scores y labels,
        #aplica wbf con umbralización, los convierte otra vez a tensores y los devuelve
        detections = []
        for output, size in zip(outputs, sizes):
            #Paso a arrays
            boxes = output['boxes'].detach().cpu().numpy()
            scores = output['scores'].detach().cpu().numpy()
            labels = output['labels'].detach().cpu().numpy()
            # indexes = np.where(labels == 1)
            boxes, scores, labels = ensemble_boxes_wbf.weighted_boxes_fusion(
                [(resize_boxes(boxes, size))],
                [scores.tolist()],
                [labels.tolist()],
                iou_thr=self.iou_thr, skip_box_thr=self.threshold)
            detections.append({
                'boxes': torch.tensor(upsize_boxes(boxes, size)).to(torch.device('cuda')),
                'scores': torch.tensor(np.array(scores)).to(torch.device('cuda')),
                'labels': torch.tensor(np.array([int(x) for x in labels])).to(torch.device('cuda'))
            })
        return detections

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        images, targets, ids = batch
        loss = self(images, targets)
        # Registramos el error de clasificación y regresión de bbox
        self.log('train_class_loss', loss['classification'].detach())
        self.log('train_box_loss', loss['bbox_regression'].detach())
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
        return {'loss' : loss[1], 'batch_predictions' : batch_predictions}
    
    @torch.no_grad()
    def predict(self, batch, batch_idx):
        #Asumimos que estas imágenes residen en un dataloader (test)
        images, targets, ids = batch
        outputs = self(images, targets)
        loss = self.loss_fn(outputs, targets)
        Visualize.inference(images, outputs, targets, loss[0])