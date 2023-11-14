import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from ensemble_boxes import ensemble_boxes_wbf
import os

import numpy as np
from torchvision.ops import box_iou, complete_box_iou_loss as iou_loss
from torchvision.ops import sigmoid_focal_loss, boxes as box_ops, complete_box_iou_loss as iou_loss
from torchmetrics import Precision, Recall, Accuracy

import sys
sys.path.append("..")
import config

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

def threshold_fusion(outputs, images, iou_thr, skip_box_thr):
    #Dados los resultados del modelo, los divide en bboxes, scores y labels,
    #aplica wbf con umbralización, los convierte otra vez a tensores y los devuelve
    detections = []
    sizes = image_sizes(images)
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
            iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        detections.append({
            'boxes': torch.tensor(upsize_boxes(boxes, size)).to(config.DEVICE),
            'scores': torch.tensor(np.array(scores)).to(config.DEVICE),
            'labels': torch.tensor(np.array([int(x) for x in labels])).to(config.DEVICE)
        })
    return detections

def compute_single_loss(boxes1, boxes2, labels1, labels2):
    """
    Función para calcular los dos tipos de loss para una predicción.
    Usamos CrossEntropyLoss para el error de clase y
    Complete_Box_IoU_Loss para el error de regresión de las bboxes.
    """
    ce = CrossEntropyLoss(reduction="mean")
    _, indices = box_iou(boxes2, boxes1).max(dim=1)
    class_loss = ce(labels1[indices].float(), labels2.float())
    box_loss = iou_loss(boxes1[indices], boxes2, reduction="mean")
    return class_loss, box_loss
    # return box_loss

def compute_loss(detections, targets):
    """
    Calcula el error total para un conjunto de predicciones(detections)
    en base a los ground truths (targets).
    detections: dict(List[Tensor[N,4]])
        - 'boxes': bounding boxes
        - 'labels': labels
    targets: dict(List[Tensor[M,4]])
        - 'boxes'
        - 'labels'
    """
    total_loss = 0
    class_losses, box_losses = 0, 0
    for detection, target in zip(detections, targets):
        p_boxes, t_boxes = detection['boxes'], target['boxes']
        p_boxes, t_boxes = p_boxes.detach().cpu(), t_boxes.detach().cpu()
        p_labels, t_labels = detection['labels'], target['labels']
        p_labels, t_labels = p_labels.detach().cpu(), t_labels.detach().cpu()
        class_loss, box_loss = compute_single_loss(
            p_boxes, t_boxes,
            p_labels, t_labels
        )
        total_loss += total_loss + class_loss + box_loss
        class_losses += class_losses + class_loss
        box_losses += box_losses + box_loss
    return {'total': total_loss, 'class': class_losses, 'box': box_losses}

# def get_metrics(detection, target):
#     """
#     Devuelve el cálculo de las métricas de Precisión, Recall y Accuracy
#     para una detección.
#     """
#     _, indices = box_iou(target, detection).max(dim=1)
#     metrics = {
#         'precision': Precision(detection, target),
#         'recall': Recall(detection, target),
#         'accuracy': Accuracy(detection, target)
#     }

# def compute_metrics(detections, targets):
#     """
#     Calcula la precisión y el recall de una predicción(detections)
#     en base a unos ground truths(targets).
#     """
