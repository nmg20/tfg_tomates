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

from lightning.pytorch.utilities.memory import recursive_detach

import sys
sys.path.append("..")
import config

def rescale_boxes(boxes, sizes):
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

def images_sizes(images):
    sizes = []
    for image in images:
        w, h = image.shape[1:]
        sizes.append((w,h))
    return sizes

def threshold_fusion(outputs, images_sizes, iou_thr, skip_box_thr):
    #Dados los resultados del modelo, los divide en bboxes, scores y labels,
    #aplica wbf con umbralización, los convierte otra vez a tensores y los devuelve
    detections = []
    # sizes = image_sizes(images)
    for output, size in zip(outputs, images_sizes):
        #Paso a arrays
        boxes = output['boxes']
        scores = output['scores']
        labels = output['labels']
        boxes, scores, labels = ensemble_boxes_wbf.weighted_boxes_fusion(
            [(rescale_boxes(boxes, size))],
            [scores.tolist()],
            [labels.tolist()],
            iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        detections.append({
            'boxes': torch.tensor(upsize_boxes(boxes, size)).to(config.DEVICE),
            'scores': torch.tensor(np.array(scores)).to(config.DEVICE),
            'labels': torch.tensor(np.array([int(x) for x in labels])).to(config.DEVICE)
        })
    return detections

def check_box(bboxes):
    """
    Workaround necesario para manejar resultados vacíos en las bounding boxes.
    """
    if len(bboxes)>0:
        return bboxes
    else:
        return torch.as_tensor([[0.,0.,0.,0.]],dtype=torch.float64, device=config.DEVICE)

def check_label(labels):
    """
    Workaround necesario para manejar resultados vacíos en las clases.
    """
    if len(labels)>0:
        labels = labels.type(torch.LongTensor)
        return labels.to(config.DEVICE)
    else:
        return torch.as_tensor([0.],dtype=torch.long, device=config.DEVICE)

def check_boxes(bboxes1, bboxes2):
    return check_box(bboxes1), check_box(bboxes2)

def check_labels(labels1, labels2):
    return check_label(labels1), check_label(labels2)

def compute_single_loss(boxes1, boxes2, labels1, labels2):
    """
    Función para calcular los dos tipos de loss para una predicción.
    Usamos CrossEntropyLoss para el error de clase y
    Complete_Box_IoU_Loss para el error de regresión de las bboxes.
    """
    ce = CrossEntropyLoss(reduction="mean")
    # boxes1, boxes2 = check_boxes(boxes1, boxes2)
    # labels1, labels2 = check_labels(labels1, labels2)
    _, indices = box_iou(boxes2, boxes1).max(dim=1)
    # indices.to(config.DEVICE)
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
        # p_boxes, t_boxes = p_boxes.detach(), t_boxes.detach()
        p_boxes, t_boxes = recursive_detach(p_boxes, to_cpu=True), recursive_detach(t_boxes, to_cpu=True)
        p_labels, t_labels = detection['labels'], target['labels']
        # p_labels, t_labels = p_labels.detach(), t_labels.detach()
        p_labels, t_labels = recursive_detach(p_labels, to_cpu=True), recursive_detach(t_labels, to_cpu=True)
        class_loss, box_loss = compute_single_loss(
            p_boxes, t_boxes,
            p_labels, t_labels
        )
        total_loss += total_loss + class_loss + box_loss
        class_losses += class_losses + class_loss
        box_losses += box_losses + box_loss
    return {'total': total_loss, 'class': class_losses, 'box': box_losses}