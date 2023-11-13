import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss as CE
from ensemble_boxes import ensemble_boxes_wbf
import os

import numpy as np
from torchvision.ops import box_iou, complete_box_iou_loss as iou_loss
from torchvision.ops import sigmoid_focal_loss, boxes as box_ops, complete_box_iou_loss as iou_loss
from torchmetrics import Precision, Recall, Accuracy

# os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py")
# os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py")
# os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py")
# os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py")
# os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py")

# def compute_loss(predictions, targets):
#     """
#     box_iou([N,4],[M,4]) -> NxM
#     indexes = box_iou(b1,b2).max(dim=1)
#     """
#     losses = []
#     total = 0.
#     for prediction, target in zip(predictions, targets):
#         p_box, p_scores, p_labels = prediction['boxes'], prediction['scores'], prediction['labels']
#         t_box, t_labels = target['boxes'], target['labels']
#         iou = box_iou(t_box, p_box)
#         best_iou, indexes = iou.max(dim=1)
#         # class_loss = sigmoid_focal_loss(p_box[indexes],t_box,reduction="sum")
#         class_loss = sigmoid_focal_loss(p_box[indexes],t_box,reduction="sum")
#         box_loss = F.l1_loss(p_box[indexes],t_box, reduction="sum")
#         total += class_loss + box_loss
#         losses.append((class_loss, box_loss))
#     return losses, total

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

def threshold_fusion(outputs, sizes, iou_thr, skip_box_thr):
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
            iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        detections.append({
            'boxes': torch.tensor(upsize_boxes(boxes, size)),
            'scores': torch.tensor(np.array(scores)),
            'labels': torch.tensor(np.array([int(x) for x in labels]))
        })
    return detections

def compute_single_loss(boxes1, boxes2, labels1, labels2):
    """
    Función para calcular los dos tipos de loss para una predicción.
    Usamos CrossEntropyLoss para el error de clase y
    Complete_Box_IoU_Loss para el error de regresión de las bboxes.
    """
    _, indices = box_iou(boxes2, boxes1).max(dim=1)
    # class_loss = CE(labels1.float(), labels2[indices].float())
    box_loss = iou_loss(boxes1, boxes2)
    # return class_loss, box_loss
    return box_loss

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
    for detection, target in zip(detections, targets):
        p_boxes, t_boxes = detection['boxes'], target['boxes']
        p_labels, t_labels = detection['labels'], target['labels']
        class_loss, box_loss = compute_single_loss(
            p_boxes, t_boxes,
            p_labels, t_labels
        )
        total_loss += total_loss + class_loss + box_loss
    return total_loss

def get_metrics(detection, target):
    """
    Devuelve el cálculo de las métricas de Precisión, Recall y Accuracy
    para una detección.
    """
    _, indices = box_iou(target, detection).max(dim=1)
    metrics = {
        'precision': Precision(detection, target),
        'recall': Recall(detection, target),
        'accuracy': Accuracy(detection, target)
    }

# def compute_metrics(detections, targets):
#     """
#     Calcula la precisión y el recall de una predicción(detections)
#     en base a unos ground truths(targets).
#     """
