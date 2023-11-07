import torch
import torch.nn as nn
import torch.nn.functional as F
from ensemble_boxes import ensemble_boxes_wbf
import os

import numpy as np
from torchvision.ops import box_iou, sigmoid_focal_loss, boxes as box_ops

# os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py")
# os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py")
# os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py")
# os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py")
# os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py")

def compute_loss(predictions, targets):
    losses = []
    total = 0.
    for prediction, target in zip(predictions, targets):
        p_box, p_scores, p_labels = prediction['boxes'], prediction['scores'], prediction['labels']
        t_box, t_labels = target['boxes'], target['labels']
        iou = box_iou(t_box, p_box)
        best_iou, indexes = iou.max(dim=1)
        # class_loss = sigmoid_focal_loss(p_box[indexes],t_box,reduction="sum")
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