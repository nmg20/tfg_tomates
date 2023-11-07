import torch
import torchvision.transforms as tfs

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from PIL import Image, ImageDraw as D, ImageFont
import os

from torchvision.transforms.functional import convert_image_dtype
from torchvision.utils import save_image#, draw_bounding_boxes

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

good_label="blue"
bad_label="red"

def denormalize(tensor):
    z = tensor * torch.tensor(std).view(3,1,1)
    z = z + torch.tensor(mean).view(3,1,1)
    # return tfs.ToPILImage(mode="RGB")(z.squeeze(0))
    return z

inverse = tfs.Normalize(
    mean = [-m/s for m, s in zip(mean, std)],
    std = [1/s for s in std]
)

def get_rectangle_edges(bbox):
    xmin_top_left, ymin_top_left, xmax_bottom_right, ymax_bottom_right = bbox

    bottom_left = (xmin_top_left, ymax_bottom_right)
    width = xmax_bottom_right - xmin_top_left
    height = ymin_top_left - ymax_bottom_right

    return bottom_left, width, height

def draw_bboxes(ax, bboxes, labels=None, linewidth=1.5, color="orange"):
    """
    Dado un axis y un conjunto de bounding boxes (numpy.array), las dibuja
    """
    for bbox in bboxes:
        bl, w, h = get_rectangle_edges(bbox)
        ax.add_patch(patches.Rectangle(
            bl, w, h,
            linewidth=linewidth,
            edgecolor=color, 
            fill=False,
            )
        )

def label_color(label):
    if label==1:
        return "red"
    else:
        return "blue"

def draw_bboxes_labels(ax, bboxes, labels, linewidth=1.5):
    """
    Dado un axis y un conjunto de bounding boxes (numpy.array), las dibuja
    """
    for bbox, label in zip(bboxes, labels):
        bl, w, h = get_rectangle_edges(bbox)
        color = label_color(label)
        ax.add_patch(patches.Rectangle(
            bl, w, h,
            linewidth=linewidth,
            edgecolor=color, 
            fill=False,
            )
        )

def show_image_tensor(tensor):
    """
    Desaplica la normalización hecha para convertir la imagen en 
    tensor y la muestra en un plot.
    """
    denormalized_tensor = denormalize(tensor)
    plt.imshow(denormalized_tensor.permute(1,2,0))
    plt.show()

def show_bboxes(image : torch.Tensor, bboxes : torch.Tensor, 
        labels=None, linewidth=2,color="orange"):
    """
    Plasma la imagen desnormalizada en un axis y dibuja por encima
    las bounding boxes -> labels opcionales (scores).
    -> seleccionables tamaños de línea y colores.
    """
    fig, ax = plt.subplots(1, figsize=(10,10))
    image = denormalize(image)
    ax.imshow(image.permute(1,2,0))
    draw_bboxes(ax,bboxes.detach().numpy(),labels,linewidth,color)
    plt.show()

def compare_preds(image, bboxes, targets, labels, loss, colors=["orange","red"]):
    """
    Dibuja una imagen en un eje y plasma sobre la misma dos conjuntos
    de bounding boxes (tensores).
    """
    # fig, ax = plt.subplots(1, figsize=(15,15))
    fig, ax = plt.subplots(1)
    fig.suptitle(f"Loss: {loss}.", fontsize=16)
    image = denormalize(image)
    ax.imshow(image.permute(1,2,0))
    draw_bboxes(ax,targets.detach().numpy(),labels,linewidth=2,color=colors[0])
    draw_bboxes_labels(ax,bboxes.detach().numpy(),labels,linewidth=1)
    plt.show()

def bbox_size(bbox):
    return (bbox[2]-bbox[0])*(bbox[3]-bbox[1])

def max_size(bboxes):
    sizes = [bbox_size(bbox) for bbox in bboxes]
    return np.max(sizes)

def equalize(bboxes, targets, ratio=5.0):
    """
    bboxes = tensor de bboxes
    targets = ~
    """
    equalized = []
    max_value = ratio*max_size(targets)
    for bbox, target in zip(bboxes, targets):
        if bbox_size(bbox)<=max_value:
            equalized.append(bbox)
    return torch.stack(equalized)

def compare(images, bboxess, targetss, labelss, losses):
    # bboxess = [equalize(bbs, tgs) for bbs, tgs in zip(bboxess, targetss)]
    for image, bboxes, targets, labels, loss in zip(images, bboxess, targetss, labelss, losses):
        compare_preds(image, bboxes, targets, labels, loss)

def compare_outputs(images, outputs, targets, loss):
    compare(
        images,
        [o['boxes'] for o in outputs],
        [t['boxes'] for t in targets],
        [o['labels'] for o in outputs],
        loss[0]
        # loss
    )
