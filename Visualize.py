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
    for bbox, label in zip(bboxes, labels):
        bl, w, h = get_rectangle_edges(bbox)
        ax.add_patch(patches.Rectangle(
            bl, w, h,
            linewidth=linewidth,
            edgecolor=color, 
            fill=False,
            )
        )

def sort_by_labels(boxes, labels):
    """
    Ordena de forma inversa las bounding boxes y los labels.
        -> para dibujar los 1s de últimos
    """
    s = [(x,y) for x,y in sorted(zip(boxes, labels), key= lambda k:k[1], reverse=True)]
    bs, ls = [], []
    for box, label in s:
        bs.append(box)
        ls.append(label)
    boxes, labels = np.stack(bs), np.stack(ls)
    return boxes, labels

def draw_bboxes_labels(ax, bboxes, labels=None, confs=None, linewidth=1.5):
    """
    Dado un axis y un conjunto de bounding boxes (numpy.array), las dibuja
    Bbox en rojo = bbox bien etiquetada.
    """
    bboxes, labels = sort_by_labels(bboxes, labels)
    for bbox, label, conf in zip(bboxes, labels, confs):
        bl, w, h = get_rectangle_edges(bbox)
        color = "blue" if label==1 else "red"
        ax.add_patch(patches.Rectangle(
            bl, w, h,
            linewidth=linewidth,
            edgecolor=color, 
            fill=False,
            )
        )
        if conf:
            plt.text(
                bbox[0], bbox[1],
                s = str(int(100*conf)) + "%",
                color = "white",
                verticalalignment = "top",
                bbox = {"color": color, "pad":0},
                fontsize=8
            )

def show_image_tensor(tensor):
    """
    Desaplica la normalización hecha para convertir la imagen en 
    tensor y la muestra en un plot.
    """
    denormalized_tensor = denormalize(tensor)
    plt.imshow(denormalized_tensor.permute(1,2,0))
    plt.show()

def imshow_tensor(tensor):
    tensor = tensor/2 + 0.5
    npimg = tensor.numpy()
    # denormalized_tensor = denormalize(tensor)
    plt.imshow(np.transpose(npimg,(1,2,0)))

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

def compare_preds(image, bboxes, targets, labels, scores, loss, colors=['orange', 'red']):
    """
    Dibuja una imagen en un eje y plasma sobre la misma dos conjuntos
    de bounding boxes (tensores).
    """
    fig, ax = plt.subplots(1)
    cl = "{:.2e}".format(float(loss[0].detach().cpu().item()))
    bl = "{:.2e}".format(float(loss[1].detach().cpu().item()))
    fig.suptitle(f"Class loss: {cl}  Box loss: {bl}.", fontsize=16)
    image = denormalize(image)
    ax.imshow(image.permute(1,2,0))
    draw_bboxes(
        ax,
        targets.detach().cpu().numpy(),
        labels.detach().cpu().numpy(),
        linewidth=2,
        color=colors[0]
    )
    draw_bboxes_labels(
        ax,
        bboxes.detach().cpu().numpy(),
        labels.detach().cpu().numpy(),
        scores.detach().cpu().numpy(),
    )
    plt.show()

def compare_outputs(images, detections, targets, labels, scores, losses):
    for i in range(len(images)):
        compare_preds(
            images[i],
            detections[i],
            targets[i],
            labels[i],
            scores[i],
            losses[i]
        )