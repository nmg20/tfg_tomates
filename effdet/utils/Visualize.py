import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from PIL import Image
import os
import cv2 as cv

colors = ["red","orange","yellow","green","blue","purple"]

def get_rectangle_edges_from_pascal_bbox(bbox):
    xmin_top_left, ymin_top_left, xmax_bottom_right, ymax_bottom_right = bbox

    bottom_left = (xmin_top_left, ymax_bottom_right)
    width = xmax_bottom_right - xmin_top_left
    height = ymin_top_left - ymax_bottom_right

    return bottom_left, width, height

def draw_pascal_voc_bboxes(
    plot_ax,
    bboxes,
    get_rectangle_corners_fn=get_rectangle_edges_from_pascal_bbox,
):
    for bbox in bboxes:
        bottom_left, width, height = get_rectangle_corners_fn(bbox)

        plot_ax.add_patch(patches.Rectangle(
            bottom_left,
            width,
            height,
            linewidth=1,
            edgecolor="orange",
            fill=False,
        ))

def draw_bboxes(image, bboxes):
    img = image.copy()
    for bbox in bboxes:
        cv.rectangle(img,(bbox[0],bbox[1]),(bbox[2],bbox[3]),
            "green", 1)
    return img

def get_img_drawn(image, bboxes_anot, predicted_bboxes, loss, size=20):
    """
        image = imagen del dataset a predecir/dibujar
        bboxes_anot = anotaciones originales de la imagen en el dataset
        predicted_bboxes = anotaciones predecidas por el modelo
    """
    # plt.figure()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(size,size))
    # fig.suptitle(title,fontsize=size*(3/2))
    ax1.imshow(image)
    ax1.set_title(f"Imagen predecida\nLoss={loss}",fontsize=size*(5/4))
    ax2.imshow(image)
    ax2.set_title("Imagen anotada",fontsize=size*(5/4))
    draw_pascal_voc_bboxes(ax1, predicted_bboxes)
    draw_pascal_voc_bboxes(ax2, bboxes_anot.tolist())
    fig.canvas.draw()
    image = Image.frombytes('RGB', 
        fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
    plt.close()
    return image

def draw_img(image, bboxes):
    # plt.figure()
    fig, ax = plt.subplots()
    # for bbox in bboxes:
    #     bb = np.array(bbox,dtype=np.float32)
    #     rect = plt.Rectangle((bb[1], bb[0]), bb[3]-bb[1], bb[2]-bb[0], color='yellow',
    #                      fill=False, lw=3)
    #     plt.gca().add_patch(rect)
    ax.imshow(image)
    draw_pascal_voc_bboxes(ax,bboxes)
    fig.canvas.draw()
    image = Image.frombytes('RGB', 
        fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
    plt.close()
    return image

def draw_img(image, gt, bboxes, losses):
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(image)
    # ax1.set_title(f"loss: {losses[0]}, box_loss: {losses[1]}")
    ax2.imshow(image)
    ax2.set_title("Ground Truth (imagen anotada)")
    draw_pascal_voc_bboxes(ax1,bboxes)
    draw_pascal_voc_bboxes(ax2,gt)
    fig.canvas.draw()
    image = Image.frombytes('RGB', 
        fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
    plt.close()
    return image

# def draw_multipred(image, gt, bboxes, losses):
#     """
#     image: imagen a dibujar
#     gt: bboxes de la imagen anotada (ground truth)
#     bboxes: bboxes de la imagen predecida
#     losses: lista de dicts con las loss y box_loss de cada pred
#     """
#     fig, (ax1, ax2) = plt.subplots(1,2)
#     ax1.imshow(image)


def set_fig(image, gts):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image)
    ax2.imshow(image)
    draw_pascal_voc_bboxes(ax2,gts)
    return fig, ax1, ax2

def draw_bbox(bboxes, ax):
    draw_pascal_voc_bboxes(ax,bboxes)

def set_loss(ax,loss):
    ax.set_title(f"Loss: {loss[0]}, Box_loss: {loss[1]}")

# def draw_img_mod(image,gts,bboxes,losses,color="orange"):
#     fig, ax1, ax2 = set_fig(image,gts)
#     ax1.set_xticks([])
#     ax1.set_yticks([])
#     ax2.set_xticks([])
#     ax2.set_yticks([])
#     draw_bbox(bboxes,ax1,color)
#     set_loss(ax1,losses)
#     fig.canvas.draw()
#     image = Image.frombytes('RGB', 
#         fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
#     return image

def uniquify_name(name):
    """
    name: ruta_del_dir+nombre -> sin números ni .jpg/.png
    """
    filename = name+"{}.jpg"
    counter = 1
    while os.path.isfile(filename.format(counter)):
        counter += 1
    return filename.format(counter)

def uniquify_dir(name):
    dirname = name+"{}"
    counter = 1
    while os.path.isdir(dirname.format(counter)):
        counter += 1
    return dirname.format(counter)

def draw_img_mod(image,gts,bboxes,losses,name):
    """
    image : PIL Image
    gts : array de bboxes anotadas
    bboxes : lista de predicciones
    losses : lista [loss, box_loss]
    """
    fig, ax1, ax2 = set_fig(image,gts)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    draw_bbox(bboxes,ax1)
    set_loss(ax1,losses)
    plt.figsave(uniquify(name))

def draw_imgs(images,gts,bboxes,losses,names):
    for i, image in enumerate(images):
        drawn_image=draw_img_mod(image,gts[i],bboxes[i],losses[i],names[i])

def show_image(
    image, bboxes=None, draw_bboxes_fn=draw_pascal_voc_bboxes, figsize=(10, 10)
):
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)

    if bboxes is not None:
        draw_bboxes_fn(ax, bboxes)

    plt.show()