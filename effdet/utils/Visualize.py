import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import patches
import numpy as np
from PIL import Image, ImageDraw as D, ImageFont
import os
import torchvision.transforms as T

colors = ["red","orange","yellow","green","blue","purple"]

def get_rectangle_edges_from_pascal_bbox(bbox):
    xmin_top_left, ymin_top_left, xmax_bottom_right, ymax_bottom_right = bbox

    bottom_left = (xmin_top_left, ymax_bottom_right)
    width = xmax_bottom_right - xmin_top_left
    height = ymin_top_left - ymax_bottom_right

    return bottom_left, width, height

def edges(bbox):
    x1,y1,x2,y2 = bbox
    minx,miny,maxx, maxy = min(x1,x2),min(y1,y2),max(x1,x2),max(y1,y2)
    return ((minx,miny),(maxx,maxy))

def get_bbox_dim(bbox):
    xmin_top_left, ymin_top_left, xmax_bottom_right, ymax_bottom_right = bbox
    width = xmax_bottom_right - xmin_top_left
    height = ymin_top_left - ymax_bottom_right

    return width, height

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

def draw_bboxes_confs(plot_ax, bboxes, confs=None,
    get_rectangle_corners_fn=get_rectangle_edges_from_pascal_bbox,
):
    for i in range(len(bboxes)):
        bottom_left, width, height = get_rectangle_corners_fn(bboxes[i])
        (x1,y1),(x2,y2) = edges(bboxes[i])
        plot_ax.add_patch(patches.Rectangle(
            bottom_left,
            width,
            height,
            linewidth=2,
            edgecolor="orange",
            fill=False,
        ))
        if confs:
            plot_ax.text(bottom_left[0]+0.5*width,y2,
                str(confs[0][i])[0:4], fontsize=12,
            horizontalalignment='center',
            verticalalignment='top',color="orange")

def draw_img(ax,img,bboxes,confs=None):
    ax.imshow(img)
    draw_bboxes_confs(ax,bboxes[0],confs)

def draw_ax(ax, img, bboxes, confs, loss, name):
    ax.imshow(img)
    draw_bboxes_confs(ax, bboxes, confs)
    ax.title.set_text(f'name: {loss}')

def draw_subplots(image, bboxes, confs, loss, names):
    fig, 

def draw_images(image, bboxes, confs, loss, name, annots=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,20))
    fig.suptitle(f"Inferencia - Loss: {loss}.", fontsize=16)
    ax1.imshow(image)
    draw_bboxes_confs(ax1,bboxes[0],confs)
    ax2.imshow(image)
    draw_bboxes_confs(ax2,annots,None)
    plt.savefig(f"{name}.png")
    plt.close(fig)

def draw_images_stacked(image, bboxes, confs, loss, name, annots=None):
    fig, axs = plt.subplots(2, figsize=(20,20))
    fig.suptitle(f"Inferencia - Loss: {loss}.", fontsize=20)
    axs[0].imshow(image)
    axs[0].set_title("Imagen predecida")
    draw_bboxes_confs(axs[0],bboxes[0],confs)
    axs[1].imshow(image)
    axs[1].set_title("Imagen anotada")
    draw_bboxes_confs(axs[1],annots,None)
    plt.savefig(f"{name}.png")
    plt.close(fig)

def draw_losses(losses, mean, name):
    # fig = plt.plot(losses)
    plt.plot(losses)
    plt.title(f"Loss media: {mean}")
    plt.ylabel("Losses")
    plt.xlabel("Imágenes")
    plt.savefig(name+".png")
    plt.close()

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
    # os.mkdir(dirname.format(counter))
    return dirname.format(counter)