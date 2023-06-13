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
                str(confs[0][i])[0:4], fontsize=18,
            horizontalalignment='center',
            verticalalignment='top',color="orange")

def draw_img(ax,img,bboxes,confs=None):
    ax.imshow(img)
    draw_bboxes_confs(ax,bboxes[0],confs)

def draw_images(image, bboxes, confs, loss, name, annots=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,20))
    plt.title(f"Inferencia - Loss: {loss}.")
    ax1.imshow(image)
    draw_bboxes_confs(ax1,bboxes[0],confs)
    ax2.imshow(image)
    draw_bboxes_confs(ax2,annots,None)
    plt.savefig(f"{name}.png")
    plt.close(fig)

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