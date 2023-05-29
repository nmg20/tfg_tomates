import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import patches
import numpy as np
from PIL import Image, ImageDraw as D, ImageFont
import os

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

# def draw_pascal_voc_bboxes(
#     plot_ax,
#     bboxes,
#     get_rectangle_corners_fn=get_rectangle_edges_from_pascal_bbox,
# ):
#     for bbox in bboxes:
#         bottom_left, width, height = get_rectangle_corners_fn(bbox)

#         plot_ax.add_patch(patches.Rectangle(
#             bottom_left,
#             width,
#             height,
#             linewidth=1,
#             edgecolor="orange",
#             fill=False,
#         ))

def draw_bboxes_conf(plot_ax, bboxes, confs,
    get_rectangle_corners_fn=get_rectangle_edges_from_pascal_bbox,
):
    for bbox, conf in zip(bboxes,confs):
        bottom_left, width, height = get_rectangle_corners_fn(bbox)
        (x1,y1),(x2,y2) = edges(bbox)
        plot_ax.add_patch(patches.Rectangle(
            bottom_left,
            width,
            height,
            linewidth=2,
            edgecolor="orange",
            fill=False,
        ))
        plot_ax.text(bottom_left[0]+0.5*width,y2,
            str(conf)[0:4], fontsize=18,
        horizontalalignment='center',
        verticalalignment='top',color="orange")


# def get_img_drawn(image, bboxes_anot, predicted_bboxes, loss, size=20):
#     """
#         image = imagen del dataset a predecir/dibujar
#         bboxes_anot = anotaciones originales de la imagen en el dataset
#         predicted_bboxes = anotaciones predecidas por el modelo
#     """
#     # plt.figure()
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(size,size))
#     # fig.suptitle(title,fontsize=size*(3/2))
#     ax1.imshow(image)
#     ax1.set_title(f"Imagen predecida\nLoss={loss}",fontsize=size*(5/4))
#     ax2.imshow(image)
#     ax2.set_title("Imagen anotada",fontsize=size*(5/4))
#     draw_pascal_voc_bboxes(ax1, predicted_bboxes)
#     draw_pascal_voc_bboxes(ax2, bboxes_anot.tolist())
#     fig.canvas.draw()
#     image = Image.frombytes('RGB', 
#         fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
#     plt.close()
#     return image

def draw_image(image, bboxes, confs, loss, name):
    fig, ax = plt.subplots(figsize=(20,20))
    ax.imshow(image)
    draw_bboxes_conf(ax,bboxes,confs)
    plt.title(f"Loss: {loss}")
    plt.savefig(f"{name}.png")
    plt.close(fig)

def draw_images(images, bboxes, confs, names):
    fig, ax = plt.subplots(figsize=(20,20))
    for image, bboxs, conf, names in zip(images, bboxes, confs, names):
        draw_bboxes_conf(ax,bboxes,confs)
        plt.savefig(f"{name}.png")
    plt.close()

# def set_fig(image, gts):
#     fig, (ax1, ax2) = plt.subplots(1, 2)
#     ax1.imshow(image)
#     ax2.imshow(image)
#     draw_pascal_voc_bboxes(ax2,gts)
#     return fig, ax1, ax2

# def draw_bbox(bboxes, ax):
#     draw_pascal_voc_bboxes(ax,bboxes)

def set_loss(ax,loss):
    ax.set_title(f"Loss: {loss[0]}, Box_loss: {loss[1]}")

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

# def draw_img_mod(image,gts,bboxes,losses,name):
#     """
#     image : PIL Image
#     gts : array de bboxes anotadas
#     bboxes : lista de predicciones
#     losses : lista [loss, box_loss]
#     """
#     fig, ax1, ax2 = set_fig(image,gts)
#     ax1.set_xticks([])
#     ax1.set_yticks([])
#     ax2.set_xticks([])
#     ax2.set_yticks([])
#     draw_bbox(bboxes,ax1)
#     set_loss(ax1,losses)
#     plt.figsave(uniquify(name))

# def draw_imgs(images,gts,bboxes,losses,names):
#     for i, image in enumerate(images):
#         drawn_image=draw_img_mod(image,gts[i],bboxes[i],losses[i],names[i])

# def show_image(
#     image, bboxes=None, draw_bboxes_fn=draw_pascal_voc_bboxes, figsize=(10, 10)
# ):
#     fig, ax = plt.subplots(1, figsize=figsize)
#     ax.imshow(image)

#     if bboxes is not None:
#         draw_bboxes_fn(ax, bboxes)

#     plt.show()