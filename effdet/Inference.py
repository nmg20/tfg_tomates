from pathlib import Path
import pandas as pd
from EffDetDataset import *
from Model import *
from utils.Visualize import uniquify_dir, draw_image
import os
import argparse
from torch.nn import CrossEntropyLoss as CE
from torchvision.transforms import Compose, ToPILImage, PILToTensor

preds_dir = "./preds/"
imagesets_dir = "../../datasets/Tomato_1280x720/ImageSets/"
data_dir = os.path.abspath("./data/")
output_dir = os.path.abspath("./outputs/")

############################
"""
Proceso de inferencia (nuevo):
    - Dado un directorio, crear un dm con las imágenes dentro
    - Leer anotaciones dados los nombres en ../../datasets/Tomato_1280x720/ImageSets/all_annotations.csv
    - Hacer la inferencia, calcular loss, registrar confianzas
    - Volcar a otro directorio /outputs:
        -> imágenes predecidas
        -> informe con las losses
        -> histograma de confianzas
"""

def get_dir_imgs_names(imgs_dir=data_dir):
    return [x for x in os.listdir(imgs_dir) if x[len(x)-4::]=='.jpg']

# def intersect(box_a, box_b):
#     """ We resize both tensors to [A,B,2] without new malloc:
#     [A,2] -> [A,1,2] -> [A,B,2]
#     [B,2] -> [1,B,2] -> [A,B,2]
#     Then we compute the area of intersect between box_a and box_b.
#     Args:
#       box_a: (tensor) bounding boxes, Shape: [A,4].
#       box_b: (tensor) bounding boxes, Shape: [B,4].
#     Return:
#       (tensor) intersection area, Shape: [A,B].
#     """
#     A = box_a.size(0)
#     B = box_b.size(0)
#     max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
#                        box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
#     min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
#                        box_b[:, :2].unsqueeze(0).expand(A, B, 2))
#     inter = torch.clamp((max_xy - min_xy), min=0)
#     return inter[:, :, 0] * inter[:, :, 1]

# def jaccard(box_a, box_b):
#     """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
#     is simply the intersection over union of two boxes.  Here we operate on
#     ground truth boxes and default boxes.
#     E.g.:
#         A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
#     Args:
#         box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
#         box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
#     Return:
#         jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
#     """
#     inter = intersect(box_a, box_b)
#     area_a = ((box_a[:, 2]-box_a[:, 0]) *
#               (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
#     area_b = ((box_b[:, 2]-box_b[:, 0]) *
#               (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
#     union = area_a + area_b - inter
#     return inter / union  # [A,B]

def file_to_bboxes(file):
    f = open(file,"r")
    return [[float(x) for x in bbox.split(" ") if len(x)>1] for bbox in f.read().strip().replace("[","").replace("]","").split("\n")]

def bbox_area(bbox):
    return abs((bbox[2]-bbox[0])*(bbox[3]-bbox[1]))

def save_hist(data, name, nbins, bboxes=False):
    """
    Recibe una lista anidada (confianzas de las predicciones o los
    bounding boxes), la aplana, dibuja su histograma y lo guarda.
    """
    # data = [item for sublist in data for item in sublist]
    if bboxes==True:
        data = [bbox_area(x) for x in data]
    fig, ax = plt.subplots(figsize=(20,20))
    if not nbins:
        bins=np.linspace(min(data),max(data))
    else:
        bins = list(range(0,max(data),nbins))
    plt.hist(data, bins)
    plt.savefig(name)
    plt.close()

def simplify_bboxes(bboxes):
    return [[[round(x,2) for x in bbox] for bbox in bboxs] for bboxs in bboxes]

def read_imageset_names(ds="d801010"):
    """
    Lee del set de imágenes de un dataset los nombres para emular que 
    se encuentran en la carpeta /data.
    """
    file = open(imagesets_dir+ds+"/test.txt")
    names = [x+".jpg" for x in file.read().split("\n")[::-1]]
    return names[0:44]

def calculate_loss(anots,bboxes):
    losses = []
    anots, bboxes = [torch.tensor(x) for x in anots], [torch.tensor(x) for x in bboxes]
    for i in len(anots):
        losses.append()


def inferencev1(model,data_dir=data_dir,output_dir=output_dir):
    """
    A partir de un directorio crea un ds ad-hoc, lee de un fichero maestro
    las anotaciones, infiere los resultados y os vuelca en un directorio "output_dir".
    Las imágenes necesitan pertenecer a la base de datos de imágenes del modelo,
    sino no se podría calcular la loss de las predicciones.
    """
    # Output_dir con un dir /runx (x=número único -> uniquify)
    output=Path(uniquify_dir(output_dir+"/run"))
    os.mkdir(output)
    data_ds = get_data_ds(get_dir_imgs_names())
    model.eval()
    images, anots = [i for i,_,_,_ in data_ds.get_imgs_and_anots()],[i for _,i,_,_ in data_ds.get_imgs_and_anots()]
    bboxes,_,confs = model.predict(images)
    # bboxes = simplify_bboxes(bboxes)
    save_hist(bboxes,str(output)+"/areas_histogram.png",True)
    save_hist(confs,str(output)+"/confidences_histogram.png",False)
    for image, bbox, conf in zip(images, bboxes, confs):
        name = str(output)+"/"+image.filename.split("/")[-1].split(".")[0]
        draw_image(image, bbox, conf, name)

def inferencev2(model,ds="d801010",output_dir=output_dir):
    """
    Versión con una ruta a un ImageSet en lugar de leer los nombres
    de los ficheros en el directorio /data.
    """
    output=Path(uniquify_dir(output_dir+"/run"))
    os.mkdir(output)
    data_ds = get_data_ds(read_imageset_names(ds))
    # data_ds = get_data_ds(read_imageset_names("d801010"))
    model.eval()
    images, anots = [i for i,_,_,_ in data_ds.get_imgs_and_anots()],[i for _,i,_,_ in data_ds.get_imgs_and_anots()]
    bboxes,_,confs, loss = model.predict(images)
    save_hist(bboxes,str(output)+f"/areas_histogram_{ds}.png",True)
    save_hist(confs,str(output)+f"/confidences_histogram_{ds}.png",False)
    for image, bbox, conf in zip(images, bboxes, losses, confs):
        name = str(output)+"/"+image.filename.split("/")[-1].split(".")[0]
        draw_image(image, bbox, conf, loss, name)
    # names = [str(output)+"/"+image.filename.split("/")[-1].split(".")[0] for image in images]
    # draw_images(images, bboxes, confs, names)

def imgs_to_tensor(images):
    tf = Compose([PILToTensor()])
    return [tf(x) for x in images]

def inf(model,ds="d801010",output_dir=output_dir):
    output=Path(uniquify_dir(output_dir+"/run"))
    os.mkdir(output)
    data_ds = get_data_ds(read_imageset_names(ds))
    model.eval()

def inference(model_name="d801010",version=2):
    if version==2:
        inferencev2(load_model(model_name),model_name)
    else:
        inferencev1(load_model(model_name))