from pathlib import Path
import pandas as pd
from EffDetDataset import *
from Model import *
from utils.Visualize import uniquify_dir, draw_images
import os
import argparse
from torch.nn import CrossEntropyLoss as CE
from torchvision.transforms import Compose, ToPILImage, PILToTensor

preds_dir = "./preds/"
images_dir = "../../datasets/Tomato_1280x720/JPEGImages/"
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

# def read_imageset_names(ds="d801010",file="test.txt"):
#     """
#     Lee del set de imágenes de un dataset los nombres para emular que 
#     se encuentran en la carpeta /data.
#     """
#     file = open(imagesets_dir+ds+f"/{file}")
#     names = [x+".jpg" for x in file.read().split("\n")[::-1]]
#     return names

def imageset_to_pil(ds="801010",name="test.txt"):
    names = read_imageset_names(ds)
    images = [Image.open(x) for x in os.listdir(images_dir) if x in names]
    return images

# def inference_step(model, batch, output, num, compare):
def inference_step(model, batch, output, num):
    image, anns, _, _ = batch
    pred_bboxes, pred_cls, pred_confs, loss = model.predict(image)
    image, anns = denormalize(image), anns['bbox'][0].numpy()
    draw_images(image, pred_bboxes, pred_confs, loss, f"{output}/predicted_img_{num}.png",anns)

# def inference(name, compare):
def inference_dm(name, dm):
    model = load_model(name)
    output=Path(uniquify_dir(output_dir+"/run"))
    os.mkdir(output)
    # dm = get_dm_standalone(data_file=data_dir)
    dl = iter(dm.pred_dataloader())
    num = 0
    model.eval()
    try:
        while True:
            batch = next(dl)
            inference_step(model, batch, output, num)
            num = num + 1
    except StopIteration:
        pass
    finally:
        del dl

def inferencev2(name="d801010",output_dir=output_dir):
    """
    Versión con una ruta a un ImageSet en lugar de leer los nombres
    de los ficheros en el directorio /data.
    """
    model = load_model(name)
    output=Path(uniquify_dir(output_dir+"/run"))
    os.mkdir(output)
    data_ds = get_data_ds(read_imageset_names(name))
    model.eval()
    images, anots = [i for i,_,_,_ in data_ds.get_imgs_and_anots()],[i for _,i,_,_ in data_ds.get_imgs_and_anots()]
    bboxes,_,confs, loss = model.predict(images)
    bboxes = simplify_bboxes(bboxes)
    save_hist(bboxes,"areas_histogram.png",True)
    save_hist(confs,"areas_histogram.png")
    # criterion = CE()
    for image, bbox, conf in zip(images, bboxes, confs):
        name = str(output)+"/"+image.filename.split("/")[-1].split(".")[0]
        # save_hist([bbox_area(x) for x in bbox],name+"_areas_hist.png")
        # save_hist(conf, name+"_confs_hist.png")
        draw_image(image, bbox, conf, name)

def inference(model, file):
    """
    File -> leer nombres de imágenes de un archivo
        -> en su defecto leer de ./data/
    """
    if file in os.listdir(imagesets_dir+model+"/"):
        dm = get_dm_standalone(name=model,data_file=file)
    else:
        dm = get_dm_standalone(name=model,data_file=data_dir)
    inference_dm(model,dm)