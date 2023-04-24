from pathlib import Path
import pandas as pd
from EffDetDataset import *
from Model import *
from utils.Visualize import uniquify_dir, draw_image
import os
import argparse
from torch.nn import CrossEntropyLoss as CE

preds_dir = "./preds/"
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

# def get_dir_imgs(imgs_dir=data_dir):
#     return [Image.open(imgs_dir+"/"+x) for x in os.listdir(imgs_dir)]

# def save_preds(imgs,bboxes,output_dir):
#     """
#     Las ímágenes deben haber sido abiertas con PIL.Image para poder
#     leer el nombre del fichero.
#     """
#     for i in range(len(imgs)):
#         img = draw_img(imgs[i],bboxes[i])
#         img.save(f"{output_dir}/pred_{imgs[i].filename.split('/')[-1]}")
#         # img.save(f"{Path(output_dir)}/anot_{imgs[i].filename.split('/')[-1]}")

# def save_preds_metric(imgs,bboxes,output_dir,losses):
#     """
#     Las ímágenes deben haber sido abiertas con PIL.Image para poder
#     leer el nombre del fichero.
#     """
#     for i in range(len(imgs)):
#         img = draw_img(imgs[i],bboxes[i],losses[i])
#         img.save(f"{output_dir}/pred_{imgs[i].filename.split('/')[-1]}")
#         # img.save(f"{Path(output_dir)}/anot_{imgs[i].filename.split('/')[-1]}")

# def get_ground_truths(output_dir, imgs_dir=data_dir):
#     """
#     Dadas las imágenes en un directorio, busca sus anotaciones.
#     """
#     images = get_dir_imgs(imgs_dir)
#     anots = [read_anots(x) for x in images]
#     save_preds(images,anots,preds_dir+output_dir)

# def get_hist(data):
#     bins = len(list(set(data)))
#     counts, bins, _ = plt.hist(data,bins=bins)

# def save_confs_and_areas(bboxes, confs):
#     areas, cfs = [], []
#     for bbox in bboxes:
#         w,h = get_bbox_dim(bbox)
#         areas.append(w*h)

def save_hist(data, name):
    fig, ax = plt.subplots(figsize=(20,20))
    plt.hist(data)
    plt.savefig(name)

def bbox_area(bbox):
    return (bbox[2]-bbox[0])*(bbox[3]-bbox[1])

def inference(model,data_dir=data_dir,output_dir=output_dir):
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
    criterion = CE()
    for image, bbox, conf in zip(images, bboxes, confs):
        name = str(output)+"/"+image.filename.split("/")[-1].split(".")[0]
        save_hist([bbox_area(x) for x in bbox],name+"_areas_hist.png")
        save_hist(conf, name+"_confs_hist.png")
        draw_image(image, bbox, conf, name)


"""
data_ds = get_data_ds(get_dir_imgs_names())
model=get_model()
load_ex_model(model,"d801010")
model.eval()
#imgs,anots = map(list,zip(*data_ds.get_imgs_and_anots()))
imgs, anots = [i for i,_,_,_ in data_ds.get_imgs_and_anots()],[i for _,i,_,_ in data_ds.get_imgs_and_anots()]
bboxes,_,confs = model.predict(imgs)


"""




# def inference(model,imgs,output_dir=None):
#     """
#     Si se le pasa un rango, coge las imágenes de imgs_dir en ese rango.
#     Se le pueden pasar también imágenes (PIL).
#     """
#     output_dir = preds_dir+output_dir
#     if imgs==[]:
#         imgs = [Image.open(x) for x in os.listdir(data_dir)]
#     model.eval()
#     bboxes, _, _ = model.predict(imgs)
#     save_preds(imgs,bboxes,output_dir)
