from pathlib import Path
import pandas as pd
from EffDetDataset import *
from Model import *
from utils.Visualize import uniquify_dir, draw_images_stacked, draw_losses
import os
import argparse
from torch.nn import CrossEntropyLoss as CE
from torchvision.transforms import Compose, ToPILImage, PILToTensor
from utils.config import *
from Dataset_Analysis import area_bbox, save_hist

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

def imageset_to_pil(ds="801010",name="test.txt"):
    names = read_imageset_names(ds)
    images = [Image.open(x) for x in os.listdir(images_dir) if x in names]
    return images

def inference_ds(model, name, ds):
    output = Path(uniquify_dir(output_dir+f"/{name}_run"))
    # if loss_flag==0:
    os.mkdir(output)
    losses = []
    for img, ann, num in zip([i for i,_,_,_ in ds.get_imgs_and_anots()], [i for _,i,_,_ in ds.get_imgs_and_anots()], list(range(len(ds.get_imgs_and_anots())))):
        bboxes, _, confs, loss = model.predict([img])
        losses.append(float(loss))
        # if loss_flag==0:
        draw_images_stacked(img, bboxes, confs, loss, f"{output}/predicted_img_{num}",ann)
    # if loss_flag==1:
    draw_losses(losses,(sum(losses)/len(losses)),f"{output}/{name}_pred_loss")
    return losses

def inference(model_name, file):
    """
    File -> leer nombres de imágenes de un archivo
        -> en su defecto leer de ./data/
    """
    if len(os.listdir(data_dir))==0:
        dm = get_dm_standalone(name=model_name,data_file=file)
        # file = file.split(".")[0]
    else:
        dm = get_dm_standalone(name=model_name,data_file=data_dir)
        # file = file.replace("/","").split(".")[1]
    # inference_dl(model,dm)
    model = load_n_eval(model_name)
    inference_ds(model,model_name,dm.pred_dataset().ds)

def get_pred(model, images):
    bboxes, _, confs, loss = model.predict(images)
    return (bboxes, confs, loss)

def load_n_eval(name):
    model = load_model(name)
    model.eval()
    return model

def inference_multimodel(file):
    model_names = ["d801010","d701515","d602020"]
    if len(os.listdir(data_dir))==0:
        dms = [get_dm_standalone(name=model,data_file=file) for model in model_names]
        file = file.split(".")[0]
    else:
        dms = [get_dm_standalone(name=model,data_file=data_dir) for model in model_names]
        file = file.replace("/","").split(".")[1]
    models = [load_n_eval(model) for model in model_names]


