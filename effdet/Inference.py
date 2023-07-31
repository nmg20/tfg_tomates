from pathlib import Path
import pandas as pd
from EffDetDataset import *
from Model import *
from utils.Visualize import uniquify_dir, draw_images_stacked, draw_losses, draw_image
import os
import argparse
from torch.nn import CrossEntropyLoss as CE
from torchvision.transforms import Compose, ToPILImage, PILToTensor
from utils.config import *
from Dataset_Analysis import area_bbox, area_bboxes, save_hist

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
    losses, confs = [], []
    for img, ann, num in zip([i for i,_,_,_ in ds.get_imgs_and_anots()], [i for _,i,_,_ in ds.get_imgs_and_anots()], list(range(len(ds.get_imgs_and_anots())))):
        bboxes, _, conf, loss = model.predict([img])
        losses.append(float(loss))
        confs.append(conf)
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

#########################################################

def inference_simple(model_name):
    output = Path(uniquify_dir(output_dir+f"/{model_name}/run"))
    os.mkdir(output)
    # output = f"{output_dir}/{model_name}/run1"
    losses, confs, areas = [], [], []
    if len(os.listdir(data_dir))==0:
        images = [Image.open(images_dir+x) for x in read_imageset_names(model_name)[0:20]]
        # images = [Image.open(images_dir+x) for x in read_imageset_names(model_name)]
    else:
        images = [Image.open(data_dir+x) for x in os.listdir(data_dir)]
    model = load_n_eval(model_name)
    for image in images:
        bboxes, conf, loss = get_pred(model, image)
        # print("\nConfs (Preds):",conf,"\n")
        # print(f"{image.filename.split('/')[-1]}")
        draw_image(image, bboxes, conf[0], loss, f"{output}/{image.filename.split('/')[-1]}")
        confs.append(conf[0])
        areas.append(area_bboxes(bboxes[0]))
    confs = np.array([item for sublist in confs for item in sublist])
    areas = np.array([item for sublist in areas for item in sublist])
    # print(confs)
    save_hist(areas, f"{output}/areas_{model_name}.png")
    save_hist(confs, f"{output}/confs_{model_name}.png")

###########################################################

def get_pred(model, image):
    bboxes, _, confs, loss = model.predict([image])
    return (bboxes, confs, loss)

def get_preds(model, images):
    return [get_pred(model,x) for x in images]

def load_n_eval(name):
    """
    Carga un modelo preentrenado y activa el modo evaluación.
    """
    model = load_model(name,0)
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



def prueba(name):
    image = Image.open(data_dir+"/"+os.listdir(data_dir)[0])
    # model = load_model(name,0)
    output = Path(uniquify_dir(output_dir+f"/test/run"))
    os.mkdir(output)
    # s = [0.2,0.25, 0.3, 0.35, 0.37, 0.4, 0.42, 0.45, 0.5]
    s = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    for x in s:
        model = load_model(name,conf=0.2,skip=x)
        model.eval()
        bboxes, confs, loss = get_pred(model, image)
        draw_image(image, bboxes, confs[0], loss, f"{output}/{name}_{x}_{image.filename.split('/')[-1]}")
        save_hist(confs, f"{output}/confs_{name}_{x}.png")
