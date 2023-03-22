from pathlib import Path
import pandas as pd
from EffDetDataset import *
from Model import *
import os
import argparse

main_dataset = "../../datasets/Tomato_1280x720/"
preds_dir = os.path.abspath("./preds/")
data_dir = os.path.abspath("./data/")

############################

def get_dir_imgs(imgs_dir):
    return [Image.open(imgs_dir+"/"+x) for x in os.listdir(imgs_dir)]

def save_preds(imgs,bboxes,output_dir):
    """
    Las ímágenes deben haber sido abiertas con PIL.Image para poder
    leer el nombre del fichero.
    """
    for i in range(len(imgs)):
        img = draw_img(imgs[i],bboxes[i])
        # img.save(f"{output_dir}/pred_{imgs[i].filename.split('/')[-1]}")
        img.save(f"{os.path.abspath(output_dir)}/pred_{imgs[i].filename.split('/')[-1]}")

def check_output_dir(d=None):
    if (d and not os.path.exists(d)) or (not d):
        os.mkdir(d)

def inference(model,imgs,output_dir=None):
    """
    Si se le pasa un rango, coge las imágenes de imgs_dir en ese rango.
    Se le pueden pasar también imágenes (PIL).
    """
    check_output_dir(output_dir)
    if imgs==[]:
        imgs = [Image.open(x) for x in os.listdir(data_dir)]
    model.eval()
    bboxes, _, _ = model.predict(imgs)
    save_preds(imgs,bboxes,output_dir)