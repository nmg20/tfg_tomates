import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import patches
import numpy as np
from PIL import Image, ImageDraw as D, ImageFont
import os
import torchvision.transforms as T
import pandas as pd
import sys

# import Inference
import EffDetDataset

main_ds = "/media/rtx3090/Disco2TB/cvazquez/nico/datasets/T1280x720_test/"
output_dir = "/media/rtx3090/Disco2TB/cvazquez/nico/tfg_tomates/effdet/otputs/"
imagesets_dir = main_ds + "ImageSets/"

def area_bbox(bbox):
    return abs((bbox[2]-bbox[0])*(bbox[3]-bbox[1]))

def area_bboxes(bboxes):
	return [area_bbox(x) for x in bboxes]

def save_hist(data, name, nbins=None):
    """
    Recibe una lista anidada (confianzas de las predicciones o los
    bounding boxes), la aplana, dibuja su histograma y lo guarda.
    """
    fig, ax = plt.subplots(figsize=(20,20))
    if not nbins:
        bins = np.linspace(np.min(data),np.max(data))
    else:
        bins = list(range(0,np.max(data),nbins))
    plt.hist(data, bins)
    plt.title("Distribución del área de las bounding boxes")
    plt.savefig(name)
    plt.close()

"""
Pensar si coger las bboxes del csv o después de ser transformadas.
-> cambio de tamaño.
-> la distribución debería mantenerse.
"""
def csv_to_df(csv_path):
	return pd.read_csv(csv_path)

def get_bboxes_from_df(df):
	return np.concatenate([x for x in [df[df.image==x][["xmin", "ymin", "xmax", "ymax"]].values
		for x in set(df.image)]])

def build_path(file, ds="d801010"):
	if file=="all":
		name = f"{imagesets_dir}all_annotations.csv"
	else:
		name = f"{imagesets_dir}{ds}/labels{file}.csv"
	return name

def get_bboxes_from_ds(ds):
	return get_bboxes_from_df(ds.annotations_df)

# def get_main_ds(path)

def read_imageset_names(ds="d801010",file="test"):
    """
    Lee del set de imágenes de un dataset los nombres para emular que 
    se encuentran en la carpeta /data.
    """
    file = open(f"{main_ds}ImageSets/{ds}/{file}.txt")
    # file = open(file+".txt")
    names = [x+".jpg" for x in file.read().split("\n")[::-1]][1::]
    return names

def ds_to_bbox_sizes(ds):
	anns = []
	# an = [anns+x.tolist() for _,x,_,_ in ds.get_imgs_and_anots()]
		# anns=anns.concatenate((anns,x))
	for _,x,_,_ in ds.get_imgs_and_anots():
		an = []
		anns.append([an+y.tolist() for y in x])
	return anns


def get_all_hists(output=output_dir):
	dss, cjs = ["d801010","d701515","d602020","d502030"], ["train","test", "val"]
	for ds in dss:
		for cj in cjs:
			save_hist(area_bboxes(get_bboxes_from_csv(
				build_path(cj,ds))),f"{output}/{ds}_{cj}.png"
				, None) 