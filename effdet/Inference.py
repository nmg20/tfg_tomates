from pathlib import Path
import pandas as pd
from EffDetDataset import *
from Model import *
from Visualize import uniquify_dir, draw_images_stacked, draw_losses, draw_image, save_hist
import os
import argparse
from torch.nn import CrossEntropyLoss as CE
from torchvision.transforms import Compose, ToPILImage, PILToTensor
from config import *
from Dataset_Analysis import area_bbox, area_bboxes #, save_hist, save_hist2

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

def imageset_to_pil(ds="d801010",name="test.txt"):
    names = read_imageset_names(ds)
    images = [Image.open(f"{main_ds}JPEGImages/{x}") for x in os.listdir(images_dir) if x in names]
    return images

# def inference_ds(model, name, ds):
#     output = Path(uniquify_dir(output_dir+f"/{name}_run"))
#     # if loss_flag==0:
#     os.mkdir(output)
#     losses, confs = [], []
#     for img, ann, num in zip([i for i,_,_,_ in ds.get_imgs_and_anots()], [i for _,i,_,_ in ds.get_imgs_and_anots()], list(range(len(ds.get_imgs_and_anots())))):
#         bboxes, _, conf, loss = model.predict([img])
#         losses.append(float(loss))
#         confs.append(conf)
#         draw_images_stacked(img, bboxes, confs, loss, f"{output}/predicted_img_{num}",ann)
#     # if loss_flag==1:
#     draw_losses(losses,(sum(losses)/len(losses)),f"{output}/{name}_pred_loss")
#     return losses

# def inference_simple2(model_name):
#     # output = Path(uniquify_dir(output_dir+f"/{model_name}/run"))
#     # os.mkdir(output)
#     output = f"{output_dir}/{model_name}/run"
#     losses, confs, areas = [], [], []
#     if len(os.listdir(data_dir))==0:
#         images = [Image.open(images_dir+"/"+x) for x in read_imageset_names(model_name)[0:20]]
#         # images = [Image.open(images_dir+x) for x in read_imageset_names(model_name)]
#     else:
#         images = [Image.open(data_dir+"/"+x) for x in os.listdir(data_dir)]
#     model = load_n_eval(model_name)
#     for image in images:
#         bboxes, conf, loss = get_pred(model, image)
#         # print("\nConfs (Preds):",conf,"\n")
#         # print(f"{image.filename.split('/')[-1]}")
#         draw_image(image, bboxes, conf[0], loss, f"{output}/{image.filename.split('/')[-1]}")
#         confs.append(conf[0])
#         areas.append(area_bboxes(bboxes[0]))
#     confs = np.array([item for sublist in confs for item in sublist])
#     areas = np.array([item for sublist in areas for item in sublist])
#     # print(confs)
#     save_hist(areas, f"{output}/areas_{model_name}.png")
#     save_hist(confs, f"{output}/confs_{model_name}.png")

###########################################################

def get_pred(model, image):
    bboxes, _, confs, loss = model.predict([image])
    return (bboxes[0], confs[0], loss)

def get_preds(model, images):
    return [get_pred(model,x) for x in images]

#################################################################

def infere(model, name, ds, output):
    for image, ann in zip([i for i,_,_,_ in ds.get_imgs_and_anots()], [i for _,i,_,_ in ds.get_imgs_and_anots()]):
        bboxes, confs, loss = get_pred(model, image)
        media = np.sum(confs)/len(confs)
        out = str(output) + "/" + image.filename.split("/")[-1].split(".")[0]
        os.mkdir(out)
        draw_images_stacked(image, bboxes, confs, (loss,media), f"{out}/{image.filename.split('/')[-1]}",ann)
        name1 = name + "_" + image.filename.split("/")[-1].split(".")[0]
        # save_hist2(confs, f"{output}/confs_{name}.png","Distribución de los porcentajes de confianza")
        # save_hist2(area_bboxes(bboxes), f"{output}/bboxes_{name}.png","Distribución del área de las bounding boxes")
        save_hist(area_bboxes(bboxes), confs, area_bboxes(ann), f"{out}/hists_{name1}.png")

def inference(model_name, output_name, cnf=0.1, sk=0.2):
    name = model_name.split("/")[-1].split("_")[0]
    dm = get_dm(name=name)
    if cnf<0 or sk<0:
        model = EfficientDetModel()
    else:
        model = load_model(model_name, conf=cnf, skip=sk)
    if len(output_name)==1:
        output = Path(uniquify_di(f"{output_dir}/{name}/run"))
    else:
        output = Path(uniquify_dir(f"{output_dir}/{output_name}/run"))
    model.eval()
    os.makedirs(output)
    infere(model, name, dm.pred_dataset().ds, output)


