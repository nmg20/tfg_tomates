from pathlib import Path
import pandas as pd
from EffDetDataset import *
from Model import *
import os
import argparse
from Inference import *

def validate_file(f):
    if not os.path.exists(f):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        # raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")
        # os.mkdir(path)
        return path

models_dir = "modelos/"
preds_dir = "preds/"

############################

def save_preds_m(model,e,ds,i,j,div):
    model.eval()
    imgs = get_preds(model,ds,i,j)
    d = f"{preds_dir}{div}preds_{e}epochs"
    name = f"{div}ED_{e}ep_test_"
    if not os.path.exists(d):
        os.mkdir(d)
    else:
        name = "alt_"+name
    for i in list(range(len(imgs))):
        imgs[i].save(f"{d}/{name}{i}.jpg")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=validate_file, help="Número de épocas de entrenamiento.")
    parser.add_argument('-o','--output',type=dir_path, help="Directorio del dataset a partir del que crear el dataframe.")

    args = parser.parse_args()
    output_dir = preds_dir if len(args.output)==1 else args.output
    model = EfficientDetModel()
    model.load_state_dict(torch.load(args.model))
    imgs = get_dir_imgs(data_dir)
    inference(model,imgs,output_dir)

if __name__=="__main__":
    main()