from pathlib import Path
import pandas as pd
from EffDetDataset import *
from Model import *
from Inference import *
import os
import argparse

models_dir = "./modelos/"
preds_dir = "./preds/"

imagesets_dir = "../../datasets/Tomato_1280x720/ImageSets/"
data_dir = os.path.abspath("./data/")
output_dir = os.path.abspath("./outputs/")

def validate_file(f):
    if not os.path.exists(f):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f

def dir_path(path):
    return Path(path).mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help="Número de épocas de entrenamiento.")
    # parser.add_argument('-o','--output',type=dir_path, help="Directorio del dataset a partir del que crear el dataframe.")
    parser.add_argument('-v','--version',type=int, help="Directorio del dataset a partir del que crear el dataframe.")

    args = parser.parse_args()
    # output_dir = preds_dir if len(args.output)==1 else args.output
    # model = load_model(args.model)
    # imgs = get_dir_imgs(data_dir)
    inference(args.model,args.version)

if __name__=="__main__":
    main()