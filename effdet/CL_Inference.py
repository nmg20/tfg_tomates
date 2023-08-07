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
    """
    Elegir si leer las imágenes de un directorio o un fichero con los nombres
    
        -> añadir un flag para simplemente registrar la loss.
    """
    #$python CL_Inference.py -m d801010 -f test.txt
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help="Número de épocas de entrenamiento.")
    parser.add_argument('-o', '--output', type=str, help="Nombre de la carpeta a crear con los resultados.")
    parser.add_argument('-c', '--conf', type=float, help="Flag para que la inferenci sólo registre la loss para cada imagen.")
    parser.add_argument('-s', '--skip', type=float, help="Flag para que la inferenci sólo registre la loss para cada imagen.")

    args = parser.parse_args()
    # inference_simple(args.model)
    inference(args.model, args.output, args.conf, args.skip)

if __name__=="__main__":
    main()