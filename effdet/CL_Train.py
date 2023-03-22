from pathlib import Path
from EffDetDataset import *
from Model import *
from Train import *
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
import os

models_dir = "modelos/"
dataset_dir = "../../datasets/Tomato_1280x720/"

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
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, help="Número de épocas de entrenamiento.")
    parser.add_argument('-n','--name',type=str)
    args = parser.parse_args()
    dm = get_dm_standalone(dataset_dir,args.name)
    model = EfficientDetModel()
    model_name = args.name
    # model_name = f"{args.epochs}eps"
    # if args.name!="no":
    #     model_name = f"{args.name}_{model_name}"
    logger = TensorBoardLogger("lightning_logs/",name=model_name)
    train_model(model,dm,args.epochs,logger)
    torch.save(model.state_dict(),f"{models_dir}{model_name}.pt")


if __name__=="__main__":
    main()