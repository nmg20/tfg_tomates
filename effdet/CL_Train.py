from pathlib import Path
from EffDetDataset import *
from Model import *
from Train import *
from Inference import file_to_bboxes, save_hist
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
    return Path(path).mkdir(parents=True, exist_ok=True)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, help="Número de épocas de entrenamiento.")
    parser.add_argument('-n','--name',type=str)
    parser.add_argument('-d','--debug',type=int, help="Flag para debuggear registrando el área de los bounding boxes.")
    args = parser.parse_args()
    dm = get_dm_standalone(dataset_dir,args.name)
    #DEBUG
    if args.debug :
        data_file = open("outputs/data.dat","w")
    else:
        data_file = False
    model = EfficientDetModel(data_file)
    model_name = args.name
    # model_name = f"{args.epochs}eps"
    # if args.name!="no":
    #     model_name = f"{args.name}_{model_name}"
    logger = TensorBoardLogger("lightning_logs/",name=model_name)
    train_model(model,dm,args.epochs,logger)
    if args.debug==0 :
        torch.save(model.state_dict(),f"{models_dir}{model_name}.pt")
    else:
        data_file.close()
        save_hist(file_to_bboxes("outputs/data.dat"),f"outputs/Áreas Training {name}.png",100)
        

        


if __name__=="__main__":
    main()