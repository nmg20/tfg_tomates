from pathlib import Path
from EffDetDataset import *
from Model import *
from Train import *
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
import os
from utils.config import *

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
    parser.add_argument('-s','--save',type=int, help="Flag para guardar el modelo entrenado.")
    parser.add_argument('-r','--res',type=int, help="Flag para seguir entrenando desde un checkpoint.")
    # parser.add_argument('-d','--debug',type=str, help="Flag para debuggear registrando el área de los bounding boxes.")
    args = parser.parse_args()
    # dm = get_dm_standalone(main_ds,args.name)
    dm = get_dm_standalone(name=args.name)
    #DEBUG
    # if args.debug :
    #     data_file = open(args.debug,"w")
    # else:
    #     data_file = False
    # data_file = False
    # model = EfficientDetModel(data_file)
    model = EfficientDetModel()
    model_name = args.name
    logger = TensorBoardLogger(logs_dir,name=model_name)
    path = None if args.res==1 else get_ckpt(args.name)
    train_model(model,dm,args.epochs,logger,None)
    if args.save==1:
        torch.save(model.state_dict(),f"{models_dir}{model_name}.pt")
    # if args.debug:
        # data_file.close()
        # save_hist(file_to_bboxes(args.debug),f"outputs/Áreas Training {name}.png",100)
        

        


if __name__=="__main__":
    main()