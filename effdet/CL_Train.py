from pathlib import Path
from EffDetDataset import *
from Model import *
from Train import *
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
import argparse
import os
from config import *

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
    $python CL_Train.py -e 40 -n d801010 -o bifpn/d801010 -s 1 -f 1
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, help="Número de épocas de entrenamiento.")
    parser.add_argument('-n','--name',type=str)
    parser.add_argument('-o','--output',type=str)
    parser.add_argument('-s','--save',type=int, help="Flag para guardar el modelo entrenado.")
    # parser.add_argument('-f','--freez',type=int)
    
    args = parser.parse_args()
    model = get_model()
    dm = get_dm(name=args.name)
    train_model(model, dm, args.epochs, args.output)
    if args.save==1:
        # torch.save(model.state_dict(),f"{models_dir}{args.output}.pt")
        save_model(model,args.output)
        


if __name__=="__main__":
    main()
