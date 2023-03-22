from EffDetDataset import *
from Model import *
from Validate import *
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
import os

models_dir = "modelos/"
dataset_dir = "../../datasets/Tomato_1280x720/"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--name',type=str)
    args = parser.parse_args()
    dm = get_dm_standalone(dataset_dir,args.name)
    model = EfficientDetModel()
    load_ex_model(model,models_dir+args.name+".pt")
    logger = TensorBoardLogger("lightning_logs/",name=args.name+"_test")
    validate_model(model,dm,num_epochs=1,logger=logger)

if __name__=="__main__":
    main()