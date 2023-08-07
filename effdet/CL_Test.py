from EffDetDataset import *
from Model import *
from Test import *
import torch
from pytorch_lightning import Trainer
import argparse
import os

models_dir = "modelos/"
dataset_dir = "../../datasets/Tomato_1280x720/"

"""
$python CL_Test.py -n "d701515"
(ruta al .pt, sin ./modelos delante y sin el .pt)
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--name',type=str)
    args = parser.parse_args()
    
    test_model(args.name)
if __name__=="__main__":
    main()