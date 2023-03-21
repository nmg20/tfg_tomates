from pathlib import Path
import pandas as pd
# import model, dataset
from EffDetDataset import *
from Model import *
import torch
import torchvision
from pytorch_lightning import Trainer 

from torch.utils.tensorboard import SummaryWriter

def get_model():
    return EfficientDetModel()

# model = get_model()

def train_model(model, dm,num_epochs=1):
    trainer = Trainer(
        gpus=1, max_epochs=num_epochs,num_sanity_val_steps=1)
        # gpus=1, num_epochs,num_sanity_val_steps=1)
    trainer.fit(model,dm)
    trainer.test()