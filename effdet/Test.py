from Model import *
from pytorch_lightning import Trainer 

from torch.utils.tensorboard import SummaryWriter

def test_model(model, dm):
    trainer = Trainer(
        gpus=1, max_epochs=num_epochs,num_sanity_val_steps=1)
        # gpus=1, num_epochs,num_sanity_val_steps=1)
    trainer.test(model,dataloaders=dm.test_dataloader())

def main():
    parser = argparse.ArgumentParser()