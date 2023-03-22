from Model import *
from pytorch_lightning import Trainer 

from torch.utils.tensorboard import SummaryWriter

def test_model(model, dm, logger=None):
    trainer = Trainer(
        gpus=1,max_epochs=1,logger=logger,)
        # gpus=1, num_epochs,num_sanity_val_steps=1)
    trainer.test(model,dataloaders=dm.test_dataloader())