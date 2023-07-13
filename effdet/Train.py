from Model import *
from pytorch_lightning import Trainer 

def train_model(model, dm, num_epochs,logger=None):
    # trainer = Trainer(gpus=1, max_epochs=num_epochs,num_sanity_val_steps=1,logger=logger,profiler="simple")
    trainer = Trainer(gpus=1, max_epochs=num_epochs,num_sanity_val_steps=1,logger=logger)
        # gpus=1, num_epochs,num_sanity_val_steps=1)
    trainer.fit(model,dm)