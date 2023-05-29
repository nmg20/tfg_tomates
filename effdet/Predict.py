from Model import *
from pytorch_lightning import Trainer 

def predict_model(model, dm,num_epochs=1,logger=None):
    trainer = Trainer(gpus=1, max_epochs=num_epochs,logger=logger,profiler="simple")
        # gpus=1, num_epochs,num_sanity_val_steps=1)
    trainer.predict(model,dataloaders=dm.pred_dataloaders())