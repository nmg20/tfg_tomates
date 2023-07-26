from pytorch_lightning import Trainer 
from utils.config import logs_dir
import os
from pathlib import Path

# def get_ckpt(name):
#     logs = Path(logs_dir)
#     versions = [(x,int(x.split("_")[-1])) for x in os.listdir(logs / name)]
#     versions.sort(key= lambda x: x[1])
#     return os.listdir(logs / name / versions[-1][0] / "checkpoints")[0]

def get_ckpt(name):
    """
    Dado un nombre (distribucion de dataset p.ej: 402040, 502030, 801010,...)
    devuelve el checkpoint guardado en la última versión.
        -> mejorar para coger el checkpoint con más steps.
    """
    logs = Path(logs_dir)
    versions = [(x,int(x.split("_")[-1])) for x in os.listdir(logs / name)]
    versions.sort(key= lambda x: x[1], reverse=True)
    for version in versions:
        result = os.listdir(logs / name / version[0] / "checkpoints")
        if result != []:
            return str(logs / name / version[0] / "checkpoints" / result[0])

def train_model(model, dm, num_epochs, logger, path=None):
    # trainer = Trainer(gpus=1, max_epochs=num_epochs,num_sanity_val_steps=1,logger=logger,profiler="simple")
    # trainer = Trainer(gpus=1, max_epochs=num_epochs,num_sanity_val_steps=1,logger=logger)
        # gpus=1, num_epochs,num_sanity_val_steps=1)
    if path!=None:
        trainer = Trainer(gpus=1, max_epochs=num_epochs,
            num_sanity_val_steps=1,logger=logger,
            resume_from_checkpoint=path)
    else:
        trainer = Trainer(gpus=1, max_epochs=num_epochs,
            num_sanity_val_steps=1,logger=logger)

    trainer.fit(model,dm)
