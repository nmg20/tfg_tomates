from pytorch_lightning import Trainer 
from utils.config import logs_dir
from Model import EfficientDetModel
from EffDetDataset import get_dm
import os
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger

# def get_ckpt(name):
#     """
#     Dado un nombre (distribucion de dataset p.ej: 402040, 502030, 801010,...)
#     devuelve el checkpoint guardado en la última versión.
#         -> mejorar para coger el checkpoint con más steps.
#     """
#     logs = Path(logs_dir)
#     versions = [(x,int(x.split("_")[-1])) for x in os.listdir(logs / name)]
#     versions.sort(key= lambda x: x[1], reverse=True)
#     for version in versions:
#         result = os.listdir(logs / name / version[0] / "checkpoints")
#         if result != []:
#             return str(logs / name / version[0] / "checkpoints" / result[0])

def train(model,dm,num_epochs,logger):
    trainer = Trainer(accelerator="gpu", devices=1, max_epochs=num_epochs,
        num_sanity_val_steps=1, logger=logger)
    trainer.fit(model, dm)

def train_model(model, name, num_epochs, output):
    """
    Prepara los parámetros para entrenar el modelo.
    """
    logger = TensorBoardLogger(logs_dir,name=output)
    dm = get_dm(name=name)
    train(model, dm, num_epochs, logger)

def resume_train(model, dm, num_epochs, logger, path):
    """
    Continúa el entrenamiento desde un punto guardado.
    """
    model = EfficientDetModel()
    logger = TensorBoardLogger(logs_dir,name=model_name)
    dm = get_dm(name=model_name)
    trainer = Trainer(gpus=1, max_epochs=num_epochs,
        num_sanity_val_steps=1,logger=logger,
        resume_from_checkpoint=path)