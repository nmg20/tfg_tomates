from Model import *
from pytorch_lightning import Trainer 
from EffDetDataset import get_dm
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.loggers import TensorBoardLogger

def test_model(model_name, logger=None):
    model = load_model(model_name)
    name = model_name.split("/")[-1]
    dm = get_dm(name=name)
    logger = TensorBoardLogger("lightning_logs/tests",name=name)
    trainer = Trainer(
        accelerator="gpu",devices=1,max_epochs=1,logger=logger)
    trainer.test(model,dataloaders=dm.test_dataloader())