import torch

from modelos.RetinaNet import RetinaNetLightning
from modelos.FasterRCNN import FasterRCNNLightning
from modelos.FCOS import FCOSLightning

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

from Visualize import *

from detection.engine import evaluate

def freeze_modules(model, modules=["regression_head"]):
    """
    Congela módulos con nombre del modelo.
    """
    for name, module in model.named_modules():
        if any(module_name in name for module_name in modules):
            for param in module.parameters():
                param.requires_grad = False
            if len(list(module.children()))>0:
                freeze_modules(module, modules)

def model_size(model):
    """
    Devuelve el tamaño del modelo completo en base al tamaño de 
    sus parámetros.
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

def load_model(path, model=None, fam="ret", threshold=0.0):
    if model is None:
        if fam=="ret":
            model = RetinaLightning(threshold=threshold)
        elif fam=="fast":
            model = FasterRCNNLightning(threshold=threshold)
        elif fam=="fcos":
            model = FCOSLightning(threshold=threshold)
    model.load_state_dict(torch.load(path))
    return model

def save_model(model, name):
    torch.save(model.state_dict(), f"{models_dir}/{name}.pt")

logger = TensorBoardLogger("./logs","retinanet")
if torch.cuda.is_available():
    trainer = Trainer(
        accelerator="cuda", 
        devices=1,
        max_epochs=40, 
        num_sanity_val_steps=1, 
        logger=logger
    )

def inference(model, batch):
    """
    Función de inferencia de un modelo sobre un conjunto de imágenes.
    Para comprobar el error asumimos que las imágenes vienen dentro de un
    dataloader, junto con los ground truths.
    """
    images, targets, ids = batch
    model.eval()
    outputs = model(images, targets)
    loss = model.loss_fn(outputs, targets)
    compare_outputs(images, outputs, targets, loss[0])
