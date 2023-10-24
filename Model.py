import torch
from modelos.RetinaNetSimple import RetinaTomatoLightning
from modelos.RetinaNetMThres import RetinaMThresTomatoLightning

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

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

def load_model(path, model=None, flag=1):
    if model is None:
        if flag==0:
            model = RetinaTomatoLightning()
        else:
            model = RetinaMThresTomatoLightning()
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