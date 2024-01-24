import torch

from modelos.RetinaNet import RetinaNetLightning
from modelos.FasterRCNN import FasterRCNNLightning
from modelos.FCOS import FCOSLightning
from modelos.SSD import SSDLightning
from modelos.SSDLite import SSDLiteLightning
from modelos.EfficientDet import EfficientDetLightning

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

from Visualize import *

from modelos.utils import compute_single_loss

import config

def freeze_modules(model, modules=["backbone"]):
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

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

def save_model(model, name):
    torch.save(model.state_dict(), f"{config.MODELS_DIR}/{name}.pt")

def set_num_classes(model, num_classes=1):
    logits = model.head.classification_head.cls_logits
    new_class_out = nn.Conv2d(
        in_channels=logits.in_channels,
        out_channels=num_classes,
        kernel_size=logits.kernel_size,
        stride=logits.stride,
        padding=logits.padding,
        dilation=logits.dilation,
        groups=logits.groups,
        padding_mode=logits.padding_mode,
        device=logits.weight.device,
        dtype=logits.weight.dtype
    )
    model.head.classification_head.cls_logits = new_class_out

def inference(model, batch):
    """
    Dado un modelo y un batch, obtiene el resultado de la inferencia,
    calcula la loss y dibuja las detecciones por pantalla junto a las
    ground truths (targets).
    """
    images, targets, ids = batch
    model.eval()
    outputs = model(images, targets)
    losses = [
        compute_single_loss(
            o['boxes'],
            t['boxes'],
            o['labels'],
            t['labels']) for (o,t) in zip(outputs, targets)]
    detections = [o['boxes'] for o in outputs]
    labels = [o['labels'] for o in outputs]
    scores = [o['scores'] for o in outputs]
    gts = [t['boxes'] for t in targets]
    compare_outputs(images, detections, gts, labels, scores, losses)
