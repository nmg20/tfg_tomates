import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import lightning.pytorch as pl

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights

from lightning.pytorch import Trainer

class AbstractPLModel(nn.Module):
    """
    Clase que en base a un backbone crea el modelo con una capa de umbralización.
    Abstracción para pasarle al Lightning Module
    """
    def __init__(self, backbone, num_classes=1):
        super(AbstractPLModel, self).__init__()
        if backbone=="fasterrcnn":
            self.model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
            self.model.out_channels = 2048
        elif backbone == "ssd":
            self.model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
        elif backbone == "resnet":
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
            num_features = self.backbone.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)
        elif backbone == "retinanet":
            self.model = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT)
        elif backbone=="fcos":
            self.model = fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.DEFAULT)

        self.threshold = torch.nn.Sigmoid()

    def forward(self, x):
        return self.model.forward(x)

class TomatoLightning(pl.LightningModule):
    """
    Clase que contiene el funcionamiento básico de un modelo a efectos de entrenamiento/validación/test.
    Crea un modelo en concreto en base al backbone específicado.
    El modelo creado ya añade la útima capa de umbralización.
    """
    def __init__(
        self,
        num_classes=1,
        backbone="fasterrcnn",
        lr=0.0002,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lr = lr
        self.model = AbstractPLModel(backbone, num_classes)

    def forward(self, images, targets):
        return self.model(images, targets)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images, targets)

        loss = F.cross_entropy(targets['bboxes'], bboxes)
        self.log('train_loss', loss)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images, targets)
        loss = F.cross_entropy(targets['bboxes'], bboxes)
        self.log('val_loss', loss)
        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images, targets)
        loss = F.cross_entropy(targets['bboxes'], bboxes)
        self.log('test_loss', loss)
        return loss 

def freeze_modules(model, modules=["regression_head"]):
    for name, module in model.named_modules():
        if any(module_name in name for module_name in modules):
            for param in module.parameters():
                param.requires_grad = False
            if len(list(module.children()))>0:
                freeze_modules(module, modules)
