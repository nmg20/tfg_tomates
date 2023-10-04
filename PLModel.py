import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights

# class FasterRCNNThres(FasterRCNN):
#     """
#     Modificación de FasterRCNN con una variable entrenable a modo de 
#     umbral de resultados.
#     """
#     def __init__(self, num_classes):
#         backbone = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
#         backbone.out_channels = 2048
#         super().__init__(backbone, num_classes, 
#             rpn_anchor_generator = AnchorGenerator(
#                 sizes=((32, 64, 128, 256, 512),), 
#                 aspect_ratios=((0.5, 1.0, 2.0),) * 5)
#             ,
#             box_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
#                 featmap_names=['0'],
#                 output_size=7,
#                 sampling_ratio=2)
#             )
#         self.threshold = torch.nn.Sigmoid()

#     def forward(self, images, targets=None):
#         outputs = super().forward(images, targets)
#         if not self.training or targets is None:
#             outputs[0]['scores'] = self.threshold(outputs[0]['scores'])
#         return outputs

class AbstractPLModel(nn.Module):
    """
    Clase que en base a un backbone crea el modelo con una capa de umbralización.
    Abstracción para pasarle al Lightning Module
    """
    def __init__(self, backbone, num_classes=1):
        super(AbstractPLModel, self).__init__()
        if backbone=="fasterrcnn":
            self.backbone = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
            self.backbone.out_channels = 2048
        elif backbone == "ssd":
            self.backbone = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
        elif backbone == "resnet":
            self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, num_classes)
        elif backbone == "retinanet":
            self.backbone = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT)
        elif backbone=="fcos":
            self.backbone = fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.DEFAULT)

        self.threshold = torch.nn.Sigmoid()

    def forward(self, x):
        return self.model.forward(x)

class TomatoLightning(LightningModule):
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
        self.model = AbstractPLModel(num_classes)

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