import torch
from pytorch_lightning import LightningModule

from torchvision.models import resnet50, mobilenet_v2
from torchvision.models.detection import ssd300_vgg16, retinanet_resnet50_fpn, fasterrcnn_resnet50_fpn, fcos_resnet50_fpn

class FasterRCNNThres(FasterRCNN):
    """
    Modificación de FasterRCNN con una variable entrenable a modo de 
    umbral de resultados.
    """
    def __init__(self, num_classes):
        backbone = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        backbone.out_channels = 2048
        super().__init__(backbone, num_classes, 
            rpn_anchor_generator = AnchorGenerator(
                sizes=((32, 64, 128, 256, 512),), 
                aspect_ratios=((0.5, 1.0, 2.0),) * 5)
            ,
            box_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
                featmap_names=['0'],
                output_size=7,
                sampling_ratio=2)
            )
        self.threshold = torch.nn.Sigmoid()

    def forward(self, images, targets=None):
        outputs = super().forward(images, targets)
        if not self.training or targets is None:
            outputs[0]['scores'] = self.threshold(outputs[0]['scores'])
        return outputs

class AbstractPLModel(nn.Module):
    """
    Clase que en base a un backbone crea el modelo con una capa de umbralización.
    Abstracción para pasarle al Lightning Module
    """
    def __init__(self, backbone, num_classes):
        super(AbstractPLModel, self).__init__()
        self.num_classes = num_classes
        if backbone == "resnet":
            self.backbone = models.resnet50(pretrained=True)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, self.num_classes)
        elif backbone == "ssd":
            self.backbone = 
        if backbone=="fasterrcnn":
            self.backbone = FasterRCNNThres(num_classes)

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