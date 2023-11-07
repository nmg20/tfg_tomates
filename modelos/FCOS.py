import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection import fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights

# import Visualize
from modelos.utils import image_sizes, compute_loss, threshold_fusion

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium') 

models_dir = "./pths/"

class FCOSLightning(LightningModule):
    def __init__(
        self,
        num_classes=1,
        lr=0.0002,
        threshold=0.1, #Threshold de scores de las bounding boxes
        iou_thr=0.3, #Threshold de IoU para considerarse la misma bounding box
    ):
        super().__init__()
        self.lr = lr
        self.model = fcos_resnet50_fpn(
            weights = FCOS_ResNet50_FPN_Weights.DEFAULT,
            weights_backbone = ResNet50_Weights.IMAGENET1K_V1
        )
        self.threshold = threshold
        self.iou_thr = iou_thr
        self.loss_fn = compute_loss
        self.model.num_classes = num_classes
        
    def forward(self, images, targets=None):
        outputs = self.model(images, targets)
        if not self.model.training or targets is None:
            outputs = threshold_fusion(
                outputs,
                image_sizes(images),
                iou_thr=self.iou_thr,
                skip_box_thr=self.threshold
            )
        return outputs

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        images, targets, ids = batch
        loss = self(images, targets)
        # Registramos el error de clasificación y regresión de bbox
        self.log('train_class_loss', loss['classification'].detach())
        self.log('train_box_loss', loss['bbox_regression'].detach())
        return {'loss' : loss['bbox_regression']}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, targets, ids = batch
        outputs = self(images, targets)
        batch_predictions = {
            'predictions' : [output['boxes'] for output in outputs],
            'targets' : targets,
            # 'image_ids' : ids,
        }
        loss = self.loss_fn(outputs, targets)
        self.log('val_loss', loss[1])
        return {'loss' : loss[1], 'batch_predictions' : batch_predictions}

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        images, targets, ids = batch
        outputs = self(images, targets)
        batch_predictions = {
            'predictions' : [output['boxes'] for output in outputs],
            'targets' : targets,
            # 'image_ids' : ids,
        }
        loss = self.loss_fn(outputs, targets)
        self.log('test_loss', loss[1])
        return {'loss' : loss[1], 'batch_predictions' : batch_predictions}
