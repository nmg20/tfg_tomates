import torch
import torch.nn as nn
from lightning.pytorch import LightningModule

import effdet
import timm

from modelos.utils import images_sizes, compute_loss, compute_single_loss, threshold_fusion
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from lightning.pytorch.utilities.memory import recursive_detach

import sys
sys.path.append("..")
import config

def create_model(num_classes=1, image_size=512, architecture="tf_efficientnetv2_l"):
    effdet.config.model_config.efficientdet_model_param_dict[architecture] = dict(
        name=architecture,
        backbone_name=architecture,
        backbone_args=dict(drop_path_rate=0.2),
        num_classes=num_classes,
        url='', )
    config = effdet.get_efficientdet_config(architecture)
    config.update({'num_classes': num_classes})
    config.update({'image_size': (image_size, image_size)})
    net = effdet.EfficientDet(config, pretrained_backbone=True)
    net.class_net = effdet.efficientdet.HeadNet(
        config,
        num_outputs=config.num_classes,
    )
    return effdet.DetBenchTrain(net, config)

class EfficientDetLightning(LightningModule):
    def __init__(
        self,
        num_classes=config.NUM_CLASSES,
        lr=config.LR,
        threshold=0.2,
        iou_thr=config.IOU_THR,
    ):
        super().__init__()
        self.lr = lr
        self.model = create_model()
        self.threshold = threshold
        self.iou_thr = iou_thr
        self.loss_fn = compute_loss
        self.mean_ap = MeanAveragePrecision()
        self.mean_ap.warn_on_many_detections = False
        self.val_step_outputs = []
        self.val_step_targets = []
        self.test_step_outputs = []
        self.test_step_targets = []
    
    def forward(self, images, targets):
        outputs = self.model(images, targets)
        if not self.model.training or targets is None:
            detections = threshold_fusion(
                self.post_process_detections(
                    outputs['detections']
                ),
                images_sizes(images),
                iou_thr=self.iou_thr,
                skip_box_thr=self.threshold)
            outputs['detections'] = detections
        return outputs

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def post_process_detections(self, detections):
        """
        Función para darle el mismo formato a las detecciones de este modelo
        que a las de los demás para poder aplicar la misma función de loss
        y de evaluación.
        """
        predictions = []
        for detection in detections:
            predictions.append({
                "boxes": detection[:, :4], 
                "scores": detection[:, 4], 
                "labels": detection[:, 5]
            })
        return predictions

    def training_step(self, batch, batch_idx):
        images, annotations, _, image_ids = batch
        losses = self.forward(images, annotations)
        logging_losses = {
            "class_loss": losses["class_loss"].detach(),
            "box_loss": losses["box_loss"].detach(),
        }
        self.log("train_box_loss", losses["box_loss"],logger=True)
        self.log("train_class_loss", losses["class_loss"],logger=True)
        return losses['loss']

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, annotations, targets, image_ids = batch
        outputs = self.forward(images, annotations)
        detections = outputs["detections"]

        #Registramos la loss generada por el modelo
        self.log("og_val_class_loss", outputs["class_loss"].detach(), logger=True)
        self.log("og_val_box_loss", outputs["box_loss"].detach(), logger=True)
        #Guardamos los resultados para calcular el mAP
        self.val_step_outputs.extend(detections)
        self.val_step_targets.extend(targets)
        #Calculamos la loss con nuestra función
        loss = self.loss_fn(detections, targets)
        self.log("val_box_loss", loss["box"], logger=True)
        self.log("val_class_loss", loss["class"], logger=True)
        return {'og_loss': outputs["loss"], 'loss': loss['total']}

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        images, anns, targets, ids = batch
        outputs = self.forward(images, targets)
        detections = outputs["detections"]

        self.log('og_test_class_loss', outputs["class_loss"].detach())
        self.log('og_test_box_loss', outputs["box_loss"].detach())
        self.test_step_outputs.extend(detections)
        self.test_step_targets.extend(targets)

        loss = self.loss_fn(detections, targets)
        self.log("test_box_loss", loss["box"], logger=True)
        self.log("test_class_loss", loss["class"], logger=True)
        return {'og_loss': outputs["loss"], 'loss': loss['total']}

    @torch.no_grad()
    def on_validation_epoch_end(self):
        val_all_outputs = recursive_detach(self.val_step_outputs, to_cpu=True)
        val_all_targets = recursive_detach(self.val_step_targets, to_cpu=True)
        mean_ap = self.mean_ap(val_all_outputs, val_all_targets)
        for k in config.KEYS:
            self.log("val_"+k, mean_ap[k], logger=True)
        self.val_step_outputs.clear()
        self.val_step_targets.clear()

    @torch.no_grad()
    def on_test_epoch_end(self):
        test_all_outputs = self.test_step_outputs
        test_all_targets = self.test_step_targets
        mean_ap = self.mean_ap(test_all_outputs, test_all_targets)
        for k in config.KEYS:
            self.log("test_"+k, mean_ap[k], logger=True)
        self.test_step_outputs.clear()
        self.test_step_targets.clear()
