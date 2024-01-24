from numbers import Number
from typing import List
from functools import singledispatch

import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss as CE
from PIL import Image

from torchvision.transforms import Compose, ToPILImage, PILToTensor

from torchvision.ops import box_iou
from fastcore.dispatch import typedispatch
from lightning.pytorch import LightningModule

from ensemble_boxes import ensemble_boxes_wbf
import effdet
from config import *
import timm
from EffDetDataset import *
import matplotlib.pyplot as plt

from fastcore.basics import patch
from objdetecteval.metrics.coco_metrics import get_coco_stats
from torchmetrics.detection.mean_ap import MeanAveragePrecision

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

class EffDetModel(LightningModule):
    def __init__(
        self,
        num_classes=config.NUM_CLASSES,
        image_size=512,
        prediction_confidence_threshold=0.4,
        learning_rate=config.LR,
        wbf_iou_threshold=config.IOU_THR,
        skip_thr=0.43,
        model_architecture='tf_efficientnetv2_l',
    ):
        super().__init__()
        self.image_size = image_size
        self.model = create_model(
            num_classes, image_size, architecture=model_architecture
        )
        self.prediction_confidence_threshold = prediction_confidence_threshold
        self.lr = learning_rate
        self.wbf_iou_threshold = wbf_iou_threshold
        self.skip_thr = skip_thr
        self.mean_ap = MeanAveragePrecision()
        self.mean_ap.warn_on_many_detections = False
        self.validation_outputs = []
        self.test_outputs = []
    
    def forward(self, images, targets):
        return self.model(images, targets)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        images, annotations, _, image_ids = batch
        losses = self.model(images, annotations)
        logging_losses = {
            "class_loss": losses["class_loss"].detach(),
            "box_loss": losses["box_loss"].detach(),
        }
        # self.log("train_loss", losses["loss"], on_epoch=True, prog_bar=True,
        #          logger=True)
        self.log("train_box_loss", losses["box_loss"],logger=True)
        self.log("train_class_loss", losses["class_loss"],logger=True)
        return losses['loss']

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, annotations, targets, image_ids = batch
        outputs = self.model(images, annotations)
        self.validation_outputs.append(outputs)
        detections = outputs["detections"]
        batch_predictions = {
            "predictions": detections,
            "targets": targets,
            "image_ids": image_ids,
        }
        logging_losses = {
            "class_loss": outputs["class_loss"].detach(),
            "box_loss": outputs["box_loss"].detach(),
        }
        # self.log("valid_loss", outputs["loss"], on_epoch=True, prog_bar=True,
        #          logger=True, sync_dist=True)
        self.log("val_box_loss", logging_losses["box_loss"], logger=True)
        self.log("val_class_loss", logging_losses["class_loss"], logger=True)
        # detections = self.process_detections(outputs['detections'], images)
        # mean_ap = self.mean_ap(detections, targets)
        # for k in config.KEYS:
        #     self.log("val_"+k, mean_ap[k], logger=True)
        return {'loss': outputs["loss"], 'batch_predictions': batch_predictions}

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        image, annotations, targets, image_ids = batch
        outputs = self.model(image, annotations)
        self.test_outputs.append(outputs)
        detections = outputs["detections"]
        batch_predictions = {
            "predictions": detections,
            "targets": targets,
            "image_ids": image_ids,
        }
        losses = [outputs["loss"],outputs["box_loss"]]
        logging_losses = {"box_loss": outputs["box_loss"].detach(),}
        # self.log("test_loss", outputs["loss"], on_epoch=True, prog_bar=True,
        #          logger=True, sync_dist=True)
        self.log("test_box_loss", logging_losses["box_loss"], logger=True)
        self.log("test_class_loss", logging_losses["class_loss"], logger=True)
        mean_ap = self.mean_ap(outputs, targets)
        for k in config.KEYS:
            self.log("test_"+k, mean_ap[k], logger=True)
        return {'loss': outputs["loss"], 'batch_predictions': batch_predictions}

    def _run_inference(self, images_tensor):
        image_sizes = [image.shape[1:] for image in images_tensor]
        targets = self._create_dummy_inference_targets(
            num_images=images_tensor.shape[0]
            )
        results = self.model(images_tensor.to(self.device), targets)
        loss = results['loss']
        detections = results[
            "detections"
        ]
        (
            predicted_bboxes,
            predicted_class_confidences,
            predicted_class_labels,
        ) = self.post_process_detections(detections)

        scaled_bboxes = self.__rescale_bboxes(
            predicted_bboxes=predicted_bboxes, image_sizes=image_sizes
        )

        return scaled_bboxes, predicted_class_labels, predicted_class_confidences, loss

    def run_wbf(self, predictions, image_sizes, image_size=512, iou_thr=0.44, skip_box_thr=0.43, weights=None):
        bboxes = []
        confidences = []
        class_labels = []
        detections = []

        for prediction, sizes in zip(predictions, image_sizes):
            boxes = [(prediction["boxes"] / image_size).tolist()]
            scores = [prediction["scores"].tolist()]
            labels = [prediction["classes"].tolist()]
            boxes, scores, labels = ensemble_boxes_wbf.weighted_boxes_fusion(
                boxes,
                scores,
                labels,
                weights=weights,
                iou_thr=iou_thr,
                skip_box_thr=skip_box_thr,
            )
            boxes = boxes * (image_size - 1)
            boxes = self.__rescale_bboxes(predicted_bboxes=boxes, image_sizes=sizes)
            detections.append({
                'boxes': torch.tensor(boxes).to(config.DEVICE),
                'scores': torch.tensor(np.array(scores)).to(config.DEVICE),
                'labels': torch.tensor(np.array([int(x) for x in labels])).to(config.DEVICE)
            })
        return detections

    def _postprocess_single_prediction_detections(self, detections):
        boxes = detections.detach().cpu().numpy()[:, :4]
        scores = detections.detach().cpu().numpy()[:, 4]
        classes = detections.detach().cpu().numpy()[:, 5]
        indexes = np.where(scores > self.prediction_confidence_threshold)[0]
        boxes = boxes[indexes]
        return {"boxes": boxes, "scores": scores[indexes], "classes": classes[indexes]}

    def process_detections(self, detections, images):
        preds = []
        for i in range(detections.shape[0]):
            preds.append(
                self._postprocess_single_prediction_detections(detections[i])
            )
        image_sizes = [(image.shape[1], image.shape[2]) for image in images]
        predictions = self.run_wbf(
            preds, image_sizes, image_size=self.image_size, iou_thr=self.wbf_iou_threshold,
            skip_box_thr=self.skip_thr)
        return predictions

    def __rescale_bboxes(self, predicted_bboxes, image_sizes):
        scaled_bboxes = []
        im_h, im_w = image_sizes
        for bboxes in predicted_bboxes:
            if len(bboxes) > 0:
                scaled_bboxes.append(
                    (
                        np.array(bboxes)
                        * [
                            round(im_w / self.image_size,2),
                            round(im_h / self.image_size,2),
                            round(im_w / self.image_size,2),
                            round(im_h / self.image_size,2),
                        ]
                    ).tolist()
                )
            else:
                scaled_bboxes.append(bboxes)

        return scaled_bboxes

# @patch
# def add_pred_outputs(self: EffDetModel, outputs):
#     detections = torch.cat(
#         [output["batch_predictions"]["predictions"] for output in outputs]
#     )
#     image_ids = []
#     targets = []
#     for output in outputs:
#         batch_predictions = output["batch_predictions"]
#         image_ids.extend(batch_predictions["image_ids"])
#         targets.extend(batch_predictions["targets"])
#     (
#         predicted_bboxes,
#         predicted_class_confidences,
#         predicted_class_labels,
#     ) = self.post_process_detections(detections)
#     return (
#         predicted_class_labels,
#         image_ids,
#         predicted_bboxes,
#         predicted_class_confidences,
#         targets,
#     )

#     @patch
#     def on_validation_epoch_end(self):
#         outputs = torch.stack(self.validation_outputs)
#         print(stats)
#         for k in config.KEYS:
#             self.log("mean_val_"+k, stats[k], logger=True)
#         return {'mean_epoch_val_loss': outputs['loss'], 'metrics': stats}

def load_model(model,path):
    model.load_state_dict(torch.load(path))
    return model

def save_model(model, name):
    torch.save(model.state_dict(), f"{models_dir}/{name}.pt")

def freeze_layers(model, layer_min=2, layer_max=4):
    """
    bifpn = min=2, max=3
    bifpn + head = min=2, max=4
    head = min=3, max=4
    """
    layers = 0
    for children in model.children():
        for children2 in children.children():
            for child in children2.children():
                layers += 1
                if layers<layer_min or layers>layer_max:
                    for param in child.parameters():
                        param.requires_grad = False
