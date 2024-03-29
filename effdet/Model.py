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
from sklearn.metrics import average_precision_score

from fastcore.dispatch import typedispatch
# from pytorch_lightning import LightningModule
from lightning.pytorch import LightningModule
# from pytorch_lightning.core.decorators import auto_move_data

from ensemble_boxes import ensemble_boxes_wbf
import effdet
# from effdet.config.model_config import efficientdet_model_param_dict
# from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
# from effdet.efficientdet import HeadNet
# from effdet.config.model_config import efficientdet_model_param_dict
from config import *
import timm
from EffDetDataset import *
import matplotlib.pyplot as plt


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
    # return DetBenchPredict(net, config)

# def images_to_tensor(images, transform=get_valid_transforms(512)):
#     image_sizes = [(image.size[1], image.size[0]) for image in images]
#     return torch.stack(
#         [
#             transform(
#                 image=np.array(image, dtype=np.float32),
#                 labels=np.ones(1),
#                 bboxes=np.array([[0, 0, 1, 1]]),
#             )["image"]
#             for image in images
#         ]
#     ), image_sizes

# def run_wbf(predictions, image_size=512, iou_thr=0.44, skip_box_thr=0.2, weights=None):
def run_wbf(predictions, image_size=512, iou_thr=0.44, skip_box_thr=0.43, weights=None):
    bboxes = []
    confidences = []
    class_labels = []

    for prediction in predictions:
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
        bboxes.append(boxes.tolist())
        confidences.append(scores.tolist())
        class_labels.append(labels.tolist())
    return bboxes, confidences, class_labels

class EfficientDetModel(LightningModule):
    def __init__(
        self,
        num_classes=1,
        img_size=512,
        prediction_confidence_threshold=0.4,
        learning_rate=0.0002,
        wbf_iou_threshold=0.44,
        skip_thr=0.43,
        inference_transforms=get_pred_transforms(target_img_size=512),
        model_architecture='tf_efficientnetv2_l',
        output_dir="./outputs/",
    ):
        super().__init__()
        self.img_size = img_size
        self.model = create_model(
            num_classes, img_size, architecture=model_architecture
        )
        self.prediction_confidence_threshold = prediction_confidence_threshold
        self.lr = learning_rate
        self.wbf_iou_threshold = wbf_iou_threshold
        self.inference_tfms = inference_transforms
        self.output_dir = output_dir
        self.skip_thr = skip_thr
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def forward(self, images, targets):
        return self.model(images, targets)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        images, annotations, _, image_ids = batch
        # if self.data_file :
            # save_sizes(annotations['bbox'][0].cpu().numpy(), self.data_file)
            # save_anots(annotations['bbox'][0].cpu().numpy(), self.data_file)
        losses = self.model(images, annotations)

        logging_losses = {
            "class_loss": losses["class_loss"].detach(),
            "box_loss": losses["box_loss"].detach(),
        }

        self.log("train_loss", losses["loss"], on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)
        self.log("train_box_loss", losses["box_loss"], on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)

        return losses['loss']

    @torch.no_grad()
    # @torch.inference_mode()
    def validation_step(self, batch, batch_idx):
        images, annotations, targets, image_ids = batch
        # print("\nAnnotations: ", annotations)
        outputs = self.model(images, annotations)
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

        self.log("valid_loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True,
                 logger=True, sync_dist=True)
        self.log("valid_box_loss", logging_losses["box_loss"], on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)

        return {'loss': outputs["loss"], 'batch_predictions': batch_predictions}
    
    @torch.no_grad()
    # @torch.inference_mode()
    def test_step(self, batch, batch_idx):
        image, annotations, targets, image_ids = batch
        outputs = self.model(image, annotations)
        detections = outputs["detections"]

        batch_predictions = {
            "predictions": detections,
            "targets": targets,
            "image_ids": image_ids,
        }
        losses = [outputs["loss"],outputs["box_loss"]]
        logging_losses = {"box_loss": outputs["box_loss"].detach(),}

        self.log("test_loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True,
                 logger=True, sync_dist=True)
        self.log("test_box_loss", logging_losses["box_loss"], on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        return {'loss': outputs["loss"], 'batch_predictions': batch_predictions}

    # @typedispatch
    # @torch.inference_mode()
    # def predict(self, images: List):
    #     """
    #     For making predictions from images
    #     Args:
    #         images: a list of PIL images

    #     Returns: a tuple of lists containing bboxes, predicted_class_labels, predicted_class_confidences

    #     """
    #     images_tensor, images_sizes = images_to_tensor(images)
    #     return self._run_inference(images_tensor, images_sizes)

    @typedispatch
    def predict(self, images: List):
        """
        For making predictions from images
        Args:
            images: a list of PIL images

        Returns: a tuple of lists containing bboxes, predicted_class_labels, predicted_class_confidences

        """
        image_sizes = [(image.size[1], image.size[0]) for image in images]
        images_tensor = torch.stack(
            [
                self.inference_tfms(
                    image=np.array(image, dtype=np.float32),
                    labels=np.ones(1),
                    bboxes=np.array([[0, 0, 1, 1]]),
                )["image"]
                for image in images
            ]
        )

        return self._run_inference(images_tensor, image_sizes)

    @typedispatch
    # @torch.inference_mode()
    def predict(self, images_tensor: torch.Tensor):
        """
        For making predictions from tensors returned from the model's dataloader
        Args:
            images_tensor: the images tensor returned from the dataloader

        Returns: a tuple of lists containing bboxes, predicted_class_labels, predicted_class_confidences

        """
        if images_tensor.ndim == 3:
            images_tensor = images_tensor.unsqueeze(0)
        if (
            images_tensor.shape[-1] != self.img_size
            or images_tensor.shape[-2] != self.img_size
        ):
            raise ValueError(
                f"Input tensors must be of shape (N, 3, {self.img_size}, {self.img_size})"
            )

        num_images = images_tensor.shape[0]
        image_sizes = [(self.img_size, self.img_size)] * num_images

        return self._run_inference(images_tensor, image_sizes)

    def _run_inference(self, images_tensor, image_sizes):
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
    
    def _create_dummy_inference_targets(self, num_images, bboxes=None):
        dummy_targets = {
            "bbox": [torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=self.device)
                for i in range(num_images)
            ],
            "cls": [torch.tensor([1.0], device=self.device) for i in range(num_images)],
            "img_size": torch.tensor(
                [(self.img_size, self.img_size)] * num_images, device=self.device
            ).float(),
            "img_scale": torch.ones(num_images, device=self.device).float(),
        }

        return dummy_targets
    
    def post_process_detections(self, detections):
        predictions = []
        for i in range(detections.shape[0]):
            predictions.append(
                self._postprocess_single_prediction_detections(detections[i])
            )

        predicted_bboxes, predicted_class_confidences, predicted_class_labels = run_wbf(
            predictions, image_size=self.img_size, iou_thr=self.wbf_iou_threshold, skip_box_thr=self.skip_thr
        )

        return predicted_bboxes, predicted_class_confidences, predicted_class_labels

    def _postprocess_single_prediction_detections(self, detections):
        boxes = detections.detach().cpu().numpy()[:, :4]
        scores = detections.detach().cpu().numpy()[:, 4]
        classes = detections.detach().cpu().numpy()[:, 5]
        # Incluir aquí umbralización 
        indexes = np.where(scores > self.prediction_confidence_threshold)[0]
        # media = (sum(scores)/len(scores))/2
        # print("Media de las confianzas: ",media)
        # indexes = np.where(scores > media)[0]
        boxes = boxes[indexes]
        return {"boxes": boxes, "scores": scores[indexes], "classes": classes[indexes]}

    def __rescale_bboxes(self, predicted_bboxes, image_sizes):
        scaled_bboxes = []
        for bboxes, img_dims in zip(predicted_bboxes, image_sizes):
            im_h, im_w = img_dims

            if len(bboxes) > 0:
                scaled_bboxes.append(
                    (
                        np.array(bboxes)
                        * [
                            round(im_w / self.img_size,2),
                            round(im_h / self.img_size,2),
                            round(im_w / self.img_size,2),
                            round(im_h / self.img_size,2),
                        ]
                    ).tolist()
                )
            else:
                scaled_bboxes.append(bboxes)

        return scaled_bboxes

    # @torch.no_grad()
    # def prediction_step(self, batch, batch_idx):
    #     image, annotations, target, image_ids = batch
    #     # bboxes,_,_,loss = self.model.predict([image],target[0]['bboxes'])
    #     # print(f"Bboxes: {bboxes}\n\tLoss: {loss}\n")
    #     # output = self.model(image,annotations)
    #     output = self.model(image,annotations)
    #     self.log("loss",loss, on_step=True, on_epoch=True, prog_bar=True,
    #         logger=True, sync_dist=True)
    #     print("Output: "+output)
    #     return {'loss': loss}

    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        images, annotations, target, image_ids = batch
        pred_bboxes, pred_classes, pred_confs, loss = self.model.predict(images)
        self.log("prediction_loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                 logger=True, sync_dist=True)
        return {'loss': loss, 'batch_predictions': pred_bboxes}



# from fastcore.basics import patch

# @patch
# def aggregate_prediction_outputs(self: EfficientDetModel, outputs):

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

# from objdetecteval.metrics.coco_metrics import get_coco_stats

# @patch
# def validation_epoch_end(self: EfficientDetModel, outputs):
#     """Compute and log training loss and accuracy at the epoch level."""
#     print("Val\n")
#     validation_loss_mean = torch.stack(
#         [output["loss"] for output in outputs]
#     ).mean()
#     self.log("mean_val_loss", validation_loss_mean,on_step=False,on_epoch=True,prog_bar=False,
#         logger=True)

#     (
#         predicted_class_labels,
#         image_ids,
#         predicted_bboxes,
#         predicted_class_confidences,
#         targets,
#     ) = self.aggregate_prediction_outputs(outputs)

#     truth_image_ids = [target["image_id"].detach().item() for target in targets]
#     truth_boxes = [
#         target["bboxes"].detach()[:, [1, 0, 3, 2]].tolist() for target in targets
#     ] # convert to xyxy for evaluation
#     truth_labels = [target["labels"].detach().tolist() for target in targets]

#     stats = get_coco_stats(
#         prediction_image_ids=image_ids,
#         predicted_class_confidences=predicted_class_confidences,
#         predicted_bboxes=predicted_bboxes,
#         predicted_class_labels=predicted_class_labels,
#         target_image_ids=truth_image_ids,
#         target_bboxes=truth_boxes,
#         target_class_labels=truth_labels,
#     )['All']
#     # Logging de las estadísitcas de COCO
#     for k in stats.keys():
#         self.log(k,stats[k], on_step=False, on_epoch=True, prog_bar=False,
#                  logger=True)

#     return {"val_loss": validation_loss_mean, "metrics": stats}

# @patch
# def test_epoch_end(self: EfficientDetModel, outputs):
#     """Compute and log training loss and accuracy at the epoch level."""

#     test_loss_mean = torch.stack(
#         [output["loss"] for output in outputs]
#     ).mean()
#     self.log("mean_test_loss",test_loss_mean,on_step=False,on_epoch=True,prog_bar=False,
#         logger=True)

#     (
#         predicted_class_labels,
#         image_ids,
#         predicted_bboxes,
#         predicted_class_confidences,
#         targets,
#     ) = self.aggregate_prediction_outputs(outputs)

#     truth_image_ids = [target["image_id"].detach().item() for target in targets]
#     truth_boxes = [
#         target["bboxes"].detach()[:, [1, 0, 3, 2]].tolist() for target in targets
#     ] # convert to xyxy for evaluation
#     truth_labels = [target["labels"].detach().tolist() for target in targets]

#     stats = get_coco_stats(
#         prediction_image_ids=image_ids,
#         predicted_class_confidences=predicted_class_confidences,
#         predicted_bboxes=predicted_bboxes,
#         predicted_class_labels=predicted_class_labels,
#         target_image_ids=truth_image_ids,
#         target_bboxes=truth_boxes,
#         target_class_labels=truth_labels,
#     )['All']
#     # Logging de las estadísitcas de COCO
#     for k in stats.keys():
#         self.log(k,stats[k], on_step=False, on_epoch=True, prog_bar=False,
#                  logger=True)

#     return {"test_loss": test_loss_mean, "metrics": stats}

def get_model(layer=0):
    return freeze(EfficientDetModel(),layer)

def load_model(name,conf=0.2, skip=0.43):
    model = EfficientDetModel(num_classes=1, img_size=512, prediction_confidence_threshold=conf,
        skip_thr=skip)
    model.load_state_dict(torch.load(models_dir+"/"+name+".pt"))
    return model

def save_model(model, output):
    torch.save(model.state_dict(), f"{models_dir}/{output}.pt")


# Estructura de EfficientDet
# - EfficientDet
#     - EfficientNetFeatures
#     - BiFpn
#     - HeadNet <- cabeza de las bounding boxes
#     - HeadNet <- cabeza de las clases
# - Anchors
# - DetectionLoss
#
#
#

def freeze(model, layer_min=0, layer_max=4):
    """
    Localiza las últimas dos capas de la estructura principal de EfficientDet:
    las dos cabezas de predicción de bboxes y clases, para congelar
    los parámetros del resto de capas.
    bifpn = min=2, max=3
    bifpn + head = min=2, max=4
    head = min=3, max=4
    nada = min=0, max=4
    """
    layers = 0
    for children in model.children():
        for children2 in children.children():
            for child in children2.children():
                layers += 1
                if layers<layer_min or layers>layer_max:
                    for param in child.parameters():
                        param.requires_grad = False
