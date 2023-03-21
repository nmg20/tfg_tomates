from effdet.config.model_config import efficientdet_model_param_dict
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet
from effdet.config.model_config import efficientdet_model_param_dict

import timm
from EffDetDataset import *
# import EffDetDataset
from torchmetrics.detection import MAP
import matplotlib.pyplot as plt

def create_model(num_classes=1, image_size=512, architecture="tf_efficientnetv2_l"):
    efficientdet_model_param_dict[architecture] = dict(
        name=architecture,
        backbone_name=architecture,
        backbone_args=dict(drop_path_rate=0.2),
        num_classes=num_classes,
        url='', )
    
    config = get_efficientdet_config(architecture)
    config.update({'num_classes': num_classes})
    config.update({'image_size': (image_size, image_size)})
    
    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(
        config,
        num_outputs=config.num_classes,
    )
    return DetBenchTrain(net, config)

from numbers import Number
from typing import List
from functools import singledispatch

import logging
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from torchvision.ops import box_iou
from sklearn.metrics import average_precision_score

from fastcore.dispatch import typedispatch
from pytorch_lightning import LightningModule
from pytorch_lightning.core.decorators import auto_move_data

from ensemble_boxes import ensemble_boxes_wbf

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
        prediction_confidence_threshold=0.2,
        learning_rate=0.0002,
        wbf_iou_threshold=0.44,
        inference_transforms=get_valid_transforms(target_img_size=512),
        model_architecture='tf_efficientnetv2_l',
    ):
        super().__init__()
        self.img_size = img_size
        self.model = create_model(
            num_classes, img_size, architecture=model_architecture
        )
        # self.dropout = nn.Dropout()
        self.prediction_confidence_threshold = prediction_confidence_threshold
        self.lr = learning_rate
        self.wbf_iou_threshold = wbf_iou_threshold
        self.inference_tfms = inference_transforms
        self.metric = MAP()

    @auto_move_data
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

        # print("\n\n",losses,"\n\n")

        # loss = F.cross_entropy(losses['loss'],torch.tensor(annotations))

        self.log("train_loss", losses["loss"], on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)
        # self.log(
        #     "train_class_loss", losses["class_loss"], on_step=True, on_epoch=True, prog_bar=True,
        #     logger=True
        # )
        self.log("train_box_loss", losses["box_loss"], on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)

        return losses['loss']

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, annotations, targets, image_ids = batch
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
        # self.log(
        #     "valid_class_loss", logging_losses["class_loss"], on_step=True, on_epoch=True,
        #     prog_bar=True, logger=True, sync_dist=True
        # )
        self.log("valid_box_loss", logging_losses["box_loss"], on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)

        return {'loss': outputs["loss"], 'batch_predictions': batch_predictions}
    
    # Reutilizo el paso de validación
    # @torch.no_grad()
    # def test_step(self, batch, batch_idx):
    #     images, annotations, targets, image_ids = batch
    #     outputs = self.model(images, annotations)
    #     detections = outputs["detections"]
    #     batch_predictions = {
    #         "predictions": detections,
    #         "targets": targets,
    #         "image_ids": image_ids,
    #     }
    #     logging_losses = {
    #         "class_loss": outputs["class_loss"].detach(),
    #         "box_loss": outputs["box_loss"].detach(),
    #     }
    #     self.log("test_loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True,
    #              logger=True, sync_dist=True)
    #     self.log("test_box_loss", logging_losses["box_loss"], on_step=True, on_epoch=True,
    #              prog_bar=True, logger=True, sync_dist=True)

    #     return {'loss': outputs["loss"], 'batch_predictions': batch_predictions}
    def test_step(self, batch, batch_idx):
        """
        pred -> lista de diccionarios. 1 dict x img predecida
            dict => 'boxes' = preds[0]
                    'labels' = preds[1]  -> todos FloatTensor
                    'scores' = preds[2]
        target -> lista de dicts. 1 dict x img.
            dict => 'boxes' = targets['bboxes']
                    'labels' = targets['labels']
        """
        images, annotations, targets, _ = batch
        boxes, labels, scores =  model.predict(images)
        pred = dict()
        self.metric
        
        

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
        dummy_targets = self._create_dummy_inference_targets(
            num_images=images_tensor.shape[0]
        )

        detections = self.model(images_tensor.to(self.device), dummy_targets)[
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

        return scaled_bboxes, predicted_class_labels, predicted_class_confidences
    
    def _create_dummy_inference_targets(self, num_images):
        dummy_targets = {
            "bbox": [
                torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=self.device)
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
            predictions, image_size=self.img_size, iou_thr=self.wbf_iou_threshold
        )

        return predicted_bboxes, predicted_class_confidences, predicted_class_labels

    def _postprocess_single_prediction_detections(self, detections):
        boxes = detections.detach().cpu().numpy()[:, :4]
        scores = detections.detach().cpu().numpy()[:, 4]
        classes = detections.detach().cpu().numpy()[:, 5]
        indexes = np.where(scores > self.prediction_confidence_threshold)[0]
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
                            im_w / self.img_size,
                            im_h / self.img_size,
                            im_w / self.img_size,
                            im_h / self.img_size,
                        ]
                    ).tolist()
                )
            else:
                scaled_bboxes.append(bboxes)

        return scaled_bboxes

from fastcore.basics import patch

@patch
def aggregate_prediction_outputs(self: EfficientDetModel, outputs):

    detections = torch.cat(
        [output["batch_predictions"]["predictions"] for output in outputs]
    )

    image_ids = []
    targets = []
    for output in outputs:
        batch_predictions = output["batch_predictions"]
        image_ids.extend(batch_predictions["image_ids"])
        targets.extend(batch_predictions["targets"])

    (
        predicted_bboxes,
        predicted_class_confidences,
        predicted_class_labels,
    ) = self.post_process_detections(detections)

    return (
        predicted_class_labels,
        image_ids,
        predicted_bboxes,
        predicted_class_confidences,
        targets,
    )

from objdetecteval.metrics.coco_metrics import get_coco_stats

@patch
def validation_epoch_end(self: EfficientDetModel, outputs):
    """Compute and log training loss and accuracy at the epoch level."""

    validation_loss_mean = torch.stack(
        [output["loss"] for output in outputs]
    ).mean()

    (
        predicted_class_labels,
        image_ids,
        predicted_bboxes,
        predicted_class_confidences,
        targets,
    ) = self.aggregate_prediction_outputs(outputs)

    truth_image_ids = [target["image_id"].detach().item() for target in targets]
    truth_boxes = [
        target["bboxes"].detach()[:, [1, 0, 3, 2]].tolist() for target in targets
    ] # convert to xyxy for evaluation
    truth_labels = [target["labels"].detach().tolist() for target in targets]

    stats = get_coco_stats(
        prediction_image_ids=image_ids,
        predicted_class_confidences=predicted_class_confidences,
        predicted_bboxes=predicted_bboxes,
        predicted_class_labels=predicted_class_labels,
        target_image_ids=truth_image_ids,
        target_bboxes=truth_boxes,
        target_class_labels=truth_labels,
    )['All']
    for k in stats.keys():
        self.log(k,stats[k], on_step=False, on_epoch=True, prog_bar=False,
                 logger=True)

    return {"val_loss": validation_loss_mean, "metrics": stats}

def load_checkpoint(path):
    return EfficientDetModel.load_from_checkpoint(path)

def load_model(path):
    model = EfficientDetModel(num_classes=1, img_size=512)
    return model.load_state_dict(torch.load(path))

def load_ex_model(model, path):
    model.load_state_dict(torch.load(path))

# def get_batch(model,ds,i):
#     img, anots, 

def get_imgs_anots_preds(model,ds,i,j):
    imgs,anots = [],[]
    for k in list(range(i,j)):
        img, anot,_,_ = ds.get_image_and_labels_by_idx(k)
        imgs.append(img)
        anots.append(anot)
    pred, _, _ = model.predict(imgs)
    return imgs,anots,pred

def get_preds(model,ds,i,j):
    imgs, anots, preds = get_imgs_anots_preds(model,ds,i,j)
    pred_imgs = []
    for k in list(range(len(imgs))):
        pred_imgs.append(get_img_drawn(imgs[k],anots[k],preds[k]))
    return pred_imgs

def get_pred(model, ds, i):
    img, box, _, _ = ds.get_image_and_labels_by_idx(i)
    pred, _ ,_ = model.predict([img])
    plt.figure()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30,30))
    ax1.imshow(img)
    ax1.set_title("Predicción")
    ax2.imshow(img)
    ax2.set_title("Anotada")
    draw_pascal_voc_bboxes(ax1, pred[0])
    draw_pascal_voc_bboxes(ax2, box.tolist())
    # plt.savefig(path+Path(imgs[i].filename).name)
    fig.canvas.draw()
    image = Image.frombytes('RGB', 
    fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
    plt.close("all")
    return image

def format_anots(anots,max_bbs):
    """
    Añade padding a los arrays de anotaciones de cada imagen para
    poder convertir la lista a tensor.
    anots : lista de np.arrays
    """
    if max_bbs==0:
        max_bbs = len(max(anots,key=len))
    padded = []
    for anot in anots:
        padded.append(np.pad(anot,[(0,(max_bbs-len(anot))),(0,0)]))
    return padded

def format_preds(preds,max_bbs):
    preds_array = []
    for pred in preds:
        preds_array.append(np.array(pred))
    return format_anots(preds_array,max_bbs)

def format_inference(anots,preds):
    max_bbs = len(max(anots,key=len))
    anots_tensor = torch.tensor(format_anots(anots,0))
    preds_tensor = torch.tensor(format_preds(preds,max_bbs))
    return anots_tensor, preds_tensor

def format_tensor(anots,preds):
    max_cols = max([len(row) for batch in anots for row in batch])
    max_rows = max([len(batch) for batch in anots])
    padded_anots = [batch + [[0] * (max_cols)] * (max_rows - len(batch)) for batch in anots]
    padded_preds = [batch + [[0] * (max_cols)] * (max_rows - len(batch)) for batch in preds]
    padded_anots = torch.tensor([row + [0] * (len(target) - len(row)) for batch in padded_anots for row in batch])
    padded_preds = torch.tensor([row + [0] * (len(target) - len(row)) for batch in padded_preds for row in batch])
    padded_anots = padded_anots.view(-1, max_rows, max_cols)
    padded_preds = padded_preds.view(-1, max_rows, max_cols)
    return padded_anots, padded_preds

def format_anots(anots):
    max_cols = max([len(row) for batch in anots for row in batch])
    max_rows = max([len(batch) for batch in anots])
    padded_anots = [batch + [[0] * (max_cols)] * (max_rows - len(batch)) for batch in anots]
    # padded_preds = [batch + [[0] * (max_cols)] * (max_rows - len(batch)) for batch in preds]
    padded_anots = torch.tensor([row + [0] * (len(target) - len(row)) for batch in padded_anots for row in batch])
    # padded_preds = torch.tensor([row + [0] * (len(target) - len(row)) for batch in padded_preds for row in batch])
    padded_anots = padded_anots.view(-1, max_rows, max_cols)
    # padded_preds = padded_preds.view(-1, max_rows, max_cols)
    return padded_anots