import numpy as np
import torch

from lightning.pytorch import LightningModule
from ensemble_boxes import ensemble_boxes_wbf
import effdet
from TomatoDataset import get_effdet_transforms
from utils import images_sizes

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

class EfficientDetModel(LightningModule):
    def __init__(
        self,
        num_classes=config.NUM_CLASSES,
        img_size=512,
        threshold=0.2,
        lr=config.LR,
        iou_thr=config.IOU_THR,
    ):
        super().__init__()
        self.img_size = img_size
        self.model = create_model()
        self.threshold = threshold
        self.lr = lr
        self.iou_thr = iou_thr
        self.val_step_outputs = []
        self.val_step_targets = []
        self.test_step_outputs = []
        self.test_step_targets = []
    
    def forward(self, images, targets=None):
        outputs = self.model(images, targets)
        if self.model.training or targets is None:
            outputs = self.inference(outputs, images, targets)
        return outputs

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        images, annotations, _, image_ids = batch
        losses = self.model(images, annotations)

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
        outputs = self.model(images, annotations)
        logging_losses = {
            "class_loss": outputs["class_loss"].detach(),
            "box_loss": outputs["box_loss"].detach(),
        }
        return {'loss': outputs["loss"], 'batch_predictions': batch_predictions}
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        image, annotations, targets, image_ids = batch
        outputs = self.model(image, annotations)
        losses = [outputs["loss"],outputs["box_loss"]]
        logging_losses = {"box_loss": outputs["box_loss"].detach(),}
        return {'loss': outputs["loss"], 'batch_predictions': batch_predictions}

    def predict(self, images_tensor):
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

    def _run_inference(self, outputs, images, targets):
        """
        Images = Tensor de 3 canales del que extraer los tamaños.
        """
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
            predicted_bboxes=predicted_bboxes, image_sizes=image_sizes(images)
        )
        return scaled_bboxes, predicted_class_labels, predicted_class_confidences, loss
    
    def post_process_detections(self, detections):
        predictions = []
        for i in range(len(detections)):
            predictions.append(
                self._postprocess_single_prediction_detections(detections[i])
            )
        predicted_bboxes, predicted_class_confidences, predicted_class_labels = run_wbf(
            predictions, image_size=self.img_size, iou_thr=self.wbf_iou_threshold, skip_box_thr=self.skip_thr
        )
        return predicted_bboxes, predicted_scores, predicted_labels

    def _postprocess_single_prediction_detections(self, detections):
        boxes = detections.detach().cpu().numpy()[:, :4]
        scores = detections.detach().cpu().numpy()[:, 4]
        classes = detections.detach().cpu().numpy()[:, 5]
        return {"boxes": boxes, "scores": scores, "labels": classes}

    def __rescale_bboxes(self, predicted_bboxes, image_sizes):
        """
        Pasa imágenes de 512x512 a su tamaño original.
        """
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

    def run_wbf(self, predictions):
        bboxes, scores, labels = [], [], []
        for prediction in predictions:
            boxes, pred_scores, pred_labels = ensemble_boxes_wbf.weighted_boxes_fusion(
                [(prediction["boxes"] / self.image_size).tolist()],
                [prediction["scores"].tolist()],
                [prediction["labels"].tolist()],
                iou_thr=self.iou_thr,
                skip_box_thr=self.threshold,
            )
            boxes = boxes * (self.image_size - 1)
            bboxes.append(boxes.tolist())
            confidences.append(pred_scores.tolist())
            labels.append(pred_labels.tolist())
        return bboxes, scores, labels


