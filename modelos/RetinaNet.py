import torch
import torch.nn as n
from lightning.pytorch import LightningModule

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights

from torchvision.models.detection.anchor_utils import AnchorGenerator

from modelos.utils import image_sizes, compute_loss, threshold_fusion

from objdetecteval.metrics.coco_metrics import get_coco_stats
from objdetecteval.metrics.image_metrics import get_inference_metrics

from torchvision.ops import box_iou, boxes as box_ops
from torchvision.models.detection.retinanet import det_utils

from fastcore.basics import patch

from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from collections import OrderedDict

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium') 

models_dir = "./pths/"

class RetinaNetLightning(LightningModule):
    """
    Clase que contiene el funcionamiento básico de un modelo a efectos de entrenamiento/validación/test.
    Crea un modelo en concreto en base al backbone específicado.
    Versión del modelo simple sin umbralización.
    """
    def __init__(
        self,
        num_classes=1,
        lr=0.0002,
        threshold=0.1, #Threshold de scores de las bounding boxes
        iou_thr=0.3, #Threshold de IoU para considerarse la misma bounding box
    ):
        super().__init__()
        self.lr = lr
        self.model = retinanet_resnet50_fpn(
            weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT,
            weights_backbone = ResNet50_Weights.IMAGENET1K_V1
        )
        self.threshold = threshold
        self.iou_thr = iou_thr
        self.loss_fn = compute_loss
        self.model.num_classes = num_classes
        self.matcher = det_utils.Matcher(
            0.5,0.4,allow_low_quality_matches=True
        )
        anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
        self.anchor_generator = AnchorGenerator(
            anchor_sizes, ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        )
        self.transform = GeneralizedRCNNTransform(720, 1280, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
    # def forward(self, images, targets=None):
    #     outputs = self.model(images, targets)
    #     if not self.model.training or targets is None:
    #         outputs = threshold_fusion(
    #             outputs,
    #             image_sizes(images),
    #             iou_thr=self.iou_thr,
    #             skip_box_thr=self.threshold
    #         )
    #     return outputs

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def get_detections(self, images, features, head_outputs, anchors, original_image_sizes):
        num_anchors_per_level = [x.size(2) * x.size(3) for x in features]
        HW = 0
        for v in num_anchors_per_level:
            HW += v
        HWA = head_outputs["cls_logits"].size(1)
        A = HWA // HW
        num_anchors_per_level = [hw * A for hw in num_anchors_per_level]
        split_head_outputs: Dict[str, List[Tensor]] = {}
        for k in head_outputs:
            split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
        split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]
        detections = self.model.postprocess_detections(split_head_outputs, split_anchors, images.image_sizes)
        detections = self.model.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        return detections

    def forward(self, images, targets=None):
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
        images, targets = self.transform(images, targets)
        features = self.model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        features = list(features.values())
        head_outputs = self.model.head(features)
        anchors = self.anchor_generator(images, features)
        losses = self.model.compute_loss(targets, head_outputs, anchors)
        if self.model.training:
            return losses
        else:
            detections = self.get_detections(images, features, head_outputs, anchors, original_image_sizes)
            return {"loss": losses, "detections": detections}

    # def compute_single_loss(self, image, output : dict, target : dict):
    #     losses = {}
    #     boxes, gt_boxes = output['boxes'], target['boxes']
    #     labels, gt_labels = output['labels'], target['boxes']
    #     iou_matrix = box_iou(gt_boxes, boxes)
    #     matched_idx = self.matcher(iou_matrix)

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
            # 'predictions' : [output['boxes'] for output in outputs],
            'predictions' : outputs['detections'],
            'targets' : targets,
            'image_ids' : ids,
        }
        loss = outputs['loss']
        self.log('val_class_loss', loss['classification'].detach())
        self.log('val_box_loss', loss['bbox_regression'].detach())
        return {'loss' : loss, 'batch_predictions' : batch_predictions}

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        images, targets, ids = batch
        outputs = self(images, targets)
        batch_predictions = {
            # 'predictions' : [output['boxes'] for output in outputs],
            'predictions' : outputs['detections'],
            'targets' : targets,
            'image_ids' : ids,
        }
        loss = outputs['loss']
        self.log('test_class_loss', loss['classification'].detach())
        self.log('test_box_loss', loss['bbox_regression'].detach())
        return {'loss' : loss, 'batch_predictions' : batch_predictions}

@patch
def add_pred_outputs(self : RetinaNetLightning, outputs):
    boxes, scores, labels, image_ids, targets = [],[],[],[],[]
    for i in range(len(outputs['batch_predictions']['predictions'])):
        preds = outputs['batch_predictions']['predictions'][i]
        boxes.append(preds['boxes'])
        scores.append(preds['scores'])
        labels.append(preds['labels'])
        image_ids.append(outputs['batch_predictions']['image_ids'][i])
        targets.append(outputs['batch_predictions']['targets'][i])

    return (labels, image_ids, boxes, scores, targets)

@patch
def on_validation_step_end(self : RetinaNetLightning, utputs):
    """
    Añadido a cada etapa de validación en el que se evalúan los resultados
    del modelo con las estadísticas de COCO.
    """
    (labels, image_ids, boxes, scores, targets) = self.add_pred_outputs(outputs)
    truth_ids, truth_boxes, truth_labels = zip(
        *[
            (
                target['image_id'].detach().item(),
                target['boxes'].detach().tolist(),
                target["labels"].detach().tolist()
            ) for target in targets
        ]
    )
    stats = get_coco_stats(
        prediction_image_ids = image_ids,
        predicted_class_confidences = scores,
        predicted_bboxes = boxes,
        predicted_class_labels = labels,
        target_image_ids = truth_ids,
        target_bboxes = truth_boxes,
        target_class_labels = truth_labels,
    )['All']
    for k in stats.keys():
        self.log(k, stats[k], on_step=False, on_epoch=True, logger=True)
    return {'val_loss': outputs['loss'], 'metrics': stats}

def forward(model, images, targets=None):
    model.eval()
    original_image_sizes: List[Tuple[int, int]] = []
    for img in images:
        val = img.shape[-2:]
        assert len(val) == 2
        original_image_sizes.append((val[0], val[1]))
    images, targets = model.transform(images, targets)
    features = model.model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])
    features = list(features.values())
    head_outputs = model.model.head(features)
    anchors = model.anchor_generator(images, features)
    losses = model.model.compute_loss(targets, head_outputs, anchors)
    # if not model.training:
    detections = model.get_detections(images, features, head_outputs, anchors, original_image_sizes)
    # return {"loss": losses, "detections": detections}
    return images, features, head_outputs, anchors, original_image_sizes