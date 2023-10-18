import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
import numpy as np

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights

from torchvision.ops import box_iou, sigmoid_focal_loss

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

from Visualize import 

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium') 

models_dir = "./pths/"

def RetinaNetLoss(predictions, targets, reduction="sum"):
    """
    Dados dos listas con diccionarios a tensores de bboxes, scores y labels respectivamente
    obtiene los mejores iou de las predicciones y las anotaciones (targets)
    y aplica sigmoid_focal_loss a cada predicción y las suma.
    """
    loss = 0.0
    for prediction, target in zip(predictions, targets):
        p_box, p_scores, p_labels = prediction['boxes'], prediction['scores'], prediction['labels']
        t_box, t_labels = target['boxes'], target['labels']
        # Tomamos las coincidencias de las predicciones en base a los targets
        iou = box_iou(t_box, p_box)
        # iou = box_iou(p_box, t_box)
        best_iou, best_preds = iou.max(dim=1)
        # best_iou, best_targets = iou.max(dim=1)
        loss += sigmoid_focal_loss(p_box[best_preds], t_box, reduction=reduction)
        # loss += sigmoid_focal_loss(p_box, t_box[best_targets], reduction=reduction)
        # loss += nn.SmoothL1Loss(p_box[best_preds], t_box)
    return loss

def l1_loss_smooth(predictions, targets, beta=1.0):
    loss = 0
    diff = predictions-targets
    mask = (diff.abs() < beta)
    loss += mask * (0.5*diff**2 / beta)
    loss += (~mask) * (diff.abs() - 0.5*beta)
    return loss.mean()

# def RetinaNetLoss(predictions, targets):
#     classification_losses = []
#     regression_losses = []

#     for pred, target in zip(predictions, targets):
#         (_,best_preds_labels), (_, best_preds_boxes) = (box_iou(pred['labels'],target['labels']).max(dim=1)), (box_iou(pred['boxes'],target['boxes']).max(dim=1))
#         iou_labels = box_iou(target['labels'],pred['labels'])
#         _, best_pred_labels = iou.max(dim=1) 
#         classification_losses.append(nn.CrossEntropyLoss(pred["labels"][best_preds_labels], target["labels"]))
        
#         regression_losses.append(nn.SmoothL1Loss(pred['boxes'][best_preds_boxes], target))
       
#     classification_loss = torch.mean(torch.stack(classification_losses))
#     regression_loss = torch.mean(torch.stack(regression_losses))

#     return classification_loss + regression_loss

class RetinaTomatoLightning(LightningModule):
    """
    Clase que contiene el funcionamiento básico de un modelo a efectos de entrenamiento/validación/test.
    Crea un modelo en concreto en base al backbone específicado.
    Versión del modelo simple sin umbralización.
    """
    def __init__(
        self,
        num_classes=1,
        lr=0.0002,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lr = lr
        self.model = retinanet_resnet50_fpn(
            weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT,
            weights_backbone = ResNet50_Weights.DEFAULT
        )
        # self.loss_fn = sigmoid_loss
        self.loss_fn = RetinaNetLoss

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        # images, targets = batch
        images, targets, ids = batch
        loss = self(images, targets)
        # Registramos el error de clasificación y regresión de bbox
        # self.log('classification', loss['classification'].detach())
        self.log('train_loss', loss['bbox_regression'].detach())
        return {'loss' : loss['bbox_regression']}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, targets, ids = batch
        outputs = self(images, targets)
        batch_predictions = {
            'predictions' : [output['boxes'] for output in outputs],
            'targets' : targets,
            'image_ids' : ids,
        }
        loss = self.loss_fn(outputs, targets)
        self.log('val_loss', loss)
        return {'loss' : loss, 'batch_predictions' : batch_predictions}

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        images, targets, ids = batch
        outputs = self(images, targets)
        batch_predictions = {
            'predictions' : [output['boxes'] for output in outputs],
            'targets' : targets,
            'image_ids' : ids,
        }
        loss = self.loss_fn(outputs, targets)
        self.log('test_loss', loss)
        return {'loss' : loss, 'batch_predictions' : batch_predictions}
    
    @torch.no_grad()
    def predict(self, batch, batch_idx):
        #Asumimos que estas imágenes residen en un dataloader (test)
        images, targets, ids = batch
        outputs = self(images, targets)
        loss = self.loss_fn(outputs, targets)
        print("Loss: ",loss)





def freeze_modules(model, modules=["regression_head"]):
    for name, module in model.named_modules():
        if any(module_name in name for module_name in modules):
            for param in module.parameters():
                param.requires_grad = False
            if len(list(module.children()))>0:
                freeze_modules(module, modules)

def model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

# def inference(model, dl, output_dir):
#     images_names = dl.dataset.ds.images
#     model.eval()

def load_model(path, model=None):
    if model is None:
        model = RetinaTomatoLightning()
    model.load_state_dict(torch.load(path))
    return model

def save_model(model, name):
    torch.save(model.state_dict(), f"{models_dir}/{name}.pt")


logger = TensorBoardLogger("./logs","retinanet")
if torch.cuda.is_available():
    trainer = Trainer(
        accelerator="cuda", 
        devices=1,
        max_epochs=30, 
        num_sanity_val_steps=1, 
        logger=logger
    )
