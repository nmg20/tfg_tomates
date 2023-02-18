from pathlib import Path
import pandas as pd
# import model, dataset
from EffDetDataset import EfficientDetDataModule, TomatoDatasetAdaptor
from Model import EfficientDetModel, load_ex_model, get_pred
import torch

dataset_path = Path("../../tomates512/")
train_data_path = dataset_path/"images/train/"
test_data_path = dataset_path/"images/test/"
val_data_path = dataset_path/"images/val/"

df_tr = pd.read_csv(dataset_path/"annotations/labelstrain.csv")
df_ts = pd.read_csv(dataset_path/"annotations/labelstest.csv")
df_vl = pd.read_csv(dataset_path/"annotations/labelsval.csv")

train_ds = TomatoDatasetAdaptor(train_data_path, df_tr)
test_ds = TomatoDatasetAdaptor(test_data_path, df_ts)
val_ds = TomatoDatasetAdaptor(val_data_path, df_vl)

############################

dm = EfficientDetDataModule(train_dataset_adaptor=train_ds, 
        validation_dataset_adaptor=val_ds,
        num_workers=4,
        batch_size=2)

model_architecture="tf_efficientnetv2_l"
wbf_iou_threshold=0.25
prediction_confidence_threshold=0.15

model = EfficientDetModel(
    model_architecture=model_architecture,
    wbf_iou_threshold=wbf_iou_threshold,
    prediction_confidence_threshold=prediction_confidence_threshold
    )

############################

def get_model(
        arch="tf_efficientnetv2_l",
        iou=0.44,
        confd=0.2):
    return EfficientDetModel(model_architecture=arch,
        wbf_iou_threshold=iou,
        prediction_confidence_threshold=confd)

def train_model(model=model, dm=dm,num_epochs=5):
    trainer = Trainer(
        gus=1, num_epochs=num_epochs,num_sanity_val_steps=1)
    trainer.fit(model,dm)


########################################
# Entrenamiento
########################################

from pytorch_lightning import Trainer 

# trainer = Trainer(
#         gpus=1, max_epochs=5, num_sanity_val_steps=1,
#     )
# trainer.fit(model,dm)
# torch.save(model.state_dict(),f"{model_architecture}_{wbf_iou_threshold}iou_{prediction_confidence_threshold}confidence.pth")

model.load_state_dict(torch.load("modelos/ed_l_025iou_015conf_5epch.pt"))

model.eval()

loader = dm.val_dataloader()
dl_iter = iter(loader)
batch = next(dl_iter)
device = model.device;device
output = model.validation_step(batch=batch,batch_idx=0)
output


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

    return {"val_loss": validation_loss_mean, "metrics": stats}