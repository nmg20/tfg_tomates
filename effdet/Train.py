from pathlib import Path
import pandas as pd
# import model, dataset
from EffDetDataset import *
from Model import *
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision

writer = SummaryWriter('lightning_logs/tomato_exp1')

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
        iou=0.3,
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

model.load_state_dict(torch.load("modelos/ED_20ep_0.3iou_0.2cf.pt"))

# model.eval()

loader = dm.val_dataloader()
dl_iter = iter(loader)
# batch = next(dl_iter)
images, targets, _, _ = next(dl_iter)
device = model.device;device
# output = model.validation_step(batch=batch,batch_idx=0)
# output

img_grid = torchvision.utils.make_grid(images)
writer.add_image('tomato_images', img_grid)