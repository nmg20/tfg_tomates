from pathlib import Path
import pandas as pd
# import model, dataset
from EffDetDataset import *
from Model import *
import torch
import torchvision
from pytorch_lightning import Trainer 

from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter('lightning_logs/tomato_exp1')

dataset_path = Path("../../tomates512/")
df_tr = pd.read_csv(dataset_path/"annotations/labelstrain.csv")
df_ts = pd.read_csv(dataset_path/"annotations/labelstest.csv")
df_vl = pd.read_csv(dataset_path/"annotations/labelsval.csv")

train_ds = TomatoDatasetAdaptor(dataset_path/"images/train/", df_tr)
test_ds = TomatoDatasetAdaptor(dataset_path/"images/test/", df_ts)
val_ds = TomatoDatasetAdaptor(dataset_path/"images/val/", df_vl)

############################

dm = EfficientDetDataModule(train_dataset_adaptor=train_ds, 
        validation_dataset_adaptor=val_ds,
        num_workers=4,
        batch_size=2)

############################

# def get_model(
#         iou=0.44,
#         skip=0.43,
#         confd=0.2):
#     return EfficientDetModel(
#         wbf_iou_threshold=iou,
#         skip_box_thr=skip,
#         prediction_confidence_threshold=confd)

def get_model():
    return EfficientDetModel()

model = get_model()

def train_model(model=model, dm=dm,num_epochs=5):
    trainer = Trainer(
        gus=1, num_epochs=num_epochs,num_sanity_val_steps=1)
    trainer.fit(model,dm)

########################################
# Entrenamiento
########################################



# trainer = Trainer(
#         gpus=1, max_epochs=5, num_sanity_val_steps=1,
#     )
# trainer.fit(model,dm)
# torch.save(model.state_dict(),f"{model_architecture}_{wbf_iou_threshold}iou_{prediction_confidence_threshold}confidence.pth")

# model.load_state_dict(torch.load("modelos/ED_20ep_0.3iou_0.2cf.pt"))
model.load_state_dict(torch.load("modelos/ED_20ep_0.44iou_0.2cf.pt"))

# model.eval()

# loader = dm.val_dataloader()
# dl_iter = iter(loader)
# batch = next(dl_iter)
# images, targets, _, _ = next(dl_iter)
# device = model.device;device
# output = model.validation_step(batch=batch,batch_idx=0)
# output

# img_grid = torchvision.utils.make_grid(images)
# writer.add_image('tomato_images', img_grid)