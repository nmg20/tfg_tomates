from pathlib import Path
import pandas as pd
# import model, dataset
from EffDetDataset import EfficientDetDataModule, TomatoDatasetAdaptor
from Model import EfficientDetModel
import torch

dataset_path = Path("../../tomates512/")
train_data_path = dataset_path/"images/train/"
test_data_path = dataset_path/"images/test/"
val_data_path = dataset_path/"images/val/"

df_tr = pd.read_csv(dataset_path/"annotations/labelstrain.csv")
df_ts = pd.read_csv(dataset_path/"annotations/labelstest.csv")
df_vl = pd.read_csv(dataset_path/"annotations/labelsval.csv")

tomato_train_ds = TomatoDatasetAdaptor(train_data_path, df_tr)
tomato_test_ds = TomatoDatasetAdaptor(test_data_path, df_ts)
tomato_val_ds = TomatoDatasetAdaptor(val_data_path, df_vl)

############################

dm = EfficientDetDataModule(train_dataset_adaptor=tomato_train_ds, 
        validation_dataset_adaptor=tomato_train_ds,
        num_workers=4,
        batch_size=2)

model_architecture="tf_efficientnetv2_b0"
wbf_iou_threshold=0.1
prediction_confidence_threshold=0.1

model = EfficientDetModel(
    model_architecture=model_architecture,
    wbf_iou_threshold=wbf_iou_threshold,
    prediction_confidence_threshold=prediction_confidence_threshold
    )

########################################
# Entrenamiento
########################################

from pytorch_lightning import Trainer 

trainer = Trainer(
        gpus=1, max_epochs=5, num_sanity_val_steps=1,
    )
# trainer.fit(model,dm)
# torch.save(model.state_dict(),f"{model_architecture}_{wbf_iou_threshold}iou_{prediction_confidence_threshold}confidence.pth")

