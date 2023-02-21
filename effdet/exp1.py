from pathlib import Path
import pandas as pd
# import model, dataset
from EffDetDataset import *
from Model import *
from Train import *

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

model = get_model()

load_ex_model(model,"modelos/ED_L_10ep_025iou_015conf.pt")
model.eval()
imgs = get_preds(model,test_ds,0,10)
import os
d = "test_044iou"
os.mkdir(d)
name = "ED_L_10e_044iou_02conf_test_"
for i in list(range(len(imgs))):
    imgs[i].save(f"{d}/{name}{i}.jpg")
