from pathlib import Path
import pandas as pd
# import model, dataset
from dataset import EfficientDetDataModule
from model import EfficientDetModel

dataset_path = Path("../../tomates512/")
train_data_path = dataset_path/"images/train/"
test_data_path = dataset_path/"images/test/"
val_data_path = dataset_path/"images/val/"

df_tr = pd.read_csv(dataset_path/"annotations/labelstrain.csv")
df_ts = pd.read_csv(dataset_path/"annotations/labelstest.csv")
df_vl = pd.read_csv(dataset_path/"annotations/labelsval.csv")

tomato_train_ds = dataset.TomatoDatasetAdaptor(train_data_path, df_tr)
tomato_test_ds = dataset.TomatoDatasetAdaptor(test_data_path, df_ts)
tomato_val_ds = dataset.TomatoDatasetAdaptor(val_data_path, df_vl)

############################

dm = EfficientDetDataModule(train_dataset_adaptor=tomato_train_ds, 
        validation_dataset_adaptor=tomato_train_ds,
        num_workers=4,
        batch_size=2)

model = EfficientDetModel(
    num_classes=1,
    img_size=512
    )

########################################
# Entrenamiento
########################################

from pytorch_lightning import Trainer 

trainer = Trainer(
        gpus=[0], max_epochs=1, num_sanity_val_steps=1,
    )
# trainer.fit(model,dm)
