from pathlib import Path
import pandas as pd
import TomatoDatasetAdaptor, DataModule, Model, TrainingLoop

dataset_path = Path("../../../tomates512/")
train_data_path = dataset_path/"images/train/"
test_data_path = dataset_path/"images/test/"
val_data_path = dataset_path/"images/val/"

df_tr = pd.read_csv(dataset_path/"annotations/labelstrain.csv")
df_ts = pd.read_csv(dataset_path/"annotations/labelstest.csv")
df_vl = pd.read_csv(dataset_path/"annotations/labelsval.csv")

tomato_train_ds = TomatoDatasetAdaptor.TomatoDatasetAdaptor(train_data_path, df_tr)
tomato_test_ds = TomatoDatasetAdaptor.TomatoDatasetAdaptor(test_data_path, df_ts)
tomato_val_ds = TomatoDatasetAdaptor.TomatoDatasetAdaptor(val_data_path, df_vl)

############################

dm = DataModule.EfficientDetDataModule(train_dataset_adaptor=tomato_train_ds, 
        validation_dataset_adaptor=tomato_train_ds,
        num_workers=4,
        batch_size=2)

model = Model.EfficientDetModel(
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
trainer.fit(model,dm)

########################################
# Comprobamos las predicciones
########################################

img1, truth_bboxes1, _, _ = tomato_train_ds.get_image_and_labels_by_idx(0)
img2, truth_bboxes2, _, _ = tomato_train_ds.get_image_and_labels_by_idx(1)

images = [img1, img2]

predicted_bboxes, predicted_class_confidences, predicted_class_labels = model.predict(images)

def compare_bboxes_for_image(
    image,
    predicted_bboxes,
    actual_bboxes,
    draw_bboxes_fn=draw_pascal_voc_bboxes,
    figsize=(20, 20),
    ):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.imshow(image)
    ax1.set_title("Prediction")
    ax2.imshow(image)
    ax2.set_title("Actual")

    draw_bboxes_fn(ax1, predicted_bboxes)
    draw_bboxes_fn(ax2, actual_bboxes)

    plt.show()

compare_bboxes_for_image(img1, predicted_bboxes=predicted_bboxes[0],actual_bboxes=truth_bboxes1.tolist())
