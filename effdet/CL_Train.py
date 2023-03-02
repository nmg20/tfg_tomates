from pathlib import Path
import pandas as pd
# import model, dataset
from EffDetDataset import *
from Model import *
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
import os

models_dir = "modelos/"

def validate_file(f):
    if not os.path.exists(f):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

def load_dss(path):
    dataset_path = Path(path)
    train_data_path = dataset_path/"images/train/"
    test_data_path = dataset_path/"images/test/"
    val_data_path = dataset_path/"images/val/"
    df_tr = pd.read_csv(dataset_path/"annotations/labelstrain.csv")
    df_ts = pd.read_csv(dataset_path/"annotations/labelstest.csv")
    df_vl = pd.read_csv(dataset_path/"annotations/labelsval.csv")

    train_ds = TomatoDatasetAdaptor(train_data_path, df_tr)
    test_ds = TomatoDatasetAdaptor(test_data_path, df_ts)
    val_ds = TomatoDatasetAdaptor(val_data_path, df_vl)
    return train_ds, test_ds, val_ds

# def get_dm(train,val):
    # return EfficientDetDataModule(train_dataset_adaptor=train, 
    #     validation_dataset_adaptor=val,
    #     num_workers=4,
    #     batch_size=2)

def get_dm(path):
    dataset_path = Path(path)
    train_data_path = dataset_path/"images/train/"
    test_data_path = dataset_path/"images/test/"
    val_data_path = dataset_path/"images/val/"
    df_tr = pd.read_csv(dataset_path/"annotations/labelstrain.csv")
    df_ts = pd.read_csv(dataset_path/"annotations/labelstest.csv")
    df_vl = pd.read_csv(dataset_path/"annotations/labelsval.csv")

    train_ds = TomatoDatasetAdaptor(train_data_path, df_tr)
    test_ds = TomatoDatasetAdaptor(test_data_path, df_ts)
    val_ds = TomatoDatasetAdaptor(val_data_path, df_vl)
    return EfficientDetDataModule(train_dataset_adaptor=train_ds, 
        validation_dataset_adaptor=val_ds,
        test_dataset_adaptor=test_ds,
        num_workers=4,
        batch_size=2)

############################

# dm = EfficientDetDataModule(train_dataset_adaptor=train_ds, 
#         validation_dataset_adaptor=val_ds,
#         num_workers=4,
#         batch_size=2)

# torch.save(model.state_dict(),model_name)

# model.load_state_dict(torch.load("modelos/ed_l_025iou_015conf_5epch.pt"))

def main():
    # python train.py -a l -e 10 -t 0.44 -cf 0.2 
    #       -load modelos/modelo.pth -save modelos/modelo.pth
    parser = argparse.ArgumentParser()
    # parser.add_argument('-a', '--architecture', help="Tipo de arquitectura del modelo",
        # type=str)
    # parser.add_argument('-d', '--d', type=int, help="Número de épocas de entrenamiento.")
    parser.add_argument('-d','--dataset',type=dir_path, help="Directorio del dataset a partir del que crear el dataframe.")
    parser.add_argument('-e', '--epochs', type=int, help="Número de épocas de entrenamiento.")
    # parser.add_argument('-t','--thresh',type=float, help="Umbral de wbf_iou.")
    # parser.add_argument('-sk','--skip',type=float, help="Nivel de iou mínimo en la predicción.")
    # parser.add_argument('-cd','--predconfd', type=float, help="idk man.")
    # parser.add_argument('-s','--save', type=int, help="Nombre con el que guardar el modelo entrenado.")
    # parser.add_argument('-n','--name', type=str, help="Nombre con el que guardar el modelo entrenado.")
    parser.add_argument('-n','--name',type=str)
    args = parser.parse_args()
    # if args.d :
    #     model = EfficientDetModel(
    #     model_architecture="tf_efficientnetv2_b0")
    # else:
    dm = get_dm(args.dataset)
    model = EfficientDetModel(
        # model_architecture=f"tf_efficientnetv2_{args.architecture}",
        # wbf_iou_threshold=args.thresh,
        # skip_box_thr=args.skip,
        # prediction_confidence_threshold=args.predconfd
        )
    # logger = TensorBoardLogger("lightning_logs/",name=f"ED_{args.epochs}ep_{args.name}")
    
    model_name = f"ED_{args.epochs}ep"
    if args.name!="no":
        model_name = args.name+model_name
    logger = TensorBoardLogger("lightning_logs/",name=model_name)
    trainer = Trainer(
        gpus = 1, max_epochs=args.epochs, num_sanity_val_steps=1, logger=logger,
    )
    trainer.fit(model,dm)

    torch.save(model.state_dict(),f"{models_dir}{model_name}.pt")



if __name__=="__main__":
    main()