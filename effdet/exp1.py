from pathlib import Path
import pandas as pd
# import model, dataset
from EffDetDataset import *
from Model import *
# from Train import *
import os
import argparse

# dataset_path = Path("../../tomates512/")
# df_tr = pd.read_csv(dataset_path/"annotations/labelstrain.csv")
# df_ts = pd.read_csv(dataset_path/"annotations/labelstest.csv")
# df_vl = pd.read_csv(dataset_path/"annotations/labelsval.csv")

# train_ds = TomatoDatasetAdaptor(dataset_path/"images/train/", df_tr)
# test_ds = TomatoDatasetAdaptor(dataset_path/"images/test/", df_ts)
# val_ds = TomatoDatasetAdaptor(dataset_path/"images/val/", df_vl)

############################

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

models_dir = "modelos/"
preds_dir = "preds/"

############################

def save_preds_m(model,e,ds,i,j,div):
    model.eval()
    imgs = get_preds(model,ds,i,j)
    d = f"{preds_dir}{div}preds_{e}epochs"
    name = f"{div}ED_{e}ep_test_"
    if not os.path.exists(d):
        os.mkdir(d)
    else:
        name = "alt_"+name
    for i in list(range(len(imgs))):
        imgs[i].save(f"{d}/{name}{i}.jpg")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataset',type=dir_path, help="Directorio del dataset a partir del que crear el dataframe.")
    # parser.add_argument('-m', '--model', type=int, help="Número de épocas de entrenamiento.")
    parser.add_argument('-e', '--eps', type=int, help="Número de épocas de entrenamiento.")
    # parser.add_argument('-p', '--prefix', type=str, help="Número de épocas de entrenamiento.")
    parser.add_argument('-div','--div',type=str)
    args = parser.parse_args()

    model = EfficientDetModel()
    if args.div!=0:
        d = f"{models_dir}t{args.div}ED_{args.eps}ep.pt"
    else:
        d = f"{models_dir}ED_{args.eps}ep.pt"
    model.load_state_dict(torch.load(d))
    train_ds, test_ds, val_ds = load_dss(args.dataset)
    save_preds_m(model,args.eps,test_ds,0,5,args.div)

if __name__=="__main__":
    main()