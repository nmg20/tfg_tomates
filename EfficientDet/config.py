import torch

NUM_CLASSES = 1
LR = 0.0002
NUM_EPOCHS = 50
IOU_THR = 0.7

DS_DIR = "../../datasets/T1280x720/"
IMAGES_DIR = DS_DIR + "JPEGImages/"
IMAGESETS_DIR = DS_DIR + "ImageSets/"
MAIN_DS = IMAGESETS_DIR + "d801010/"
BATCH_SIZE = 4
TEST_BATCH_SIZE = 1
NUM_WORKERS = 8

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium') 

#Outputs
OUTPUTS_DIR = "./outputs"

#Modelos
MODELS_DIR = "../pths/"

#Logs
LOGS_DIR = "./logs/"