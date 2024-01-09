import torch

#Parámetros de entrenamiento
NUM_CLASSES = 1
LR = 0.0002
NUM_EPOCHS = 50
IOU_THR = 0.7

#Parámetros del DataModule
DS_DIR = "../datasets/T1280x720/"
IMAGES_DIR = DS_DIR + "JPEGImages/"
IMAGESETS_DIR = DS_DIR + "ImageSets/"
MAIN_DS = IMAGESETS_DIR + "d801010/"
BATCH_SIZE = 4
TEST_BATCH_SIZE = 1
NUM_WORKERS = 8

#Cuda/CPU
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium') 

#Outputs
OUTPUTS_DIR = "./outputs"

#Modelos
MODELS_DIR = "./pths/"

#Logs
LOGS_DIR = "./logs/"

#MeanAP
KEYS = ['map','map_50','map_75','map_small','map_medium','map_large',
    'mar_1','mar_10','mar_100','mar_small','mar_medium','mar_large',
    'map_per_class', 'mar_100_per_class']