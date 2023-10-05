import os
datasets_dir = os.path.abspath("../datasets/")
# main_ds = "/media/rtx3090/Disco2TB/cvazquez/nico/datasets/Tomato_1280x720/"
# main_ds = "/media/rtx3090/Disco2TB/cvazquez/nico/datasets/T1280x720_test/"
main_ds = os.path.abspath("../datasets/T1280x720/")
images_dir = main_ds+"JPEGImages/"
imagesets_dir = main_ds+"ImageSets/"
data_dir = os.path.abspath("./data/")
# data_dir = "/media/rtx3090/Disco2TB/cvazquez/nico/tfg_tomates/effdet/data/"
output_dir = os.path.abspath("./outputs/")
# output_dir = "/media/rtx3090/Disco2TB/cvazquez/nico/tfg_tomates/effdet/outputs/"
preds_dir = os.path.abspath("./preds/")
# preds_dir = "/media/rtx3090/Disco2TB/cvazquez/nico/tfg_tomates/effdet/preds/"
models_dir = os.path.abspath("./modelos/")
# models_dir = "/media/rtx3090/Disco2TB/cvazquez/nico/tfg_tomates/effdet/modelos/"
# models_dir = os.path.abspath("./modelos/backup/")
logs_dir = os.path.abspath("./lightning_logs/")
# logs_dir = "/media/rtx3090/Disco2TB/cvazquez/nico/tfg_tomates/effdet/lightning_logs/"
