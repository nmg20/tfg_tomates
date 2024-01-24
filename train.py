from Model import *
from TomatoDataset import *
from config import *
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import NeptuneLogger

# from lightning.pytorch.loggers import TensorBoardLogger

# logger = TensorBoardLogger("./logs","retinanet")

# if torch.cuda.is_available():
#     trainer = Trainer(
#         accelerator="cuda", 
#         devices=1,
#         max_epochs=30, 
#         num_sanity_val_steps=1, 
#         logger=logger
#     )

# model = RetinaNetTomatoLightning(threshold=0.15)
# model = FasterRCNNTomatoLightning(threshold=0.15)
# dm = TomatoDataModule(imagesets_dir+"d801010/", images_dir, 4, 8)
# trainer.fit(model,dm)
# torch.save(model.state_dict(), "./pths/fasterrcnn.pt")

neptune_logger = NeptuneLogger(
    project="nmg20/tfg",
    api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3MGM3NzQzMC04NGQ3LTQ4M2UtYjAxMi02YWRhNWM4MTZjODcifQ==",
    log_model_checkpoints=False,
    tags = ["502030", "torchvision", "fasterrcnn"]
)

PARAMS = {
    "batch_size": config.BATCH_SIZE,
    "lr": config.LR,
    "max_epochs": config.NUM_EPOCHS,
}

dm = TomatoDataModule()
#model = SSDLightning(threshold=0.0)
#model = SSDLiteLightning(threshold=0.0)
model = FasterRCNNLightning(threshold=0.0)
#model = FCOSLightning(threshold=0.0)
# model = RetinaNetLightning(threshold=0.0)
# freeze_modules(model)

trainer = Trainer(
   accelerator="cuda",
   devices=1,
    max_epochs=config.NUM_EPOCHS,
    num_sanity_val_steps=1,
#   logger=neptune_logger,
)

trainer.fit(model, dm)

# torch.save(model.state_dict(), "./pths/neptune/head/ssdlite2.pt")
# torch.save(model.state_dict(), "./pths/neptune/701515/faster70.pt")
torch.save(model.state_dict(), "./pths/neptune/502030/faster50.pt")
