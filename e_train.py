from Model import *
from TomatoDataset import *
from config import *
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import NeptuneLogger

neptune_logger = NeptuneLogger(
    project="nmg20/tfg",
    api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3MGM3NzQzMC04NGQ3LTQ4M2UtYjAxMi02YWRhNWM4MTZjODcifQ==",
    log_model_checkpoints=False,
    tags = ["custom_loss", "effdet"]
)

PARAMS = {
    "batch_size": config.BATCH_SIZE,
    "lr": config.LR,
    "max_epochs": config.NUM_EPOCHS,
}

dm = EffDetDataModule()
model = EfficientDetLightning(threshold=0.0)

trainer = Trainer(
   accelerator="cuda",
   devices=1,
    max_epochs=config.NUM_EPOCHS,
    num_sanity_val_steps=1,
  logger=neptune_logger,
)

trainer.fit(model, dm)

torch.save(model.state_dict(), "./pths/neptune/effdet_custom_loss.pt")
