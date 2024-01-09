from Model import *
from EffDetDataset import *
from config import *
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import NeptuneLogger

neptune_logger = NeptuneLogger(
    project="nmg20/tfg",
    api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3MGM3NzQzMC04NGQ3LTQ4M2UtYjAxMi02YWRhNWM4MTZjODcifQ==",
    log_model_checkpoints=False,
)

PARAMS = {
    "batch_size": config.BATCH_SIZE,
    "lr": config.LR,
    "max_epochs": config.NUM_EPOCHS,
}

if torch.cuda.is_available():
    trainer = Trainer(
        accelerator="cuda", 
        devices=1,
        max_epochs=config.NUM_EPOCHS, 
        num_sanity_val_steps=1, 
        logger=neptune_logger
    )

dm = EffDetDataModule()
model = EffDetModel()
freeze_layers(model)

trainer.fit(model, dm)

torch.save(model.state_dict(), "../pths/neptune/effdet.pt")
