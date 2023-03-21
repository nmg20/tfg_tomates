from pytorch_lightning import Trainer 

def validate_model(model,dm):
    trainer = Trainer(gpus=1)
    trainer.validate(model,dm)