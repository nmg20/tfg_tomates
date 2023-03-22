from pytorch_lightning import Trainer 

def validate_model(model,dm,num_epochs=1,logger=None):
    trainer = Trainer(gpus=1,max_epochs=num_epochs,logger=logger)
    trainer.validate(model,dm)