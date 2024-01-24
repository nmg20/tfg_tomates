from EffDetDataset import *
from Model import *
import config

model = EffDetModel()
model = load_model(model, "../pths/neptune/full/effdet.pt")
model.eval()

dm = EffDetDataModule()
batch = next(iter(dm.test_dataloader()))
images, anns, targets, ids = batch

# outputs = model(images, targets)
# inference(model, batch)
# del model