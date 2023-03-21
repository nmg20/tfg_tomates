from Train import *

model = get_model()
load_ex_model(model, "modelos/exp602020ED_30ep.pt")
train_ds, test_ds, val_ds = load_dss("../../datasets/Tomato_1280x720","exp602020")
model.eval()
imgs, anots, preds = get_imgs_anots_preds(model,val_ds,0,2)
dm = get_dm(train_ds,val_ds,test_ds)
loader = dm.val_dataloader()
dl_iter = iter(loader)
batch = next(dl_iter)
output = model.validation_step(batch=batch,batch_idx=0)
loss = output['loss']
print(loss)


# def 

# max_bbxs = len(max(anots,key=len))
# anots_array, pred


# loader = dm.val_dataloader()
# dl_iter = iter(loader)
# batch = next(dl_iter)
# images, targets, _, _ = next(dl_iter)
# device = model.device;device
# output = model.validation_step(batch=batch,batch_idx=0)
# output

# img_grid = torchvision.utils.make_grid(images)
# writer.add_image('tomato_images', img_grid)