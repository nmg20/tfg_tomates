from Inference import *

name="d701515"
# output = Path(uniquify_dir(output_dir+f"/test/run"))
output = output_dir+f"/test/test/"
# os.mkdir(output)
s = [0.15, 0.20, 0.25, 0.30, 0.35]
dm = get_dm_standalone(name=name)
for x in s:
    model = load_model(name,conf=0.0,skip=x)
    model.eval()
    ds = dm.pred_dataset().ds
    for image, ann in zip([i for i,_,_,_ in ds.get_imgs_and_anots()], [i for _,i,_,_ in ds.get_imgs_and_anots()]):
        bboxes, confs, loss = get_pred(model, image)
        draw_images_stacked(image, bboxes, confs[0], (loss,np.sum(confs[0])/len(confs[0])), f"{output}/{name}_{x}_{image.filename.split('/')[-1]}",ann)
        # save_hist(confs, f"{output}/confs_{name}_{x}.png")