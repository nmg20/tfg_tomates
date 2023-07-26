from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import os
data_dir = "./data/"
output_dir = "./outputs/"

image_names = os.listdir(data_dir)
# img = read_image(data_dir+image_name)
images = [read_image(data_dir+x) for x in image_names]
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
model.eval()

preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = [preprocess(x) for x in images]

# Step 4: Use the model and visualize the prediction
predictions = model(batch)
for prediction, image_name, image in zip(predictions, image_names, images):
	labels = [weights.meta["categories"][i] for i in prediction["labels"]]
	box = draw_bounding_boxes(image, boxes=prediction["boxes"],
	                          labels=labels,
	                          colors="orange",
	                          width=4, font_size=40)
	im = to_pil_image(box.detach())
	# im.show()
	im.save(f"{output_dir}retinanet/retinanet_{image_name}")