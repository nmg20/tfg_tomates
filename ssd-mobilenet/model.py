import torchvision
def get_model(device):
    # load the model 
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
    # load the model onto the computation device
    model = model.eval().to(device)
    return model