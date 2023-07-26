import torch
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
from ultralytics import YOLO
model = YOLO('yolov5s.pt')