import os
from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')
results = model.train(data='.', epochs=10, device='mps')