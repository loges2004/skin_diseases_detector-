import torch

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Perform inference on an image
results = model(r'E:\train\Acne\1.png')

# Print results
print(results)
