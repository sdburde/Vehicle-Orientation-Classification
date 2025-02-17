# Importing YOLO
from ultralytics import YOLO

# Load the trained classification model
model = YOLO('runs/classify/yolo11_classifier7/weights/best.pt')

# Test the model on the test dataset
results = model.val(
    data='./split_dataset',        # Root directory of the dataset
    split='test',                  # Specify the test split
    batch=4,                       # Batch size for testing
    imgsz=64,                     # Image size for inference
    device=0,                      # Specify the GPU device
    workers=8                      # Number of workers for data loading
)


