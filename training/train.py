# Importing YOLO
from ultralytics import YOLO

# Load the classification model
model = YOLO('yolo11x-cls.pt')

# Train the model with all specified augmentations and parameters
results = model.train(
    data='./split_dataset',
    epochs=40, 
    imgsz=128, 
    batch=9, 
    lr0=0.0001, 
    device=0, 
    workers=8, 
    name='yolo11_classifier', 
    resume=False, 
    optimizer='Adam', 
    val=True,
    erasing=0.02,   # Random erasing augmentation
    scale=0.00,
    translate=0.0,
    flipud=0.0,    # Flip image vertically (disabled)
    fliplr=0.0,    # Flip image horizontally with probability 0.5
)


# # Train the model with all specified augmentations and parameters
# results = model.train(
#     data='./split_dataset', 
#     epochs=100, 
#     imgsz=100, 
#     batch=32, 
#     lr0=0.0001, 
#     device=0, 
#     workers=8, 
#     name='yolo11_classifier', 
#     resume=False, 
#     optimizer='Adam', 
#     val=True,
#     flipud=0.0,    # Flip image vertically (disabled)
#     fliplr=0.5,    # Flip image horizontally with probability 0.5
#     scale=0.1,     # Scale image by up to 0.5
#     hsv_h=0.015,   # Adjust hue by 0.015
#     hsv_s=0.7,     # Adjust saturation by 0.7
#     hsv_v=0.4,     # Adjust brightness by 0.4
#     mosaic=0.0,    # Enable mosaic augmentation
#     mixup=0.0,     # Disable mixup augmentation
#     degrees=2.0,   # No rotation augmentation
#     translate=0.0, # Image translation
#     shear=0.0,     # No shearing
#     perspective=0.4,  # No perspective distortion
#     erasing=0.02,   # Random erasing augmentation
#     copy_paste=0.0, # Disable copy-paste
#     dropout=0.0,   # No dropout augmentation
#     auto_augment=None  # Using random augmentations
# )
