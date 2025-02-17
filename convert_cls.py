from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("vehicle_ori_classifier_128.pt")

# Export the model to TFLite format
model.export(format="tflite" , imgsz=128)


# # Load the exported TFLite model
# model = YOLO("yolov11x_model/vehicle_ori_classifier_best.pt")

# # Run inference
# results = model("test/1_frontbumper/F7nZtsHkr5_1646890799747.jpg")

# print(results[0].names)
