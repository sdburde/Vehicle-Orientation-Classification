import tensorflow as tf
import numpy as np
from PIL import Image

# Load TFLite model
tflite_model_path = "vehicle_ori_classifier_128_saved_model/vehicle_ori_classifier_128_float32.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess image
def preprocess_image(image_path, input_size=128):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((input_size, input_size))
    input_data = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
    return input_data

input_data = preprocess_image("test/1_frontbumper/15qnSjz1NQ_1648015365264.jpg")

# Set the model input
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get the model output
output_data = interpreter.get_tensor(output_details[0]['index'])

# Process the results
def postprocess(output_data):
    # Assuming the output is probabilities and class labels
    probs = output_data[0]
    top1_idx = np.argmax(probs)
    top1_prob = probs[top1_idx]
    return top1_idx, top1_prob

# Replace with actual class names
# class_names = {0: '1_frontbumper', 1: '2_frontleft', 2: '3_rearleft', 3: '4_rear', 4: '5_rearright', 5: '6_frontright', 6: '7_full', 7: '8_part'}  # Update this list as needed
class_names = ['1_front', '2_frontleft', '3_rearleft', '4_rear', '5_rearright', '6_frontright', '7_full', '8_part']  # Update this list as needed

# Postprocess the results
top1_idx, top1_prob = postprocess(output_data)

# Print results
# print("Top-1 Class Name:", top1_idx)
print("Top-1 Class Name:", class_names[top1_idx])
print("Top-1 Probability:", top1_prob)
