import os
import shutil
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

# Load TFLite model
tflite_model_path = "vehicle_ori_classifier_128_saved_model/vehicle_ori_classifier_128_float32.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preprocess image
def preprocess_image(image_path, input_size=128):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((input_size, input_size))
    input_data = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
    return input_data

# Postprocess the results
def postprocess(output_data):
    probs = output_data[0]
    top1_idx = np.argmax(probs)
    top1_prob = probs[top1_idx]
    return top1_idx, top1_prob

# Class names
class_names = ['1_front', '2_frontleft', '3_rearleft', '4_rear', '5_rearright', '6_frontright', '7_full', '8_part']

# Initialize the results list
results = []

# Path to test folder
test_folder = "test"

# Output folder
output_folder = "output_predicted_classes"
os.makedirs(output_folder, exist_ok=True)
for class_name in class_names:
    os.makedirs(os.path.join(output_folder, class_name), exist_ok=True)

# Iterate over subfolders
for subfolder in os.listdir(test_folder):
    subfolder_path = os.path.join(test_folder, subfolder)
    if not os.path.isdir(subfolder_path):
        continue

    # Iterate over images in the subfolder
    for image_file in os.listdir(subfolder_path):
        image_path = os.path.join(subfolder_path, image_file)
        if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        # Preprocess the image
        input_data = preprocess_image(image_path)

        # Set the model input
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        interpreter.invoke()

        # Get the model output
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Postprocess the results
        top1_idx, top1_prob = postprocess(output_data)

        # Determine if prediction is correct
        actual_class = subfolder
        predicted_class = class_names[top1_idx]
        result_type = "True Positive" if actual_class == predicted_class else "False Positive"

        print({
            "Image Path": image_path,
            "Actual Class": actual_class,
            "Predicted Class": predicted_class,
            "Confidence": top1_prob,
        })

        # Append results to the list
        results.append({
            "Image Path": image_path,
            "Actual Class": actual_class,
            "Predicted Class": predicted_class,
            "Prediction Confidence": top1_prob,
            "Result Type": result_type
        })

        # Save the image in the predicted class folder
        predicted_class_folder = os.path.join(output_folder, predicted_class)
        shutil.copy(image_path, predicted_class_folder)

# Convert results to a pandas DataFrame
results_df = pd.DataFrame(results)

# Get the true and predicted classes
true_classes = results_df['Actual Class']
predicted_classes = results_df['Predicted Class']

# Generate confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes, labels=class_names)
print("Confusion Matrix:")
print(conf_matrix)

# Generate classification report (includes Precision, Recall, and F1 score)
report = classification_report(true_classes, predicted_classes, target_names=class_names)
print("Classification Report:")
print(report)

# Save the classification report to a text file
with open("classification_report.txt", "w") as f:
    f.write(report)

print("Classification report saved to 'classification_report.txt'")

# Save the DataFrame to a CSV file
output_csv_path = "predictions.csv"
results_df.to_csv(output_csv_path, index=False)

print(f"Predictions saved to {output_csv_path}")
