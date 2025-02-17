import os
import shutil
from sklearn.model_selection import train_test_split

# Paths to the main dataset folder and where the split data will be stored
main_dataset_dir = "new_classified_images_0"
output_dir = "split_dataset"
train_dir = os.path.join(output_dir, "train")
valid_dir = os.path.join(output_dir, "val")
test_dir = os.path.join(output_dir, "test")

# Create train, validation, and test directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Get all class folders in the main dataset
class_folders = os.listdir(main_dataset_dir)

# Function to split and copy files
def split_class_data(class_name):
    class_path = os.path.join(main_dataset_dir, class_name)
    images = os.listdir(class_path)
    
    # Split into train (70%), validation (20%), and test (10%)
    train_images, temp_images = train_test_split(images, test_size=0.3, random_state=2)
    valid_images, test_images = train_test_split(temp_images, test_size=1/3, random_state=2)

    # Create class directories in train, validation, and test folders
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(valid_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    # Copy training images
    for img in train_images:
        src_path = os.path.join(class_path, img)
        dst_path = os.path.join(train_dir, class_name, img)
        shutil.copy(src_path, dst_path)

    # Copy validation images
    for img in valid_images:
        src_path = os.path.join(class_path, img)
        dst_path = os.path.join(valid_dir, class_name, img)
        shutil.copy(src_path, dst_path)

    # Copy test images
    for img in test_images:
        src_path = os.path.join(class_path, img)
        dst_path = os.path.join(test_dir, class_name, img)
        shutil.copy(src_path, dst_path)

    # Print the number of images in train, validation, and test for each class
    print(f"Class: {class_name}")
    print(f"Train: {len(train_images)} images")
    print(f"Validation: {len(valid_images)} images")
    print(f"Test: {len(test_images)} images\n")

# Iterate over each class folder and split the images
for class_folder in class_folders:
    split_class_data(class_folder)

print("Dataset split into train, validation, and test folders successfully.")
