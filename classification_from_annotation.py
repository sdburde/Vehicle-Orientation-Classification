import os
import json
import shutil

# Define paths
base_folder = "exercise_1"  # Root folder containing subdirectories
output_folders = {
    "1_frontbumper": {
        "include_classes": {"frontbumper", "frontbumpergrille", "logo", "frontws"},
        "exclude_classes": {"rightfrontdoor", "leftfrontdoor", "wheelrim", "tyre"}
    },
    "2_frontleft": {
        "include_classes": {"leftfrontdoor", "leftfrontdoorglass"},
        "exclude_classes": {"leftreardoor", "lefttaillamp", "lefttailgate", "leftreardoorglass"}
    },
    "3_rearleft": {
        "include_classes": {"leftreardoor", "leftreardoorglass"},
        "exclude_classes": {"leftfrontdoor", "leftfrontdoorglass", "leftfrontbumper", "leftfrontdoorcladding", "leftfrontventglass"}
    },
    "4_rear": {
        "include_classes": {"rearbumper", "logo", "rearws"},
        "exclude_classes": {"leftreardoor", "rightreardoor", "tyre"}
    },
    "5_rearright": {
        "include_classes": {"rightreardoor", "rightreardoorglass"},
        "exclude_classes": {"rightfrontdoor", "rightfrontdoorglass", "rightfrontbumper", "rightfrontdoorcladding", "rightfrontventglass"}
    },
    "6_frontright": {
        "include_classes": {"rightfrontdoor", "rightfrontdoorglass"},
        "exclude_classes": {"rightreardoor", "righttaillamp", "righttailgate", "rightreardoorglass", "rightreardoorcladding"}
    }
}

# Folder for unclassified images
fallback_folder = "none"
os.makedirs(fallback_folder, exist_ok=True)

# Ensure all output folders exist
for folder in output_folders:
    os.makedirs(folder, exist_ok=True)

def copy_filtered_images(base_folder, output_folders, fallback_folder):
    """
    Classify images into multiple categories based on inclusion and exclusion conditions.

    Args:
        base_folder (str): Root folder containing subdirectories with VIA files and images.
        output_folders (dict): Dictionary of output folders with include and exclude classes.
        fallback_folder (str): Folder for images not matching any category.
    """
    for subfolder in os.listdir(base_folder):
        subfolder_path = os.path.join(base_folder, subfolder)

        # Check if subfolder is a directory
        if not os.path.isdir(subfolder_path):
            continue

        # Check for the "via_region_data.json" file
        via_json_path = os.path.join(subfolder_path, "via_region_data.json")
        if not os.path.exists(via_json_path):
            print(f"No VIA JSON file in {subfolder_path}. Skipping...")
            continue

        # Load the VIA JSON annotations
        with open(via_json_path, 'r') as json_file:
            annotations = json.load(json_file)

        # Iterate through each image's annotations
        for image_filename, annotation_data in annotations.items():
            regions = annotation_data.get("regions", [])
            image_classes = set()

            # Collect all classes for the image
            for region in regions:
                region_attributes = region.get("region_attributes", {})
                class_name = region_attributes.get("identity")
                if class_name:
                    image_classes.add(class_name)

            # Check each category's inclusion and exclusion conditions
            classified = False
            for folder, conditions in output_folders.items():
                include_classes = conditions["include_classes"]
                exclude_classes = conditions["exclude_classes"]

                if include_classes.issubset(image_classes) and not exclude_classes.intersection(image_classes):
                    image_path = os.path.join(subfolder_path, image_filename)
                    if os.path.exists(image_path):
                        shutil.copy(image_path, os.path.join(folder, image_filename))
                        print(f"Copied {image_filename} to {folder}")
                        classified = True
                    else:
                        print(f"Image not found: {image_path}")
                    break  # Stop checking other categories if classified

            # If not classified, move to fallback folder
            if not classified:
                image_path = os.path.join(subfolder_path, image_filename)
                if os.path.exists(image_path):
                    shutil.copy(image_path, os.path.join(fallback_folder, image_filename))
                    print(f"Copied {image_filename} to {fallback_folder}")
                else:
                    print(f"Image not found: {image_path}")

# Run the classification
copy_filtered_images(base_folder, output_folders, fallback_folder)

print("Classification complete.")
