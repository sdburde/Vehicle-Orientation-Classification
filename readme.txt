Vehicle Orientation Classification - README

Overview
This project involves training a YOLO-based classification model to classify vehicle orientations and predicting the class labels for test images. 
The following sections provide details on the training process, hyperparameters, augmentations, and prediction workflow.

---

Dataset Preparation

Script: classification_from_annotation.py
Classification Using Annotated Data
The dataset preparation script uses annotated images in VIA JSON format to classify them into six categories based on vehicle orientation. 

Input Dataset Structure
- Base Folder: `exercise_1`
  - Subfolders contain images and `via_region_data.json` files.
  - Each `via_region_data.json` file includes annotation details for the images.

Categories
1. 1_frontbumper
   - Include: `frontbumper`, `frontbumpergrille`, `logo`, `frontws`
   - Exclude: `rightfrontdoor`, `leftfrontdoor`, `wheelrim`, `tyre`
2. 2_frontleft
   - Include: `leftfrontdoor`, `leftfrontdoorglass`
   - Exclude: `leftreardoor`, `lefttaillamp`, `lefttailgate`, `leftreardoorglass`
3. 3_rearleft
   - Include: `leftreardoor`, `leftreardoorglass`
   - Exclude: `leftfrontdoor`, `leftfrontdoorglass`, `leftfrontbumper`, `leftfrontdoorcladding`, `leftfrontventglass`
4. 4_rear
   - Include: `rearbumper`, `logo`, `rearws`
   - Exclude: `leftreardoor`, `rightreardoor`, `tyre`
5. 5_rearright
   - Include: `rightreardoor`, `rightreardoorglass`
   - Exclude: `rightfrontdoor`, `rightfrontdoorglass`, `rightfrontbumper`, `rightfrontdoorcladding`, `rightfrontventglass`
6. 6_frontright
   - Include: `rightfrontdoor`, `rightfrontdoorglass`
   - Exclude: `rightreardoor`, `righttaillamp`, `righttailgate`, `rightreardoorglass`, `rightreardoorcladding`

Fallback Folder
- Images that do not match any category are moved to a `none` folder.

Classification Script
The provided Python script:
- Reads `via_region_data.json` files from subfolders.
- Extracts annotated class names for each image.
- Classifies images into categories based on inclusion and exclusion conditions.
- Copies classified images to respective output folders.

---

Training the Model

Script: train.py
The training script uses the YOLO classification framework. Below are the details:

Model Used: yolo11x-cls.pt
Dataset: Located in ./split_dataset
Hyperparameters:
- epochs: 40
- batch: 9
- imgsz: 128 (input image size)
- lr0: 0.0001 (learning rate)
- device: 0 (GPU ID)
- optimizer: Adam
- workers: 8 (number of dataloading workers)

Augmentations:
- erasing: 0.02 (random erasing augmentation)
- scale: 0.0
- translate: 0.0
- flipud: 0.0 (no vertical flipping)
- fliplr: 0.0 (no horizontal flipping, as model is trained for orientation classification)

Training Details:
- Validation enabled (val=True)
- Resume training: Disabled
- Deterministic training for reproducibility
- Auto-augment strategy: randaugment

Output: Training results are saved in runs/classify/yolo11_classifier7.

---

Prediction Workflow

Script: test_predict.py
This script uses a pre-trained TFLite model for inference.

Model: vehicle_ori_classifier_128_float32.tflite
Input Details:
- Images resized to 128x128
- Pixel values normalized to [0, 1]

Classes:
- 1_front
- 2_frontleft
- 3_rearleft
- 4_rear
- 5_rearright
- 6_frontright
- 7_full   (complete image of car from side)
- 8_part   (imgage with very close view of any part)

Process:
- Images from the test folder are preprocessed and fed into the TFLite model.
- Predictions include class label and confidence score.
- Results are evaluated for accuracy using:
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-score)

Outputs:
- Results saved as predictions.csv
- Classification report saved as classification_report.txt

---

Key Parameters and Augmentations

Training Parameters:
- epochs: Number of training iterations (40 epochs)
- batch: Batch size (9 images per batch)
- lr0: Initial learning rate (0.0001)
- optimizer: Adam optimizer for gradient descent
- device: GPU ID for training
- imgsz: Input image resolution (128x128)

Data Augmentations:
- erasing: Randomly erases parts of the image with a probability of 0.02.
- flipud: No vertical flipping applied.
- fliplr: No horizontal flipping applied, as the model is trained for orientation classification.

Inference Parameters:
- Normalization: Image pixels normalized to [0, 1].
- Input size: 128x128 for the model.

---

Outputs
Predictions:
   - CSV File: Predictions saved to predictions.csv.
   - Classification Report: Saved to classification_report.txt.

---
F1 score of each class
              precision    recall  f1-score   support

     1_front       0.81      0.91      0.86        65
 2_frontleft       0.92      0.88      0.90        68
  3_rearleft       0.93      0.89      0.91        64
      4_rear       0.83      0.95      0.88        40
 5_rearright       0.96      0.93      0.94        70
6_frontright       0.92      0.86      0.89        64
      7_full       0.88      0.84      0.86        99
      8_part       0.67      0.71      0.69        51

    accuracy                           0.87       521
   macro avg       0.86      0.87      0.87       521
weighted avg       0.87      0.87      0.87       521


---

How to Run
Prediction:
   python test_predict.py


