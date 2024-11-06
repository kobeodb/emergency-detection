import mediapipe as mp
import cv2
import numpy as np
import os
import pandas as pd

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# Define paths (adjust these paths as needed)
TRAIN_IMAGE_PATH = '../src/data/dataset/train/images'
VAL_IMAGE_PATH = '../src/data/dataset/val/images'
TRAIN_LABEL_PATH = '../src/data/dataset/train/labels'
VAL_LABEL_PATH = '../src/data/dataset/val/labels'

# Define your classes (should match YOLO class IDs)
CLASSES = ['Need help', 'No need for help']


def extract_keypoints(image):
    """
    Extracts pose keypoints from an image using MediaPipe Pose.

    Args:
        image (numpy.ndarray): Input image in BGR format.

    Returns:
        numpy.ndarray: A flattened array of keypoints [x, y, z] if pose landmarks are detected.
                       Returns an array of zeros if no landmarks are found (length 99).
    """
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if hasattr(results, 'pose_landmarks'):
        return np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
    return np.zeros(99)


def create_dataset(image_dir, label_dir, classes):
    data = []
    labels = []

    for label_file in os.listdir(label_dir):
        # Get the corresponding image file
        image_file = label_file.replace('.txt', '.jpg')  # Assuming images are in .jpg format
        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, label_file)

        # Read and process image
        image = cv2.imread(image_path)
        if image is None:
            continue

        # Extract keypoints
        keypoints = extract_keypoints(image)

        # Read label and assign the class
        with open(label_path, 'r') as f:
            label_content = f.readline().strip()

            # Check if label content is empty
            if not label_content:
                print(f"Warning: Label file {label_file} is empty or improperly formatted.")
                continue

            # Parse class ID
            class_id = int(label_content.split()[0])  # YOLO format stores class ID as the first item
            label = classes[class_id]

        # Append to dataset
        data.append(keypoints)
        labels.append(label)

    # Create DataFrame
    df = pd.DataFrame(data)
    df['label'] = labels
    return df


# Create datasets
train_df = create_dataset(TRAIN_IMAGE_PATH, TRAIN_LABEL_PATH, CLASSES)
val_df = create_dataset(VAL_IMAGE_PATH, VAL_LABEL_PATH, CLASSES)

# Save datasets for classifier training
train_df.to_csv('train_keypoints.csv', index=False)
val_df.to_csv('val_keypoints.csv', index=False)

print("Keypoint extraction complete. Datasets saved.")
