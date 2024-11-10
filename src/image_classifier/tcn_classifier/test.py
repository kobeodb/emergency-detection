# from collections import Counter
#
# import torch
# from torch.utils.data import DataLoader
#
# from src.image_classifier.tcn_classifier.dataset import PoseSequenceDataset
#
# # Load dataset and check label distribution
# train_data = PoseSequenceDataset('../../data/dataset/train/images', '../../data/dataset/train/labels')
#
# # Extract all labels from the dataset
# train_labels = [label for _, label in DataLoader(train_data, batch_size=len(train_data))]
# train_labels = torch.cat(train_labels).tolist()
#
# # Print class distribution
# print("Training Labels Distribution:", Counter(train_labels))


import os

image_files = os.listdir('../../data/dataset/train/images')
label_files = os.listdir('../../data/dataset/train/labels')

print(f"Number of images: {len(image_files)}")
print(f"Number of labels: {len(label_files)}")

missing_labels = [img for img in image_files if f"{os.path.splitext(img)[0]}.txt" not in label_files]

if missing_labels:
    print("Images with missing labels:", missing_labels)
else:
    print("All images have corresponding labels.")

