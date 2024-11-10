from torch.utils.data import Dataset
import os
import torch
import cv2
import numpy as np
import mediapipe as mp

class PoseSequenceDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.images = sorted(os.listdir(image_dir))
        self.labels = sorted(os.listdir(label_dir))
        self.sequences = []
        self.targets = []

        # Initialize MediaPipe Pose once (with GPU disabled)
        self.mp_pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False)

        for img, lbl in zip(self.images, self.labels):
            label_path = os.path.join(self.label_dir, lbl)
            img_path = os.path.join(self.image_dir, img)

            label = self._read_yolo_label(label_path)
            self.sequences.append(img_path)
            self.targets.append(label)

        print(f"Total Samples Loaded: {len(self.sequences)}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        img_path = self.sequences[idx]
        label = self.targets[idx]

        image = cv2.imread(img_path)
        keypoints = self._extract_keypoints(image)

        return (
            torch.tensor(keypoints, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )

    def _read_yolo_label(self, label_path):
        with open(label_path, 'r') as f:
            for line in f:
                class_id = int(line.split()[0])
                if class_id == 1:  # Fall Detected
                    return 1
        return 0  # Not Fall

    def _extract_keypoints(self, image):
        """Extract pose keypoints using MediaPipe (CPU-only)."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.mp_pose.process(image_rgb)
        if results.pose_landmarks:
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
            return keypoints
        else:
            return np.zeros(33 * 3)  # 33 keypoints with 3D coordinates each
