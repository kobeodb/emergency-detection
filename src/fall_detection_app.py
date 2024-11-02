import sys
import cv2
import joblib
import numpy as np
import mediapipe as mp
import pandas as pd
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Load the trained model and label encoder
model = joblib.load('fall_detection_model_xgb.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)


class FallDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fall Detection")
        self.setGeometry(100, 100, 800, 600)

        # Layout and components
        self.layout = QVBoxLayout()

        # Video display
        self.video_label = QLabel(self)
        self.layout.addWidget(self.video_label)

        # Buttons
        self.load_video_button = QPushButton("Load Video", self)
        self.load_video_button.clicked.connect(self.load_video)
        self.layout.addWidget(self.load_video_button)

        self.start_detection_button = QPushButton("Start Detection", self)
        self.start_detection_button.clicked.connect(self.start_detection)
        self.layout.addWidget(self.start_detection_button)

        self.setLayout(self.layout)

        # Video capture and timer
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def load_video(self):
        video_path, _ = QFileDialog.getOpenFileName(self, "Open Video")
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            print("Video loaded.")

    def start_detection(self):
        if self.cap is not None:
            self.timer.start(30)  # Adjust frame rate if needed
            print("Detection started.")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            return

        # Pose estimation
        keypoints = self.extract_keypoints(frame)

        # Classification
        if keypoints is not None:
            # Apply feature engineering to match training data
            keypoints_df = self.feature_engineering(pd.DataFrame([keypoints]))

            # Make prediction
            prediction_encoded = model.predict(keypoints_df)
            prediction = label_encoder.inverse_transform(prediction_encoded)
            label = prediction[0]
            color = (0, 0, 255) if label == "Need Help" else (0, 255, 0)
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Display frame
        self.display_frame(frame)

    def extract_keypoints(self, frame):
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
            return keypoints
        else:
            return None

    def feature_engineering(self, df):
        # Assuming torso keypoint is at index 11 for normalization
        torso_x = df.iloc[:, 33]  # Torso x-coordinate
        torso_y = df.iloc[:, 34]  # Torso y-coordinate

        # Normalize keypoints by torso position
        for i in range(0, len(df.columns) - 1, 3):  # Iterating over each x, y, z triplet
            df.iloc[:, i] -= torso_x
            df.iloc[:, i + 1] -= torso_y

        # Calculate angles as additional features
        df['angle_shoulder_elbow'] = np.degrees(
            np.arctan2(df.iloc[:, 1] - df.iloc[:, 4], df.iloc[:, 0] - df.iloc[:, 3]))
        df['angle_elbow_wrist'] = np.degrees(np.arctan2(df.iloc[:, 4] - df.iloc[:, 7], df.iloc[:, 3] - df.iloc[:, 6]))

        return df

    def display_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FallDetectionApp()
    window.show()
    sys.exit(app.exec_())
