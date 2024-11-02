import sys
import cv2
import joblib
import numpy as np
import mediapipe as mp
import pandas as pd
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, QHBoxLayout, QProgressBar, QMessageBox
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Load the trained model and label encoder
model = joblib.load('../src/models/fall_detection_model_xgb.pkl')
label_encoder = joblib.load('../src/models/label_encoder.pkl')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)


class FallDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fall Detection")
        self.setGeometry(100, 100, 1200, 800)

        # Layout and components
        self.layout = QVBoxLayout()

        # Instructions label
        self.instructions_label = QLabel("Welcome to the Fall Detection App.\n\n1. Load a video file.\n2. Click 'Start Detection' to begin.")
        self.instructions_label.setFont(QFont("Arial", 14))
        self.instructions_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.instructions_label)

        # Video display
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(1080, 720)  # Set larger size for video display
        self.layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        # Status label
        self.status_label = QLabel("Status: Waiting for video to be loaded...")
        self.status_label.setFont(QFont("Arial", 12))
        self.layout.addWidget(self.status_label)

        # Buttons
        self.button_layout = QHBoxLayout()

        self.load_video_button = QPushButton("Load Video", self)
        self.load_video_button.clicked.connect(self.load_video)
        self.button_layout.addWidget(self.load_video_button)

        self.start_detection_button = QPushButton("Start Detection", self)
        self.start_detection_button.clicked.connect(self.start_detection)
        self.start_detection_button.setEnabled(False)
        self.button_layout.addWidget(self.start_detection_button)

        # Button for using the laptop camera
        self.camera_button = QPushButton("Use Camera", self)
        self.camera_button.clicked.connect(self.use_camera)
        self.button_layout.addWidget(self.camera_button)

        self.layout.addLayout(self.button_layout)

        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.layout.addWidget(self.progress_bar)

        self.setLayout(self.layout)

        # Video capture and timer
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def use_camera(self):
        self.cap = cv2.VideoCapture(0)  # Use the laptop's camera
        if self.cap.isOpened():
            self.start_detection_button.setEnabled(True)
            self.status_label.setText("Status: Camera loaded. Ready to start detection.")
            print("Camera loaded.")
        else:
            QMessageBox.warning(self, "Warning", "Could not access the camera!")

    def load_video(self):
        video_path, _ = QFileDialog.getOpenFileName(self, "Open Video")
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            self.start_detection_button.setEnabled(True)
            self.status_label.setText("Status: Video loaded. Ready to start detection.")
            print("Video loaded.")
        else:
            QMessageBox.warning(self, "Warning", "No video selected!")

    def start_detection(self):
        if self.cap is not None:
            self.timer.start(30)  # Adjust frame rate if needed
            self.progress_bar.setValue(0)
            self.status_label.setText("Status: Detection in progress...")
            print("Detection started.")

    def update_frame(self):
        ret, frame = self.cap.read()
        label = None
        if not ret:
            self.timer.stop()
            self.status_label.setText("Status: Video ended. Detection completed.")
            QMessageBox.information(self, "Info", "Detection completed!")
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
            color = (0, 0, 255) if label.lower() == "need help" else (0, 255, 0)
            border_color = (0, 255, 0)  # Default to green
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Update progress bar
        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else 1
        progress = int((current_frame / total_frames) * 100) if total_frames > 1 else 0
        self.progress_bar.setValue(progress)

        # Display frame
        if label is not None:
            border_color = (0, 0, 255) if label.lower() == "need help" else (0, 255, 0)
        if 'border_color' in locals():
            frame = cv2.copyMakeBorder(frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=border_color)
        self.display_frame(cv2.resize(frame, (1080, 720)))

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

        # Add missing features with default values
        df['angle_hip_knee_ankle'] = 0  # Default value, adjust if possible
        df['angle_shoulder_hip'] = 0  # Default value, adjust if possible
        df['velocity'] = 0  # Default value, replace with actual calculation if available
        df['acceleration'] = 0  # Default value, replace with actual calculation if available

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
