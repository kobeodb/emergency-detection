import os
import cv2
import numpy as np
import joblib
import mediapipe as mp
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QProgressBar, QMessageBox, QFileDialog
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage, QFont
from ultralytics import YOLO
from sklearn.preprocessing import StandardScaler
from collections import deque
import time

# Load models
yolo_model = YOLO('../data/weights/best.pt')
classifier = joblib.load('../data/models/improved_fall_detection_model_xgb.pkl')
label_encoder = joblib.load('../data/models/improved_label_encoder.pkl')

# Initialize pose estimation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

# Constants
MOTION_THRESHOLD = 5  # Low movement threshold
TIME_THRESHOLD = 4  # Seconds of low movement to trigger classifier
MAX_HISTORY = 30  # Bounding box history size


class FallDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Emergency Detector")
        self.setGeometry(100, 100, 1200, 800)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self._setup_ui()

        # Video capture and timer
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Timer for motion tracking
        self.motion_timer = QTimer()
        self.motion_timer.timeout.connect(self.update_motion_time)
        self.motion_elapsed_time = 0  # Elapsed time for motion tracking

        # Tracking state
        self.motion_history = deque(maxlen=MAX_HISTORY)
        self.fall_start_time = None
        self.current_bbox = None

        self.ground_truth_labels = []
        self.metrics = {
            "Truth": 0,
            "Found": 0,
            "Correct": 0,
            "False": 0,
            "Missed": 0
        }

    def generate_labels(self):
        """Generate labels for each frame based on defined ranges."""
        return ["Not Fall" if 0 <= i <= 189 else
                "Fall Detected" if 190 <= i <= 275 else
                "Need help" if 276 <= i <= 585 else
                "Not Fall" if 586 <= i <= 646 else
                "unlabeled" for i in range(self.frame_count)]

    def _setup_ui(self):
        """Setup UI elements."""
        self.timer_label = QLabel("Motion Timer: 0 s")
        self.timer_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.timer_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.timer_label)

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(1080, 720)
        self.layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        self.status_label = QLabel("Status: Waiting for video input...")
        self.status_label.setFont(QFont("Arial", 12))
        self.layout.addWidget(self.status_label)

        self.start_button = QPushButton("Start Detection")
        self.start_button.clicked.connect(self.start_detection)
        self.layout.addWidget(self.start_button)

        self.load_button = QPushButton("Load Video")
        self.load_button.clicked.connect(self.load_video)
        self.layout.addWidget(self.load_button)

    def start_detection(self):
        """Starts video capture and detection from camera."""
        self.reset_timers()
        self.timer.start(24)  # Set frame update interval
        self.status_label.setText("Status: Detection in progress...")

    def load_video(self):
        """Allows the user to load a video file for detection."""
        video_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Error", "Failed to load video.")
                return

            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.ground_truth_labels = self.generate_labels()

            self.reset_timers()
            self.timer.start(30)
            self.status_label.setText(f"Status: Loaded video {os.path.basename(video_path)}")

    def update_frame(self):
        """Process each frame for YOLO detection, motion tracking, and fall classification."""
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.status_label.setText("Status: Detection complete.")
            self.reset_timers()

            print("\n\nClassification Frame Report:")
            print(pd.DataFrame([self.metrics]).to_string())

            return

        results = yolo_model(frame)
        detections = results[0].boxes
        count = 0

        self.metrics["Truth"] = self.ground_truth_labels.count("Need help")

        fall_detected = False
        for det in detections:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            conf = float(det.conf[0])
            cls = int(det.cls[0])
            cls_name = yolo_model.names[cls]

            color = (0, 255, 0)
            if cls_name == "Fall Detected":
                fall_detected = True
                self.current_bbox = (x1, y1, x2, y2)
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{cls_name}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if fall_detected:
            self.track_motion()
            if self.low_movement_detected():
                self.check_with_classifier(frame)
        else:
            self.reset_fall_state()

        self.display_frame(frame)

    def save_metrics(self):
        """Save metrics and labels to a CSV file."""
        # Convert metrics to DataFrame
        metrics_df = pd.DataFrame([self.metrics])

        # Save metrics to CSV
        metrics_df.to_csv("./data/metrics.csv", index=False)

    def track_motion(self):
        """Track bounding box position and check for movement."""
        if self.current_bbox:
            self.motion_history.append(self.current_bbox)
            if not self.motion_timer.isActive():
                self.motion_timer.start(1000)  # Update motion time every second
            if not self.fall_start_time:
                self.fall_start_time = time.time()

    def update_motion_time(self):
        """Update motion timer label."""
        self.motion_elapsed_time += 1
        self.timer_label.setText(f"Motion Timer: {self.motion_elapsed_time} s")

    def reset_fall_state(self):
        """Reset the fall detection state."""
        self.motion_history.clear()
        self.fall_start_time = None
        self.current_bbox = None
        self.motion_timer.stop()

    def reset_timers(self):
        """Reset timers and UI elements to their initial state."""
        self.motion_timer.stop()
        self.motion_elapsed_time = 0
        self.timer_label.setText("Motion Timer: 0 s")

    def low_movement_detected(self):
        """Check if movement has been low for the specified time."""
        if len(self.motion_history) > 1:
            movement = np.linalg.norm(
                np.array(self.motion_history[-1][:2]) - np.array(self.motion_history[-2][:2])
            )
            return movement < MOTION_THRESHOLD and time.time() - self.fall_start_time >= TIME_THRESHOLD
        return False

    def check_with_classifier(self, frame):
        """Use classifier to confirm if a fall has occurred and update metrics accurately."""
        if self.current_bbox:
            x1, y1, x2, y2 = self.current_bbox
            cropped_frame = frame[y1:y2, x1:x2]
            keypoints = self.extract_keypoints(cropped_frame)

            if keypoints is not None:
                prediction = classifier.predict([keypoints])[0]
                label = label_encoder.inverse_transform([prediction])[0]

                frame_index = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                truth_label = self.ground_truth_labels[frame_index]

                # Suppose 'Need help' means a fall is detected
                if label == "Need help":
                    self.metrics["Found"] += 1
                    if label == truth_label:
                        self.metrics["Correct"] += 1
                    else:
                        self.metrics["False"] += 1
                    color = (0, 0, 255)
                else:
                    if label == truth_label:
                        self.metrics["Missed"] += 1
                    color = (0, 255, 0)

                # Display result
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    def extract_keypoints(self, frame):
        """Extract pose keypoints using MediaPipe Pose."""
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
            if len(keypoints) < 104:
                # Pad missing values with NaN to match expected feature length
                keypoints = np.pad(keypoints, (0, 104 - len(keypoints)), constant_values=np.nan)
            return keypoints
        # Return NaNs if no keypoints are detected
        return np.full(104, np.nan)

    def display_frame(self, frame):
        """Display frame on the GUI."""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

if __name__ == "__main__":
    app = QApplication([])
    main_window = FallDetectionApp()
    main_window.show()
    app.exec()


