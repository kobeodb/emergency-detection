import sys
import cv2
import joblib
import numpy as np
import mediapipe as mp
import pandas as pd
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QHBoxLayout, \
    QProgressBar, QMessageBox, QComboBox
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt
import warnings

from src.data.db.main import MinioBucketWrapper

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Load the trained model and label encoder
model = joblib.load('data/models/fall_detection_model_xgb.pkl')
label_encoder = joblib.load('data/models/label_encoder.pkl')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

minio_client = MinioBucketWrapper()


def extract_keypoints(frame):
    """
       Extracts pose keypoints from an image using MediaPipe Pose.

       Args:
           frame (numpy.ndarray): Input image in BGR format.

       Returns:
           numpy.ndarray: A flattened array of keypoints [x, y, z] if pose landmarks are detected.
                          Returns an array of zeros if no landmarks are found (length 99).
    """
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if hasattr(results, 'pose_landmarks'):
        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
        return keypoints
    return None


def feature_engineering(df):
    """
    Performs feature engineering on a DataFrame containing keypoints, normalizing by torso position
    and adding angle, velocity, and acceleration features.

    Args:
        df (pandas.DataFrame): DataFrame with keypoints [x, y, z] in columns, assumed to be ordered in triplets.

    Returns:
        pandas.DataFrame: DataFrame with normalized keypoints and additional engineered features.
    """
    # Torso keypoint coordinates (index 11 for x and y in flattened keypoints)
    torso_x, torso_y = df.iloc[:, 33], df.iloc[:, 34]

    # Normalize all keypoints by subtracting torso position
    df.iloc[:, ::3] -= torso_x.values[:, np.newaxis]  # Normalize x-coordinates
    df.iloc[:, 1::3] -= torso_y.values[:, np.newaxis]  # Normalize y-coordinates

    # Calculate joint angles as additional features
    df['angle_shoulder_elbow'] = np.degrees(
        np.arctan2(df.iloc[:, 1] - df.iloc[:, 4],
                   df.iloc[:, 0] - df.iloc[:, 3]))

    df['angle_elbow_wrist'] = np.degrees(
        np.arctan2(df.iloc[:, 4] - df.iloc[:, 7],
                   df.iloc[:, 3] - df.iloc[:, 6]))

    # Add placeholders for missing features with default values
    df = df.assign(
        angle_hip_knee_ankle=0,
        angle_shoulder_hip=0,
        velocity=0,
        acceleration=0
    )

    return df


def _annotate_frame(frame, label):
    """Annotates the frame with the detection label and adds a border based on detection status."""
    color = (0, 0, 255) if label.lower() == "need help" else (0, 255, 0)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    border_color = color
    cv2.copyMakeBorder(frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=border_color)


class FallDetectionApp(QWidget):
    """
    A PyQt5-based GUI application for real-time fall detection using video input.

    This application allows users to load a video file or use a camera to detect falls based on pose estimation
    and classification of movement patterns.
    """

    def __init__(self):
        """Initializes the Fall Detection App with UI components, layout, and event handling."""
        super().__init__()
        self.setWindowTitle("Fall Detection")
        self.setGeometry(100, 100, 1200, 800)

        # Layout setup
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Initialize UI components
        self._setup_labels()
        self._setup_video_display()
        self._setup_buttons()
        self._setup_progress_bar()

        self.videos_dropdown = None

        # Video capture and timer
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)  # type: ignore

    def _setup_labels(self):
        """Sets up instructional and status labels."""
        self.instructions_label = QLabel(
            "Welcome to the Fall Detection App.\n\n1. Load a video file.\n2. Click 'Start Detection' to begin.")
        self.instructions_label.setFont(QFont("Arial", 14))
        self.instructions_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.instructions_label)

        self.status_label = QLabel("Status: Waiting for video to be loaded...")
        self.status_label.setFont(QFont("Arial", 12))
        self.layout.addWidget(self.status_label)

    def _setup_video_display(self):
        """Sets up the video display label."""
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(1080, 720)
        self.layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

    def _setup_buttons(self):
        """Sets up buttons for loading video, starting detection, and using the camera."""
        self.button_layout = QHBoxLayout()

        # Dropdown for MinIO video selection
        self.videos_dropdown = QComboBox(self)
        self.videos_dropdown.addItem("Choose video media from MinIO")
        self.videos_dropdown.currentIndexChanged.connect(self.load_video_from_dropdown)  # type: ignore
        self.button_layout.addWidget(self.videos_dropdown)

        self.start_detection_button = self._create_button("Start Detection", self.start_detection, enabled=False)
        self.button_layout.addWidget(self.start_detection_button)

        self.camera_button = self._create_button("Use Camera", self.use_camera)
        self.button_layout.addWidget(self.camera_button)

        self.layout.addLayout(self.button_layout)
        self._populate_minio_videos()  # Populate dropdown with videos from MinIO

    def _populate_minio_videos(self):
        """Fetches video file names from MinIO and populates the dropdown."""
        try:
            objects = minio_client.list_obj()
            for obj in objects:
                if obj.endswith((".mp4", ".avi", ".mov")):
                    self.videos_dropdown.addItem(obj)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not fetch videos from MinIO: {e}")

    def _setup_progress_bar(self):
        """Sets up a progress bar for visualizing detection progress."""
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.layout.addWidget(self.progress_bar)

    def _create_button(self, text, callback, enabled=True):
        """Helper function to create and configure a button."""
        button = QPushButton(text, self)
        button.clicked.connect(callback)  # type: ignore
        button.setEnabled(enabled)
        return button

    def use_camera(self):
        """Activates the laptop camera for video capture."""
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.start_detection_button.setEnabled(True)
            self.status_label.setText("Status: Camera loaded. Ready to start detection.")
        else:
            QMessageBox.warning(self, "Warning", "Could not access the camera!")

    def load_video_from_dropdown(self):
        """Loads a selected video file from MinIO and prepares it for detection."""
        selected_video = self.videos_dropdown.currentText()
        if selected_video and selected_video != "Choose video media from MinIO":
            try:
                # Download video to local temporary storage
                minio_client.get_obj_file(selected_video, f"../out/temp/{selected_video}")
                self.cap = cv2.VideoCapture(f"../out/temp/{selected_video}")
                self.start_detection_button.setEnabled(True)
                self.status_label.setText(f"Status: Video '{selected_video}' loaded. Ready to start detection.")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not load video from MinIO: {e}")
        else:
            self.start_detection_button.setEnabled(False)
            self.status_label.setText("Status: Waiting for video to be loaded.")

    def start_detection(self):
        """Begins the fall detection process by starting the frame update timer."""
        if self.cap is not None:
            self.timer.start(30)  # Adjust frame rate as needed
            self.progress_bar.setValue(0)
            self.status_label.setText("Status: Detection in progress...")

    def update_frame(self):
        """Processes each frame from the video/camera, performs pose estimation, and updates the UI."""
        ret, frame = self.cap.read()
        if not ret:
            self._end_detection()
            return

        # Pose estimation and classification
        keypoints = extract_keypoints(frame)
        if keypoints is not None:
            keypoints_df = feature_engineering(pd.DataFrame([keypoints]))
            prediction = label_encoder.inverse_transform(model.predict(keypoints_df))[0]
            _annotate_frame(frame, prediction)

        # Update progress and display frame
        self._update_progress()
        self.display_frame(cv2.resize(frame, (1080, 720)))

    def _end_detection(self):
        """Ends the detection process and updates the status."""
        self.timer.stop()
        self.status_label.setText("Status: Video ended. Detection completed.")
        QMessageBox.information(self, "Info", "Detection completed!")

    def _update_progress(self):
        """Updates the progress bar based on the current frame count."""
        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        progress = int((current_frame / total_frames) * 100)
        self.progress_bar.setValue(progress)

    def display_frame(self, frame):
        """Displays the given frame in the video label."""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        q_image = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0],
                         rgb_image.strides[0], QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

    def closeEvent(self, event):
        """Handles cleanup on application close."""
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FallDetectionApp()
    window.show()
    sys.exit(app.exec_())
