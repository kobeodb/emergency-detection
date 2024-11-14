import os
import cv2
import numpy as np
import json
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QTableWidget, QTableWidgetItem, QSlider, QMessageBox
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage, QFont
from ultralytics import YOLO
from collections import deque

class BotBrigadeApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Emergency Detector")
        self.setGeometry(100, 100, 1200, 800)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self._setup_ui()

        # Tracking state
        self.current_video_name = None
        self.ground_truth_frames = set()

    def _setup_ui(self):
        """Setup UI elements."""
        self.status_label = QLabel("Status: Select Video...")
        self.status_label.setFont(QFont("Arial", 12))
        self.layout.addWidget(self.status_label)

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(1080, 720)
        self.video_label.setScaledContents(True)
        self.layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        button_layout = QHBoxLayout()

        self.load_button = QPushButton("Load Video")
        self.load_button.clicked.connect(self.load_video)
        button_layout.addWidget(self.load_button)

        self.mark_frame_button = QPushButton("Mark Frame as Ground Truth")
        self.mark_frame_button.clicked.connect(self.mark_frame_as_ground_truth)
        self.mark_frame_button.setEnabled(False)
        button_layout.addWidget(self.mark_frame_button)

        self.layout.addLayout(button_layout)

        # Slider for video navigation
        self.video_slider = QSlider(Qt.Horizontal)
        self.video_slider.setEnabled(False)
        self.video_slider.sliderMoved.connect(self.slider_moved)
        self.layout.addWidget(self.video_slider)

    def load_video(self):
        """Allows the user to load a video file for detection."""
        video_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if video_path:
            video_name = os.path.basename(video_path)
            self.current_video_name = video_name
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Error", "Failed to load video.")
                return

            self.load_ground_truth(video_name)
            self.status_label.setText(f"Status: Loaded video {video_name}")
            self.video_slider.setEnabled(True)
            self.video_slider.setMaximum(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            self.mark_frame_button.setEnabled(True)
            self.update_frame()

    def mark_frame_as_ground_truth(self):
        """Mark the current frame as ground truth for 'Need help'."""
        if self.cap is not None:
            frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.ground_truth_frames.add(frame_number)
            self.save_ground_truth(self.current_video_name, frame_number)
            self.status_label.setText(f"Status: Marked Ground Truth Frame: {frame_number}")

    def save_ground_truth(self, video_name, frame_number):
        """Save ground truth to a file."""
        try:
            with open("ground_truth.json", "r") as f:
                ground_truth_data = json.load(f)
        except FileNotFoundError:
            ground_truth_data = {}

        ground_truth_data.setdefault(video_name, []).append(frame_number)
        with open("ground_truth.json", "w") as f:
            json.dump(ground_truth_data, f, indent=4)

    def load_ground_truth(self, video_name):
        """Load ground truth data for the given video."""
        try:
            with open("ground_truth.json", "r") as f:
                content = f.read().strip()
                if content:
                    ground_truth_data = json.loads(content)
                    self.ground_truth_frames = set(ground_truth_data.get(video_name, []))
                else:
                    self.ground_truth_frames = set()
        except FileNotFoundError:
            self.ground_truth_frames = set()

    def slider_moved(self, position):
        """Handle video slider movement."""
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            self.update_frame()

    def update_frame(self):
        """Display current frame."""
        ret, frame = self.cap.read()
        if not ret:
            return
        self.display_frame(frame)

    def display_frame(self, frame):
        """Display frame on the GUI."""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))


if __name__ == "__main__":
    app = QApplication([])
    main_window = BotBrigadeApp()
    main_window.show()
    app.exec()
