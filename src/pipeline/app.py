import os
import sys

import dotenv
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QPushButton, QWidget, QMessageBox, QComboBox
from minio import Minio, S3Error

from src.pipeline.main import Tracker

dotenv.load_dotenv()

DOWNLOAD_DIR = "./data/pipeline_eval_data/test_videos/"
BUCKET_NAME = os.environ["MINIO_BUCKET_NAME"]

minio_client = Minio(
    os.environ['MINIO_ENDPOINT'],
    os.environ['MINIO_ACCESS_KEY'],
    os.environ['MINIO_SECRET_KEY'],
    secure=False
)


class App(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Bot Brigade GUI')
        self.layout = QVBoxLayout()

        self.video_dropdown = QComboBox()
        self.video_dropdown.addItem("Select a video")
        self.layout.addWidget(self.video_dropdown)

        self.video_dropdown.currentIndexChanged.connect(self.video_selected)  # type: ignore

        self.play_button = QPushButton('Play Selected Video')
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.play_video)  # type: ignore
        self.layout.addWidget(self.play_button)

        self.fetch_videos()

        self.setLayout(self.layout)
        self.video_thread = None

    def play_video(self):
        selected_video = self.video_dropdown.currentText()

        if selected_video == "Select a video":
            QMessageBox.warning(self, "No Video Selected", "Please select a video to play.")
            return

        temp_video_path = f"../data/pipeline_eval_data/{selected_video}"
        try:
            minio_client.fget_object(BUCKET_NAME, selected_video, temp_video_path)
        except S3Error as e:
            QMessageBox.critical(self, "Error", f"Failed to download video: {e}")
            return

        Tracker(temp_video_path, '../../config.yaml')

    def fetch_videos(self):
        try:
            self.video_dropdown.clear()
            self.video_dropdown.addItem("Select a video")
            objects = minio_client.list_objects(BUCKET_NAME, recursive=True)
            video_files = [
                obj.object_name for obj in objects if obj.object_name.endswith(('.mp4', '.avi', '.mkv'))
            ]

            if not video_files:
                QMessageBox.information(self, "No Videos", "No video files found in the bucket.")
                return

            self.video_dropdown.addItems(video_files)
        except S3Error as e:
            QMessageBox.critical(self, "Error", f"Failed to fetch videos: {e}")

    def video_selected(self, index):
        """Handle video selection from the dropdown."""
        self.play_button.setEnabled(index != 0)

    def closeEvent(self, event):
        """Stop the video thread when the application is closed."""
        if self.video_thread:
            self.video_thread.stop()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = App()
    player.show()
    sys.exit(app.exec_())
