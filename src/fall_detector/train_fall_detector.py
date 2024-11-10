import argparse
import math
import os
import cv2
import threading
from typing import Callable
from ultralytics import YOLO
from comet_ml import start
from comet_ml.integration.pytorch import log_model
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('COMET_ML_API_KEY')

experiment = start(
  api_key=API_KEY,
  project_name="general",
  workspace="mxttywxtty"
)

# Constants
FONT_SCALE = 0.5
FONT_COLOR = (0, 0, 255)
FONT_THICKNESS = 1
CLASSES = ['Fall Detected', 'Not Fall', 'Sitting', 'Walking']
TRAIN_EPOCHS = 10
IMG_SIZE = 640


def initialize_model(weights: str) -> YOLO:
    """Initialize YOLO model with specified weights."""
    return YOLO(weights)


def detect(video_path: str, weights: str, callback: Callable, stop_event: threading.Event, ground_truth: dict):
    """Perform detection on video frames."""
    model = initialize_model(weights)

    cap = cv2.VideoCapture(video_path)
    count = 0

    while cap.isOpened() and not stop_event.is_set():
        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame, stream=True)

        detections = []
        for detection in results[0].boxes.data:
            x1, x2, y1, y2, conf, cls = detection.cpu().numpy()

            label = CLASSES[int(cls)]

            if label == "Fall-Detected":
                detections.append((label, conf))

            cv2.putText(img, f'{label}: {conf}%', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
            cv2.rectangle(img, (x1, y1), (x2, y2), FONT_COLOR, 1)

        truth = ground_truth[count].get("Fall-Detected", 0)
        found = len(detections)
        correct = min(truth, found)
        false = max(0, found - truth)
        missed = max(0, truth - found)

        metrics["Truth"] += truth
        metrics["Found"] += found
        metrics["Correct"] += correct
        metrics["False"] += false
        metrics["Missed"] += missed

        count += 1

        callback(annotated_frame)

    cap.release()
    cv2.destroyAllWindows()

    return pd.DataFrame([{"Video": video_path.split('/')[-1], **metrics}])

def train_model(weights: str, data_yaml: str, epochs: int = TRAIN_EPOCHS, img_size: int = IMG_SIZE):
    """Train the YOLO model on the dataset."""
    model = initialize_model(weights)
    model.train(data=data_yaml, epochs=epochs, imgsz=img_size)
    log_model(experiment, model=model, model_name="yoloNewDataSet")


if __name__ == "__main__":
    data_yaml_path = os.path.abspath("../data/data.yaml")
    weights_path = os.path.abspath("../data/weights/yolo11n.pt")

    # Train the model
    train_model(weights=weights_path, data_yaml=data_yaml_path)
