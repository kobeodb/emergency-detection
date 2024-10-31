import math
import os
import threading
from typing import Callable
import cv2
from ultralytics import YOLO
from path import TRAIN_PATH, VAL_PATH

# Constants
FONT_SCALE = 0.5
FONT_COLOR = (0, 0, 255)
FONT_THICKNESS = 1
CLASSES = ['standing', 'notfall', 'fall']
TRAIN_EPOCHS = 1
IMG_SIZE = 640

def initialize_model(weights: str) -> YOLO:
    """Initialize YOLO model with specified weights."""
    return YOLO(weights)

def display_detection_results(img, results):
    """Annotate image with detection results."""
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            label = CLASSES[class_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = math.ceil(box.conf[0] * 100)
            cv2.putText(img, f'{label}: {confidence}%', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
            cv2.rectangle(img, (x1, y1), (x2, y2), FONT_COLOR, 1)
    return img

def detect(video_path: str, weights: str, callback: Callable, stop_event: threading.Event):
    """Perform detection on video frames."""
    model = initialize_model(weights)
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened() and not stop_event.is_set():
        success, img = cap.read()
        if not success:
            break
        results = model(img, stream=True)
        annotated_img = display_detection_results(img, results)
        callback(annotated_img)

    cap.release()
    cv2.destroyAllWindows()

def train_model(weights: str, data_yaml: str, epochs: int = 1, img_size: int = 640):
    """Train the YOLO model on the dataset."""
    model = YOLO(weights)
    model.train(data=data_yaml, epochs=epochs, imgsz=img_size)
