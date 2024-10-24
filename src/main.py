import math
import os
import threading
from typing import Callable
import cv2
from sentry_sdk.utils import epoch
from ultralytics import YOLO
from src.data.db.main import MinioBucketWrapper
from path import *

FONT_SCALE = 0.5
FONT_COLOR = (0, 0, 255)
FONT_THICKNESS = 1
CLASSES = ['fall', 'notfall', 'standing']

def _val_model(weights: str) -> YOLO:
    return YOLO(weights)

def download_data_from_minio(minio_client: MinioBucketWrapper, folder: str, local_dir: str):
    os.makedirs(local_dir, exist_ok=True)
    for obj in minio_client.list_obj(folder):
        minio_client.get_obj_file(obj, local_dir)

def detect(video: str, weights: str, callback: Callable, stop_event: threading.Event):
    model = _val_model(weights)
    cap = cv2.VideoCapture(video)

    while cap.isOpened():
        if stop_event.is_set():
            break

        success, img = cap.read()
        if not success:
            break

        results = model(img, stream=True)
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                label = CLASSES[class_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = math.ceil(box.conf[0] * 100)
                out = f'{label}: {confidence}%'
                cv2.putText(img, out, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
                cv2.rectangle(img, (x1, y1), (x2, y2), FONT_COLOR, 1)

        callback(img)

    cap.release()
    cv2.destroyAllWindows()

def train_model(minio_client: MinioBucketWrapper, weights: str, data_yaml: str):
    local_train_dir = TRAIN_PATH
    download_data_from_minio(minio_client, 'labeled_frames_roboflow/train', local_train_dir)
    download_data_from_minio(minio_client, 'labeled_frames_roboflow/valid', VAL_PATH)

    model = YOLO(weights)
    model.train(data=data_yaml, epochs=50, imgsz=640)