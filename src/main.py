#!/usr/bin/env python3
import functools
import os
import math
from typing import Callable

import dotenv
import cv2
from ultralytics import YOLO

from src.data.db.main import MinioBucketWrapper
from path import *

FONT_SCALE = 1
FONT_COLOR = (255, 255, 255)
FONT_THICKNESS = 2


def minio_temp_val(func):
    @functools.wraps(func)
    def wrapper(c: MinioBucketWrapper, filename: str, *args, **kwargs):
        f, _ = c.get_obj_file(filename, DATASET_PATH)
        nf = f.split('/')[-1]

        func(DATASET_PATH + nf, *args, **kwargs)

        if os.path.exists(DATASET_PATH + nf):
            os.remove(DATASET_PATH + nf)

    return wrapper


def minio_temp_train(func):
    @functools.wraps(func)
    def wrapper(c: MinioBucketWrapper, weights: str, amt: int = 10, *args, **kwargs):
        if not os.listdir(TRAIN_PATH):
            for obj in c.list_obj()[:amt]:
                c.get_obj_file(obj, TRAIN_PATH)

        func(weights, *args, **kwargs)

        if os.path.exists(TRAIN_PATH):
            for f in os.listdir(TRAIN_PATH):
                os.remove(f)

    return wrapper


def _use_model(path: str) -> YOLO:
    return YOLO(path)


@minio_temp_train
def _train_model(weights: str):
    return _use_model(weights).train(data=MODEL_PATH)


@minio_temp_val
def detect(video: str, weights: str, callback: Callable) -> None:
    m = _use_model(weights)
    cap = cv2.VideoCapture(video)

    prev = None

    while True:
        success, img = cap.read()

        if not success:
            break

        results = m(img, stream=True)
        for r in results:
            for box in r.boxes:
                _id = int(box.cls[0])

                if _id == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    width = x2 - x1
                    height = y2 - y1
                    center = (x1 + x2) / 2, (y1 + y2) / 2

                    if prev is not None:
                        d = ((center[0] - prev[0]) ** 2 + (center[1] - prev[1]) ** 2) ** 0.5

                        if d <= 10:
                            # stationary
                            pass

                        elif d > 10 and height / width > 1:
                            # moving
                            pass

                        elif d > 10 and height / width < 1:
                            # falling
                            pass

                    prev = center

                    out = f'id {_id}: {math.ceil((box.conf[0] * 100))}%'
                    cv2.putText(img, out, [x1, y1], cv2.FONT_HERSHEY_SIMPLEX,
                                FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
                    cv2.rectangle(img, (x1, y1), (x2, y2), FONT_COLOR, 3)

        callback(img)

    cap.release()
