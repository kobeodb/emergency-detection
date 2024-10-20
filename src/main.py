#!/usr/bin/env python3
import functools
import os
import math
import dotenv
import cv2
from ultralytics import YOLO

from src.data.db.main import MinioBucketWrapper

DATASET_PATH = './out/temp/'
WEIGHTS_PATH = './src/data/weights/'
MODEL_PATH = './src/data/data.yaml'
TRAIN_PATH = './src/data/dataset/train'
VAL_PATH = './src/data/dataset/val'

FONT_SCALE = 1
FONT_COLOR = (255, 255, 255)
FONT_THICKNESS = 2


def minio_init() -> MinioBucketWrapper:
    dotenv.load_dotenv()

    minio_url = os.getenv("MINIO_URL")
    minio_user = os.getenv("MINIO_USER")
    minio_password = os.getenv("MINIO_PASSWORD")
    minio_bucket_name = os.getenv("MINIO_BUCKET_NAME")

    if not all([minio_url, minio_user, minio_password, minio_bucket_name]):
        with open(".env", "w") as env_file:
            env_file.write(f"MINIO_URL=\n")
            env_file.write(f"MINIO_USER=\n")
            env_file.write(f"MINIO_PASSWORD=\n")
            env_file.write(f"MINIO_BUCKET_NAME=\n")

        print("Missing environment variables. A new .env file has been created with placeholders.")
        raise EnvironmentError("Please set the required environment variables in the .env file.")

    return MinioBucketWrapper(
        minio_url,
        minio_user,
        minio_password,
        minio_bucket_name
    )


def minio_temp_val(func):
    @functools.wraps(func)
    def wrapper(c: MinioBucketWrapper, filename: str, *args, **kwargs):
        f, _ = c.get_obj_file(filename, DATASET_PATH)

        func(DATASET_PATH + f, *args, **kwargs)

        if os.path.exists(DATASET_PATH + f):
            os.remove(DATASET_PATH + f)

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


def _use_model(weights: str) -> YOLO:
    if not weights or weights not in os.listdir(WEIGHTS_PATH):
        weights = 'yolo11n.pt'
    else:
        weights = WEIGHTS_PATH + weights

    return YOLO(weights)


@minio_temp_train
def _train_model(weights: str):
    return _use_model(weights).train(data=MODEL_PATH)


@minio_temp_val
def detect(video: str, weights: str) -> None:
    m = _use_model(weights)
    cap = cv2.VideoCapture(video)

    while True:
        success, img = cap.read()
        if not success:
            break

        prev = None
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
                            # in movement
                            pass

                        elif d > 10 and height / width < 1:
                            # falling
                            pass

                    prev = center

                    out = f'{_id}: {math.ceil((box.conf[0] * 100))}%'

                    cv2.putText(img, out, [x1, y1],
                                cv2.FONT_HERSHEY_SIMPLEX,
                                FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
                    cv2.rectangle(img, (x1, y1), (x2, y2), FONT_COLOR, 3)

            os.system('cls')

        cv2.imshow(video, img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
