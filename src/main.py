#!/usr/bin/env python3
import argparse
import functools
import os
import math
import dotenv
import cv2
from ultralytics import YOLO

from data.db.main import MinioBucketWrapper

DATASET_PATH = '../out/dataset/'
WEIGHTS_PATH = './data/weights/yolov8n.pt'
MODEL_PATH = './data/data.yaml'

WEIGHTS_DEFAULT = 'yolov8n.pt'

FONT_SCALE = 1
FONT_COLOR = (255, 255, 255)
FONT_THICKNESS = 2


def minio_init() -> MinioBucketWrapper:
    dotenv.load_dotenv()

    minio_url = os.getenv("MINIO_URL")
    minio_user = os.getenv("MINIO_USER")
    minio_password = os.getenv("MINIO_PASSWORD")
    minio_bucket_name = os.getenv("MINIO_BUCKET_NAME")

    return MinioBucketWrapper(
        minio_url,
        minio_user,
        minio_password,
        minio_bucket_name
    )


def minio_temp_file(func):
    @functools.wraps(func)
    def wrapper(c: MinioBucketWrapper, filename: str, *args, **kwargs):
        f = c.get_obj_file(filename, DATASET_PATH)

        func(DATASET_PATH + f, *args, **kwargs)

        if os.path.exists(DATASET_PATH + f):
            os.remove(DATASET_PATH + f)

    return wrapper


@minio_temp_file
def detect(video: str, weights: str = WEIGHTS_PATH) -> None:
    model = YOLO(weights or WEIGHTS_PATH)
    cap = cv2.VideoCapture(video)

    while True:
        success, img = cap.read()
        if not success:
            break

        results = model(img, stream=True)
        for r in results:
            for box in r.boxes:
                _id = int(box.cls[0])

                if _id == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    out = f'{_id}: {math.ceil((box.conf[0] * 100))}%'

                    cv2.putText(img, out, [x1, y1],
                                cv2.FONT_HERSHEY_SIMPLEX,
                                FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
                    cv2.rectangle(img, (x1, y1), (x2, y2), FONT_COLOR, 3)

        cv2.imshow(video, img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    client = minio_init()

    parser = argparse.ArgumentParser()

    parser.add_argument('filename')
    parser.add_argument('-w', '--weights')

    args = parser.parse_args()

    detect(client, args.filename, args.weights)
