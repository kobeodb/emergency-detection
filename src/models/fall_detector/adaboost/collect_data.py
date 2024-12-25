import cv2
import pandas as pd
from ultralytics import YOLO
from pathlib import Path

current_file = Path().resolve()

src_dir = current_file
while src_dir.name != "src" and src_dir != src_dir.parent:
    src_dir = src_dir.parent

yolo_pose_name = "yolo11n.pt"
yolo_pose_path = Path(src_dir / "data" / "weights" / "yolo_detection" / yolo_pose_name)

yolo_model = YOLO(yolo_pose_path)

person_class_id = 0


def extract_features(bbox, prev_bbox):
    x_min, y_min, x_max, y_max = bbox
    area = (x_max - x_min) * (y_max - y_min)
    aspect_ratio = (x_max - x_min) / (y_max - y_min)

    if prev_bbox is not None:
        x_min_prev, y_min_prev, x_max_prev, y_max_prev = prev_bbox
        centroid = ((x_min + x_max) / 2, (y_min + y_max) / 2)
        prev_centroid = ((x_min_prev + x_max_prev) / 2, (y_min_prev + y_max_prev) / 2)
        velocity = np.linalg.norm(np.array(centroid) - np.array(prev_centroid))
    else:
        velocity = 0

    return area, aspect_ratio, velocity


def process_video(video_path, video_id):
    cap = cv2.VideoCapture(video_path)
    features = []
    prev_bbox = None

    frame_count = 0


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (640, 360))

        results = yolo_model.track(resized_frame, persist=True)[0]

        for bbox in results.boxes:
            if int(bbox.cls) == person_class_id:
                area, aspect_ratio, velocity = extract_features(bbox, prev_bbox)
                features.append([frame_count, area, aspect_ratio, velocity, 0])
                prev_bbox = bbox

        frame_count += 1

    cap.release()
    return pd.DataFrame(features, columns=["Frame", "Area", "Aspect Ratio", "Velocity", "Label"])