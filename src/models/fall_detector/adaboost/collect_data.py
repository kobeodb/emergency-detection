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

def process_video(video_path, video_id):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (640, 360))

        results = yolo_model.track(resized_frame, persist=True)[0]

        for box in results.boxes:
            if int(box.cls) == person_class_id:  # Only consider 'person' class
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                track_id = int(box.id) if box.id is not None else -1
                width = xmax - xmin
                height = ymax - ymin

                detections.append({
                    "video_id": video_id,
                    "frame": frame_count,
                    "track_id": track_id,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "width": width,
                    "height": height
                })

        frame_count += 1

    cap.release()
    return pd.DataFrame(detections)