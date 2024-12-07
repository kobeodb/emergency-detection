import time
from collections import defaultdict
import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

from src.models.classifiers.classifier import CNN

# Load the YOLO model
model = YOLO("../data/weights/best.pt")

# Open the video file
video_path = "../data/pipeline_eval_data/sudden cardiac arrest tatami.webm"

# Dictionary to store the state and bounding box of each person
history = defaultdict(lambda: {'state': 'MONITORING', 'bbox': None})

dtime = defaultdict(float)
mtime = defaultdict(float)
last_pos = defaultdict(lambda: None)
static_back = defaultdict(lambda: None)

with open('../../config/config.yaml') as f:
    config = yaml.safe_load(f)
    classifier = CNN(config).to('cuda')
    classifier.eval()


def preprocess_frame(frame, box):
    x1, y1, x2, y2 = map(int, box)
    person_crop = frame[y1:y2, x1:x2]

    # Resize and normalize the cropped image for CNN
    crop_tensor = cv2.resize(person_crop, (128, 128))
    crop_tensor = crop_tensor / 255.0
    crop_tensor = torch.tensor(crop_tensor, dtype=torch.float32).permute(2, 0, 1)
    crop_tensor = crop_tensor.unsqueeze(0).to('cuda')  # Add batch dimension

    return crop_tensor


def reset(track_id):
    history[track_id]['state'] = 'MONITORING'
    history[track_id]['bbox'] = None
    dtime[track_id] = 0.0
    mtime[track_id] = 0.0
    last_pos[track_id] = None


def process_video(video: str) -> None:
    cap = cv2.VideoCapture(video)
    start = time.time()

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        ctime = time.time() - start
        process_frame(frame, ctime)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def process_frame(frame, ctime):
    global history

    results = model.track(frame, persist=True)

    annotated_frame = frame.copy()
    for result in results:
        if result.boxes is None:
            continue

        for box, cls, track_id, conf in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
                result.boxes.id.cpu().numpy(),
                result.boxes.conf.cpu().numpy()
        ):
            track_id = int(track_id)

            if track_id not in history:
                reset(track_id)

            history[track_id]['bbox'] = box

            if history[track_id]['state'] == 'MONITORING' and int(cls) == 1:
                history[track_id]['state'] = 'FALL_DETECTED'
                dtime[track_id] = ctime

            elif history[track_id]['state'] == 'FALL_DETECTED':
                if ctime - dtime[track_id] >= 2.0:
                    history[track_id]['state'] = 'MOTION_TRACKING'
                    mtime[track_id] = ctime
                    last_pos[track_id] = box

            elif history[track_id]['state'] == 'MOTION_TRACKING':
                elapsed = ctime - mtime[track_id]
                motion = detect_motion(frame, track_id)

                if motion:
                    reset(track_id)
                elif elapsed >= 4.0:
                    with torch.no_grad():
                        frame_tensor = preprocess_frame(frame, box)
                        output = classifier(frame_tensor)
                        emergency_prob = output.item()
                        print(f"Emergency probability: {emergency_prob:.2f}")

                    if emergency_prob < 0.5:
                        history[track_id]['state'] = 'EMERGENCY'

            x1, y1, x2, y2 = box

            color = (0, 255, 0)
            if history[track_id]['state'] == 'FALL_DETECTED':
                color = (0, 165, 255)
            elif history[track_id]['state'] == 'MOTION_TRACKING':
                color = (0, 255, 255)
            elif history[track_id]['state'] == 'EMERGENCY':
                color = (0, 0, 255)

            cv2.rectangle(
                annotated_frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                2,
            )

            label = f"ID: {track_id}, State: {history[track_id]['state']}"
            label_position = (int(x1), int(y1) - 10)
            cv2.putText(
                annotated_frame,
                label,
                label_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                lineType=cv2.LINE_AA
            )
            print(f"Track ID {track_id}: State: {history[track_id]['state']}, BBox: {history[track_id]['bbox']}")

    cv2.imshow("Bot Brigade", annotated_frame)


def detect_motion(frame, track_id):
    global history
    global static_back

    motion = False
    x1, y1, x2, y2 = map(int, history[track_id]['bbox'])

    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if static_back[track_id] is None or static_back[track_id].shape != gray.shape:
        static_back[track_id] = gray
        return motion

    diff_frame = cv2.absdiff(static_back[track_id], gray)
    thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=1)
    cnts, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        motion = True

    return motion


if __name__ == '__main__':
    process_video(video_path)
