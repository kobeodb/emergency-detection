import time
from collections import defaultdict

import cv2
import joblib
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

CONFIG_PATH = '../../config.yaml'
RF_WEIGHT_PATH = '../models/weights/best_rf_model_M.pkl'
YOLO_WEIGHT_PATH = '../models/weights/best_4.pt'

MOTION_THRESHOLD = 2.0
EMERGENCY_THRESHOLD = 3.0
MOVEMENT_THRESHOLD = 10000

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

yolo_model = YOLO(YOLO_WEIGHT_PATH)
rf_model = joblib.load(RF_WEIGHT_PATH)

detection_time = defaultdict(float)
motion_time = defaultdict(float)

tracker = defaultdict(lambda: {'state': 'MONITORING', 'bbox': None, 'static_back': None})


def _process_frame(frame, current_time):
    """
    Process a single frame of the video, detecting and tracking falls.

    :param frame: The frame of the video.
    :param current_time: The current time in seconds.
    :return: None
    """
    results = yolo_model.track(frame, persist=True)

    annotated_frame = frame.copy()
    for result in results:
        if not result.boxes:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(),
                                   score_threshold=0.5, nms_threshold=0.5)

        if len(indices) > 0:
            for i in indices.flatten():  # type: ignore
                track_id = int(result.boxes.id[i])

                if track_id not in tracker:
                    _reset(track_id)

                process_box(frame, track_id, boxes[i], classes[i], current_time)
                annotated_frame = _draw(annotated_frame, track_id, color_map(track_id))

    cv2.imshow("Bot Brigade", annotated_frame)


def process_box(frame, track_id, box, cls, current_time):
    """
    Process a detected bounding box for a tracked object.

    :param frame: The frame of the video.
    :param track_id: The current id of the tracked object.
    :param box: The detected bounding box.
    :param cls: The class of the detected predicted object.
    :param current_time: The current time in seconds.
    :return: None
    """
    tracker[track_id]['bbox'] = box
    state = tracker[track_id]['state']

    if state == 'MONITORING' and int(cls) == 1:
        tracker[track_id]['state'] = 'FALL_DETECTED'
        detection_time[track_id] = current_time

    elif state == 'FALL_DETECTED':
        if current_time - detection_time[track_id] >= MOTION_THRESHOLD:
            tracker[track_id]['state'] = 'MOTION_TRACKING'
            motion_time[track_id] = current_time

    elif state == 'MOTION_TRACKING':
        handle_motion(frame, track_id, current_time)


def handle_motion(frame, track_id, current_time):
    """
    Handle the motion state of a tracked object, determining whether it is an emergency.

    :param frame: The frame of the video.
    :param track_id: The current id of the tracked object.
    :param current_time: The current time in seconds.
    :return: None
    """
    elapsed = current_time - motion_time[track_id]
    motion = detect_motion(frame, track_id)

    if motion:
        _reset(track_id)
    elif elapsed >= EMERGENCY_THRESHOLD:
        frame = _preprocess_frame(frame, track_id)
        prob = predict(frame)

        if prob <= 0.5:
            tracker[track_id]['state'] = 'EMERGENCY'


def detect_motion(frame, track_id):
    """
    Detect significant motion for a tracked object.

    :param frame: The frame of the video.
    :param track_id: The current id of the tracked object.
    :return: True if the object is in motion, False otherwise.
    """
    motion = False
    x1, y1, x2, y2 = map(int, tracker[track_id]['bbox'])

    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if tracker[track_id]['static_back'] is None or tracker[track_id]['static_back'].shape != gray.shape:  # type: ignore
        tracker[track_id]['static_back'] = gray  # type: ignore
        return motion

    diff_frame = cv2.absdiff(tracker[track_id]['static_back'], gray)  # type: ignore
    thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=1)  # type: ignore
    cnts, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < MOVEMENT_THRESHOLD:
            continue
        motion = True

    return motion


def extract_kps(image):
    """"
    Extract keypoints from an image using MediaPipe Pose.

    :param image: The image to be processed.
    :return: The keypoints extracted from the image.
    """
    result = pose.process(image)
    kps = np.zeros(33 * 3)

    if result.pose_landmarks:
        for i, landmark in enumerate(result.pose_landmarks.landmark):  # ignore
            kps[i * 3:i * 3 + 3] = [landmark.x, landmark.y, landmark.z]

    return kps


def predict(frame):
    """
    Makes a prediction based on the extracted keypoints.

    :param frame: The frame of the video.
    :return: The predicted probability.
    """
    pose_keypoints = extract_kps(frame).reshape(1, -1)
    return rf_model.predict_proba(pose_keypoints)[0, 1]


def _preprocess_frame(frame, track_id):
    """
    Makes a prediction based on the extracted keypoints.

    :param frame: The frame of the video.
    :param track_id: The current id of the tracked object.
    :return: The predicted probability.
    """
    x1, y1, x2, y2 = map(int, tracker[track_id]['bbox'])

    frame = frame[y1:y2, x1:x2]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (256, 256))

    return frame


def _draw(frame, track_id, color):
    """
    Draw a bounding box and state label on the frame for a tracked object.

    :param frame: The frame of the video.
    :param track_id: The current id of the tracked object.
    :param color: The border color of the drawn bounding box.
    :return: The frame with drawn bounding box.
    """
    x1, y1, x2, y2 = map(int, tracker[track_id]['bbox'])

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"ID: {track_id}, State: {tracker[track_id]['state']}"
    label_position = (x1, y1 - 10)

    cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, lineType=cv2.LINE_AA)

    return frame


def color_map(track_id):
    """
    Determine color of bounding box by state of detected object.

    :param track_id: The current id of the tracked object.
    :return: The color of the bounding box.
    """
    cmap = {
        'FALL_DETECTED': (0, 165, 255),
        'MOTION_TRACKING': (0, 255, 255),
        'EMERGENCY': (0, 0, 255),
    }
    return cmap.get(tracker[track_id]['state'], (0, 255, 0))


def _reset(track_id):
    """
    Reset the tracker to its default position

    :param track_id: The current id of the tracked object.
    :return: None
    """
    detection_time[track_id] = 0.0
    motion_time[track_id] = 0.0

    tracker[track_id] = {
        'state': 'MONITORING',
        'bbox': None,
        'static_back': None
    }


def process_video(video) -> None:
    """
    Entry point for processing video.

    :param video: The video to be processed.
    :return: None
    """
    cap = cv2.VideoCapture(video)
    start = time.time()

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        current_time = time.time() - start
        _process_frame(frame, current_time)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    pass


