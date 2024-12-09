import time
from collections import defaultdict
import cv2
import numpy as np
import torch
import yaml
from torchvision.transforms import transforms
from ultralytics import YOLO

from src.models.classifiers.classifier import CNN


class Tracker:
    def __init__(self, video_path: str, config: str = '../../config.yaml'):
        with open(config) as f:
            _config = yaml.safe_load(f)

        self.video = video_path
        self.model = YOLO(_config['model']['detector']['finetuned_weights_path'])

        self.device = _config['system']['device']

        self.classifier = CNN(_config).to(self.device)
        self.classifier.eval()

        self.dtime = defaultdict(float)
        self.mtime = defaultdict(float)

        self.history = defaultdict(lambda: {'state': 'MONITORING', 'bbox': None})
        self.last_pos = defaultdict(lambda: None)
        self.static_back = defaultdict(lambda: None)

        self.process_video()

    def _process_frame(self, frame, ctime):
        results = self.model.track(frame, persist=True)

        annotated_frame = frame.copy()
        for result in results:
            if not result.boxes or result.boxes.id is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(),
                                       score_threshold=0.5, nms_threshold=0.5)

            if len(indices) > 0:
                for i in indices.flatten():  # type: ignore
                    box = boxes[i]
                    cls = classes[i]
                    conf = confidences[i]
                    track_id = int(result.boxes.id[i])

                    if track_id not in self.history:
                        self._reset(track_id)

                    self.history[track_id]['bbox'] = box

                    if self.history[track_id]['state'] == 'MONITORING' and int(cls) == 1:
                        self.history[track_id]['state'] = 'FALL_DETECTED'
                        self.dtime[track_id] = ctime

                    elif self.history[track_id]['state'] == 'FALL_DETECTED':
                        if ctime - self.dtime[track_id] >= 2.0:
                            self.history[track_id]['state'] = 'MOTION_TRACKING'
                            self.mtime[track_id] = ctime
                            self.last_pos[track_id] = box

                    elif self.history[track_id]['state'] == 'MOTION_TRACKING':
                        elapsed = ctime - self.mtime[track_id]
                        motion = self._detect_motion(frame, track_id)

                        if motion:
                            self._reset(track_id)
                        elif elapsed >= 3.0:
                            with torch.no_grad():
                                frame_tensor = self._preprocess_frame(frame, track_id)
                                output = self.classifier(frame_tensor)
                                emergency_prob = output.item()
                            if emergency_prob <= 0.5:
                                self.history[track_id]['state'] = 'EMERGENCY'

                    annotated_frame = self._draw(annotated_frame, track_id, self._get_state_color(track_id))

        cv2.imshow("Bot Brigade", annotated_frame)

    def _preprocess_frame(self, frame, track_id):
        x1, y1, x2, y2 = map(int, self.history[track_id]['bbox'])
        frame = frame[y1:y2, x1:x2]

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (256, 256))
        frame = frame / 255.0
        frame = np.transpose(frame, (2, 0, 1))

        return torch.tensor(frame, dtype=torch.float32).unsqueeze(0).to(self.device)

    def _draw(self, frame, track_id, color):
        x1, y1, x2, y2 = self.history[track_id]['bbox']

        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            2,
        )

        label = f"ID: {track_id}, State: {self.history[track_id]['state']}"
        label_position = (int(x1), int(y1) - 10)
        cv2.putText(
            frame,
            label,
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            lineType=cv2.LINE_AA
        )

        return frame

    def _get_state_color(self, track_id):
        state = self.history[track_id]['state']
        if state == 'FALL_DETECTED':
            return 0, 165, 255
        elif state == 'MOTION_TRACKING':
            return 0, 255, 255
        elif state == 'EMERGENCY':
            return 0, 0, 255
        return 0, 255, 0

    def _reset(self, track_id):
        self.history[track_id] = {'state': 'MONITORING', 'bbox': None}
        self.dtime[track_id] = 0.0
        self.mtime[track_id] = 0.0
        self.last_pos[track_id] = None
        self.static_back[track_id] = None

    def _detect_motion(self, frame, track_id):
        motion = False
        x1, y1, x2, y2 = map(int, self.history[track_id]['bbox'])

        gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.static_back[track_id] is None or self.static_back[track_id].shape != gray.shape:  # type: ignore
            self.static_back[track_id] = gray  # type: ignore
            return motion

        diff_frame = cv2.absdiff(self.static_back[track_id], gray)  # type: ignore
        thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=1)  # type: ignore
        cnts, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in cnts:
            if cv2.contourArea(contour) < 10000:
                continue
            motion = True

        return motion

    def process_video(self) -> None:
        cap = cv2.VideoCapture(self.video)
        start = time.time()

        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            ctime = time.time() - start
            self._process_frame(frame, ctime)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    Tracker('../data/pipeline_eval_data/sudden cardiac arrest tatami.webm')
