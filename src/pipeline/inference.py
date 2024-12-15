import time
from collections import defaultdict

import cv2
import torch
import yaml
from ultralytics import YOLO

from src.models.classifiers.img_classifier.classifier import CNN

class EmergencyDetector:
    def __init__(self, config_path, model_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device(self.config['system']['device'])

        self.fall_detector = YOLO(self.config['model']['detector']['finetuned_weights_path'])

        self.dtime = defaultdict(float)
        self.mtime = defaultdict(float)

        self.history = defaultdict(lambda: {'state': 'MONITORING', 'bbox': None})
        self.last_pos = defaultdict(lambda: None)
        self.static_back = defaultdict(lambda: None)

        # input_size = self.config['model']['classifier']['input_size']
        # self.classifier = make_model(trial=None, input_size=input_size).to(self.config['system']['device'])
        # checkpoint = torch.load(model_path, map_location=self.config['system']['device'])
        # self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        # self.classifier.eval()

        self.classifier = CNN(self.config).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.classifier.eval()


    def reset(self, track_id):
        self.history[track_id] = {'state': 'MONITORING', 'bbox': None}
        self.dtime[track_id] = 0.0
        self.mtime[track_id] = 0.0
        self.last_pos[track_id] = None
        self.static_back[track_id] = None

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        start_time = time.time()

        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            ctime = time.time() - start_time
            self.process_frame(frame, ctime)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame, ctime):
        try:
            if frame.shape[0] > 1080:
                scale = 1080 / frame.shape[0]
                width = int(frame.shape[1] * scale)
                frame = cv2.resize(frame, (width, 1080))

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.fall_detector.track(
                frame_rgb,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
                conf=0.5,
                iou=0.4,
            )

            annotated_frame = frame.copy()
            detections = []

            for result in results:
                if not result.boxes or result.boxes.id is None:
                    continue

                for bbox, cls, track_id, conf in zip(
                    result.boxes.xyxy.cpu().numpy(),
                    result.boxes.cls.cpu().numpy(),
                    result.boxes.id.cpu().numpy(),
                    result.boxes.conf.cpu().numpy()
            ):
                    track_id = int(track_id)
                    if track_id not in self.history:
                        self.reset(track_id)

                    self.history[track_id]['bbox'] = bbox
                    state = self.history[track_id]['state']

                    if state == 'MONITORING' and int(cls) == 1:
                        self.history[track_id]['state'] = 'FALL_DETECTED'
                        self.dtime[track_id] = ctime
                        print(f"Fall detected for bbox {bbox} at {ctime:.2f}s with confidence: {conf}")

                    elif state == 'FALL_DETECTED':
                        fall_time = ctime - self.dtime[track_id]
                        if fall_time >= 2.0:
                            self.history[track_id]['state'] = 'MOTION_TRACKING'
                            self.mtime[track_id] = ctime
                            self.last_pos[track_id] = bbox
                            print(f"Starting motion tracking for bbox {bbox} at {ctime:.2f}s")

                    elif state == 'MOTION_TRACKING':
                        elapsed = ctime - self.mtime[track_id]
                        track_time = elapsed
                        has_motion = self.detect_motion(frame, track_id)
                        print(elapsed)

                        if has_motion:
                            self.reset(track_id)
                            print(f"Motion detected for bbox {bbox} at {ctime:.2f}s")
                        elif elapsed >= 4.0:
                            with torch.no_grad():
                                frame_tensor = self.preprocess_frame(frame, track_id)
                                output = self.classifier(frame_tensor)
                                emergency_prob = output.item()
                                print(f"Emergency probability for bbox {bbox}: {emergency_prob:.2f}")

                                if emergency_prob <= 0.5:
                                    self.history[track_id]['state'] = 'EMERGENCY'
                                    print(
                                        f"Emergency confirmed for bbox {bbox} at {ctime:.2f}s with confidence {emergency_prob:.2f}")
                                else:
                                    print("no emergency for this person he okokokokok")
                                    self.reset(track_id)


                    x1, y1, x2, y2 = bbox
                    color = (0, 255, 0)
                    if self.history[track_id]['state'] == 'FALL_DETECTED':
                        color = (0, 165, 255)
                    elif self.history[track_id]['state'] == 'MOTION_TRACKING':
                        color = (0, 255, 255)
                    elif self.history[track_id]['state'] == 'EMERGENCY':
                        color = (0, 0, 255)

                    cv2.rectangle(
                        annotated_frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        color,
                        2,
                    )

                    label = f"ID: {track_id}, State: {self.history[track_id]['state']}"
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

                    detections.append({
                        'bbox': bbox,
                        'class': int(cls),
                        'track_id': track_id,
                        'confidence': conf,
                        'state': self.history[track_id]['state']
                    })

                cv2.imshow("Bot Brigade", annotated_frame)

            return {'detections': detections}

        except cv2.error as e:
            print(f"OpenCV error: {str(e)}")
            self.fall_detector = YOLO(self.config['model']['detector']['finetuned_weights_path'])
            return {'detections': []}

        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return {'detections': []}


    def preprocess_frame(self, frame, track_id):
        x1, y1, x2, y2 = map(int, self.history[track_id]['bbox'])
        person_crop = frame[y1:y2, x1:x2]

        crop_tensor = cv2.resize(person_crop, (
            self.config['model']['classifier']['input_size'],
            self.config['model']['classifier']['input_size']))
        crop_tensor = crop_tensor / 255.0
        crop_tensor = torch.tensor(crop_tensor, dtype=torch.float32).permute(2, 0, 1)
        crop_tensor = crop_tensor.unsqueeze(0).to(self.device)

        return crop_tensor

    def detect_motion(self, frame, track_id):
        motion = False
        x1, y1, x2, y2 = map(int, self.history[track_id]['bbox'])

        gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.static_back[track_id] is None or self.static_back[track_id].shape != gray.shape:  # type: ignore
            self.static_back[track_id] = gray
            return motion

        diff_frame = cv2.absdiff(self.static_back[track_id], gray)
        thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=1)
        cnts, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in cnts:
            if cv2.contourArea(contour) < 10000:
                continue
            motion = True

        return motion


if __name__ == "__main__":
    system = EmergencyDetector('../../config/config.yaml', '../models/classifiers/img_classifier/best_model.pth')
    system.process_video("../data/pipeline_eval_data/test_videos/simulation_chantier.mp4")