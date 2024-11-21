import mediapipe as mp
import cv2
import torch
import numpy as np
import time
import yaml
from ultralytics import YOLO
from src.models.classifiers.classifier import SingleFrameClassifier

class EmergencyDetection:
    def __init__(self, config_path, model_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device(self.config['system']['device'])

        self.fall_detector = YOLO(self.config['model']['detector']['finetuned_weights_path'])

        self.classifier = SingleFrameClassifier(self.config)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.classifier.eval()

        self.state = 'MONITORING'
        self.fall_detection_time = None
        self.motion_tracking_start = None
        self.last_position = None
        self.motion_threshold = 50 #these are pixels

        self.reset()

    def reset(self):
        self.state = 'MONITORING'
        self.fall_detection_time = None
        self.motion_tracking_start = None
        self.last_position = None

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time() - start_time
            results = self.process_frame(frame, current_time)

            self._visualize_frame(frame, results)

            cv2.imshow('Emergency Detection Pipeline', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame, current_time):
        results = self.fall_detector(frame)

        if len(results[0].boxes) > 0:
            bbox = results[0].boxes.xyxy[0].cpu().numpy()
            confidence = results[0].boxes[0].conf[0].item()
            fall_detected = results[0].boxes[0].cls[0].item() == 0
        else:
            return {'state': self.state, 'bbox': None}


        if self.state == 'MONITORING':
            if fall_detected:
                self.state = 'FALL_DETECTED'
                self.fall_detection_time = current_time
                print(f"Fall detected at {current_time:.2f}s with confidence: {confidence}")

        elif self.state == 'FALL_DETECTED':
            wait_time = current_time - self.fall_detection_time
            if wait_time >= 2.0:
                self.state = 'MOTION_TRACKING'
                self.motion_tracking_start = current_time
                self.last_position = bbox
                print(f"Starting motion tracking at {current_time:.2f}s")

        elif self.state == 'MOTION_TRACKING':
            elapsed_time = current_time - self.motion_tracking_start
            has_motion = self.detect_motion(bbox)

            if has_motion:
                self.state = 'MONITORING'
                print(f"Motion detected at {current_time:.2f}s")
            elif elapsed_time >= 4.0:
                with torch.no_grad():
                    frame_tensor = self.preprocess_frame(frame, bbox)
                    emergency_prob = self.classifier(frame_tensor).item()
                    print(f"Emergency probability: {emergency_prob}")
                if emergency_prob <= 0.5: # >
                    print(f"Emergency confirmed at {current_time:.2f}s with confidence {emergency_prob:.2f}")
                    self.state = 'EMERGENCY'
                    return {
                        'state': self.state,
                        'confidence': emergency_prob,
                        'bbox': bbox
                    }
                self.state = 'MONITORING'

        return {
            'state': self.state,
            'bbox': bbox,
            'confidence': confidence
        }

    def detect_motion(self, current_bbox):
        if self.last_position is None:
            self.last_position = current_bbox
            return True

        current_center = [(current_bbox[0] + current_bbox[2]) / 2,
                          (current_bbox[1] + current_bbox[3]) / 2]
        last_center = [(self.last_position[0] + self.last_position[2]) / 2,
                       (self.last_position[1] + self.last_position[3]) / 2]

        distance = np.sqrt((current_center[0] - last_center[0]) ** 2 +
                           (current_center[1] - last_center[1]) ** 2)

        self.last_position = current_bbox
        return distance > self.motion_threshold

    def preprocess_frame(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        person_crop = frame[y1:y2, x1:x2]

        mp_pose = mp.solutions.pose
        with mp_pose.Pose(
                static_image_mode=True,
                model_complexity=1,
        ) as pose:
            rgb_frame = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if not results.pose_landmarks: #type: ignore
                return None

            h, w = person_crop.shape[:2]
            keypoints = np.array([[lm.x * w, lm.y * h] for lm in results.pose_landmarks.landmark]) #type: ignore

            keypoints[:, 0] /= w
            keypoints[:, 1] /= h

            keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32)
            keypoints_tensor = keypoints_tensor.unsqueeze(0)

            return keypoints_tensor

    def _visualize_frame(self, frame, results):
        if results['bbox'] is not None:
            x1, y1, x2, y2 = map(int, results['bbox'])
            confidence = results.get('confidence', 0)

            colors = {
                'MONITORING': (0, 255, 0),  #green
                'FALL_DETECTED': (0, 165, 255),  #orange
                'MOTION_TRACKING': (0, 255, 255),  #yellow
                'EMERGENCY': (0, 0, 255) #red
            }

            color =  colors[self.state]


            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

            cv2.putText(frame, f"State: {self.state}", (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(frame, f"Confidence: {confidence}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


            if self.state == 'FALL_DETECTED':
                wait_time = time.time() - self.fall_detection_time
                cv2.putText(frame, f"Wait Time: {wait_time:.1f}s", (x1, y1 - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            elif self.state == 'MOTION_TRACKING':
                track_time = time.time() - self.motion_tracking_start
                cv2.putText(frame, f"Track Time: {track_time:.1f}s", (x1, y1 - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            elif self.state == 'EMERGENCY':
                cv2.putText(frame, f"EMERGENCY DETECTED", (x1, y1 - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

if __name__ == "__main__":
    system = EmergencyDetection('../../config/config.yaml', '../models/classifiers/best_model.pth')
    system.process_video('../data/pipeline_eval_data/test_videos/simulation_chantier_2.mp4')