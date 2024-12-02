import time

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO
import torch.nn.functional as func
from src.models.classifiers.classifier import CNN


class EmergencyDetection:
    def __init__(self, config_path, model_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device(self.config['system']['device'])

        # Initialize YOLOv8 for fall detection
        self.fall_detector = YOLO(self.config['model']['detector']['finetuned_weights_path'])

        # Load CNN classifier for emergency detection
        self.classifier = CNN(self.config).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.classifier.eval()

        self.state = 'MONITORING'
        self.fall_detection_time = None
        self.motion_tracking_start = None
        self.last_position = None
        self.motion_threshold = 50 #frames

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
        # Perform YOLO detection
        results = self.fall_detector(frame)

        if len(results[0].boxes) > 0:
            bbox = results[0].boxes.xyxy[0].cpu().numpy()
            confidence = results[0].boxes[0].conf[0].item()
            fall_detected = results[0].boxes[0].cls[0].item() == 1

            if confidence < 0.5:
                return {'state': self.state, 'bbox': None}
        else:
            return {'state': self.state, 'bbox': None}

        if self.state == 'MONITORING' and fall_detected:
            self.state = 'FALL_DETECTED'
            self.fall_detection_time = current_time
            print(f"Fall detected at {current_time:.2f}s with confidence: {confidence}")

        elif self.state == 'FALL_DETECTED':
            if current_time - self.fall_detection_time >= 2.0:
                self.state = 'MOTION_TRACKING'
                self.motion_tracking_start = current_time
                self.last_position = bbox
                print(f"Starting motion tracking at {current_time:.2f}s")

        elif self.state == 'MOTION_TRACKING':
            elapsed_time = current_time - self.motion_tracking_start
            has_motion = self.detect_motion(bbox)

            if has_motion:
                self.reset()
                print(f"Motion detected at {current_time:.2f}s")
            elif elapsed_time >= 4.0:
                self.classifier.eval()
                with torch.no_grad():
                    frame_tensor = self.preprocess_frame(frame, bbox)
                    output = self.classifier(frame_tensor)
                    emergency_prob = output.item()
                    print(f"Emergency probability: {emergency_prob:.2f}")

                if emergency_prob <= 0.5:
                    self.state = 'EMERGENCY'
                    print(f"Emergency confirmed at {current_time:.2f}s with confidence {emergency_prob:.2f}")
                    return {
                        'state': self.state,
                        'confidence': emergency_prob,
                        'bbox': bbox
                    }
                self.reset()

        return {
            'state': self.state,
            'bbox': bbox,
            'confidence': confidence
        }

    def preprocess_frame(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        person_crop = frame[y1:y2, x1:x2]

        # Resize and normalize the cropped image for CNN
        crop_tensor = cv2.resize(person_crop, (
            self.config['model']['classifier']['input_size'],
            self.config['model']['classifier']['input_size']))
        crop_tensor = crop_tensor / 255.0
        crop_tensor = torch.tensor(crop_tensor, dtype=torch.float32).permute(2, 0, 1)
        crop_tensor = crop_tensor.unsqueeze(0).to(self.device)  # Add batch dimension

        return crop_tensor

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

    def _visualize_frame(self, frame, results):
        if results['bbox'] is not None:
            x1, y1, x2, y2 = map(int, results['bbox'])
            confidence = results.get('confidence', 0)

            colors = {
                'MONITORING': (0, 255, 0),
                'FALL_DETECTED': (0, 165, 255),
                'MOTION_TRACKING': (0, 255, 255),
                'EMERGENCY': (0, 0, 255)
            }

            color = colors[self.state]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

            cv2.putText(frame, f"State: {self.state}", (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if self.state == 'FALL_DETECTED':
                cv2.putText(frame, f"Wait Time: {time.time() - self.fall_detection_time:.1f}s",
                            (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            elif self.state == 'MOTION_TRACKING':
                cv2.putText(frame, f"Track Time: {time.time() - self.motion_tracking_start:.1f}s",
                            (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            elif self.state == 'EMERGENCY':
                cv2.putText(frame, f"EMERGENCY DETECTED", (x1, y1 - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


if __name__ == "__main__":
    system = EmergencyDetection('../../config/config.yaml', '../models/classifiers/best_model.pth')
    system.process_video('../data/pipeline_eval_data/test_videos/positive_8_falling_away_from_camera.mp4')