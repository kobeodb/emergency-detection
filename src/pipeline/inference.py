import time
import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from src.models.classifiers.classifier import CNN


class EmergencyDetection:
    def __init__(self, config_path, model_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device(self.config['system']['device'])

        self.fall_detector = YOLO(self.config['model']['detector']['finetuned_weights_path'])

        self.classifier = CNN(self.config).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.classifier.eval()

        self.detection_states = {}
        self.motion_threshold = 50

    def reset_detection(self, bbox):
        """Reset the state for a specific detection."""
        if bbox in self.detection_states:
            del self.detection_states[bbox]

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

        detections = []
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                bbox = tuple(box.xyxy.cpu().numpy()[0])
                confidence = box.conf.item()
                fall_detected = box.cls.item() == 1

                if confidence >= 0.5:
                    detections.append({
                        'bbox': bbox,
                        'confidence': confidence,
                        'fall_detected': fall_detected
                    })

        for bbox, state_data in self.detection_states.items():
            print(f"BBOX: {bbox}, Current State: {state_data['state']}, Detection Data: {state_data}")

        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            fall_detected = detection['fall_detected']

            if bbox not in self.detection_states:
                self.detection_states[bbox] = {
                    'state': 'MONITORING',
                    'fall_detection_time': None,
                    'motion_tracking_start': None,
                    'last_position': None
                }

            state_data = self.detection_states[bbox]
            current_state = state_data['state']

            if current_state == 'MONITORING':
                if fall_detected:
                    state_data['state'] = 'FALL_DETECTED'
                    state_data['fall_detection_time'] = current_time
                    print(f"Fall detected for bbox {bbox} at {current_time:.2f}s with confidence: {confidence}")

            elif current_state == 'FALL_DETECTED':
                if current_time - state_data['fall_detection_time'] >= 2.0:
                    state_data['state'] = 'MOTION_TRACKING'
                    state_data['motion_tracking_start'] = current_time
                    state_data['last_position'] = bbox
                    print(f"Starting motion tracking for bbox {bbox} at {current_time:.2f}s")

            elif current_state == 'MOTION_TRACKING':
                elapsed_time = current_time - state_data['motion_tracking_start']
                has_motion = self.detect_motion(bbox, state_data)

                if has_motion:
                    self.reset_detection(bbox)
                    print(f"Motion detected for bbox {bbox} at {current_time:.2f}s")
                elif elapsed_time >= 4.0:
                    self.classifier.eval()
                    with torch.no_grad():
                        frame_tensor = self.preprocess_frame(frame, bbox)
                        output = self.classifier(frame_tensor)
                        emergency_prob = output.item()
                        print(f"Emergency probability for bbox {bbox}: {emergency_prob:.2f}")

                    if emergency_prob <= 0.5:
                        state_data['state'] = 'EMERGENCY'
                        print(
                            f"Emergency confirmed for bbox {bbox} at {current_time:.2f}s with confidence {emergency_prob:.2f}")
                        return {
                            'state': state_data['state'],
                            'confidence': emergency_prob,
                            'bbox': bbox
                        }
                    self.reset_detection(bbox)

        return {
            'states': self.detection_states,
            'detections': detections
        }

    def preprocess_frame(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        person_crop = frame[y1:y2, x1:x2]

        crop_tensor = cv2.resize(person_crop, (
            self.config['model']['classifier']['input_size'],
            self.config['model']['classifier']['input_size']))
        crop_tensor = crop_tensor / 255.0
        crop_tensor = torch.tensor(crop_tensor, dtype=torch.float32).permute(2, 0, 1)
        crop_tensor = crop_tensor.unsqueeze(0).to(self.device)

        return crop_tensor

    def detect_motion(self, current_bbox, state_data):
        if state_data['last_position'] is None:
            state_data['last_position'] = current_bbox
            return True

        current_center = [(current_bbox[0] + current_bbox[2]) / 2,
                          (current_bbox[1] + current_bbox[3]) / 2]
        last_center = [(state_data['last_position'][0] + state_data['last_position'][2]) / 2,
                       (state_data['last_position'][1] + state_data['last_position'][3]) / 2]

        distance = np.sqrt((current_center[0] - last_center[0]) ** 2 +
                           (current_center[1] - last_center[1]) ** 2)

        state_data['last_position'] = current_bbox
        return distance > self.motion_threshold

    def _visualize_frame(self, frame, results):
        colors = {
            'MONITORING': (0, 255, 0),  # Green
            'FALL_DETECTED': (0, 165, 255),  # Orange
            'MOTION_TRACKING': (0, 255, 255),  # Yellow
            'EMERGENCY': (0, 0, 255)  # Red
        }

        for detection in results['detections']:
            bbox = detection['bbox']
            confidence = detection['confidence']
            state = self.detection_states[bbox]['state']
            color = colors[state]

            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

            cv2.putText(frame, f"State: {state}", (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if state == 'FALL_DETECTED':
                fall_time = time.time() - self.detection_states[bbox]['fall_detection_time']
                cv2.putText(frame, f"Wait Time: {fall_time:.1f}s", (x1, y1 - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            elif state == 'MOTION_TRACKING':
                track_time = time.time() - self.detection_states[bbox]['motion_tracking_start']
                cv2.putText(frame, f"Track Time: {track_time:.1f}s", (x1, y1 - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            elif state == 'EMERGENCY':
                cv2.putText(frame, "EMERGENCY DETECTED", (x1, y1 - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


if __name__ == "__main__":
    system = EmergencyDetection('../../config/config.yaml', '../models/classifiers/best_model.pth')
    system.process_video("../data/pipeline_eval_data/test_videos/sudden cardiac arrest tatami.webm")
