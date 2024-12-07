import time
import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from src.models.classifiers.classifier import CNN


class EmergencyDetectionSystem:
    def __init__(self, config_path, model_path):
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.device = torch.device(self.config['system']['device'])

        # Initialize fall detection model (YOLO)
        self.fall_detector = YOLO('../data/weights/yolo11n.pt')

        # Initialize classifier model
        self.classifier = CNN(self.config).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.classifier.eval()

        # Initialize SORT tracker
        self.detection_states = {}
        self.motion_threshold = 50

    def reset_detection_state(self, track_id):
        """Reset the detection state for a specific track ID."""
        if track_id in self.detection_states:
            del self.detection_states[track_id]

    def process_video(self, video_path):
        """Process the video frame by frame."""
        cap = cv2.VideoCapture(video_path)
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time() - start_time
            results = self.process_frame(frame, current_time)

            self.visualize_frame(frame, results)

            # Display the frame
            cv2.imshow('Emergency Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame, current_time):
        """Process a single video frame."""
        detections = []
        results = self.fall_detector.track(frame)

        for box in results[0].boxes:
            bbox = box.xyxy[0].tolist()
            confidence = box.conf.item()
            if confidence >= 0.5:
                detections.append([*bbox, confidence])

        tracked_objects = self.tracker.update(np.array(detections))

        for track in tracked_objects:
            track_id = int(track[4])
            bbox = tuple(track[:4])

            if track_id not in self.detection_states:
                self.detection_states[track_id] = {
                    'state': 'MONITORING',
                    'fall_detection_time': None,
                    'motion_tracking_start': None,
                    'last_position': None,
                }

            state_data = self.detection_states[track_id]
            current_state = state_data['state']

            # State transitions
            if current_state == 'MONITORING':
                if self.is_fall_detected(bbox, results):
                    state_data['state'] = 'FALL_DETECTED'
                    state_data['fall_detection_time'] = current_time
                    print(f"Fall detected for Track ID {track_id} at {current_time:.2f}s")

            elif current_state == 'FALL_DETECTED':
                if current_time - state_data['fall_detection_time'] >= 2.0:
                    state_data['state'] = 'MOTION_TRACKING'
                    state_data['motion_tracking_start'] = current_time
                    state_data['last_position'] = bbox
                    print(f"Starting motion tracking for Track ID {track_id} at {current_time:.2f}s")

            elif current_state == 'MOTION_TRACKING':
                elapsed_time = current_time - state_data['motion_tracking_start']
                has_motion = self.detect_motion(bbox, state_data)

                if has_motion:
                    self.reset_detection_state(track_id)
                    print(f"Motion detected for Track ID {track_id} at {current_time:.2f}s")
                elif elapsed_time >= 4.0:
                    emergency_prob = self.classify_emergency(frame, bbox)
                    print(f"Emergency probability for Track ID {track_id}: {emergency_prob:.2f}")

                    if emergency_prob <= 0.5:
                        state_data['state'] = 'EMERGENCY'
                        print(f"Emergency confirmed for Track ID {track_id} at {current_time:.2f}s")
                    self.reset_detection_state(track_id)

        return {'states': self.detection_states, 'tracked_objects': tracked_objects}

    def is_fall_detected(self, bbox, results):
        """Determine if a fall is detected."""
        # Check if bbox closely matches a detected fall box
        for box in results[0].boxes:
            if box.cls.item() == 1:
                detected_bbox = box.xyxy.cpu().numpy()[0]
                if np.allclose(detected_bbox, bbox, atol=5):
                    return True
        return False

    def classify_emergency(self, frame, bbox):
        """Classify if a detected fall is an emergency."""
        frame_tensor = self.preprocess_frame(frame, bbox)
        with torch.no_grad():
            output = self.classifier(frame_tensor)
        return output.item()

    def preprocess_frame(self, frame, bbox):
        """Preprocess frame for classifier input."""
        x1, y1, x2, y2 = map(int, bbox)
        person_crop = frame[y1:y2, x1:x2]
        input_size = self.config['model']['classifier']['input_size']

        crop_tensor = cv2.resize(person_crop, (input_size, input_size)) / 255.0
        crop_tensor = torch.tensor(crop_tensor, dtype=torch.float32).permute(2, 0, 1)
        return crop_tensor.unsqueeze(0).to(self.device)

    def detect_motion(self, current_bbox, state_data):
        """Detect motion by comparing positions."""
        last_position = state_data['last_position']
        if last_position is None:
            state_data['last_position'] = current_bbox
            return True

        current_center = self.get_center(current_bbox)
        last_center = self.get_center(last_position)

        distance = np.linalg.norm(np.array(current_center) - np.array(last_center))
        state_data['last_position'] = current_bbox
        return distance > self.motion_threshold

    @staticmethod
    def get_center(bbox):
        """Get the center coordinates of a bounding box."""
        x1, y1, x2, y2 = bbox
        return [(x1 + x2) / 2, (y1 + y2) / 2]

    def visualize_frame(self, frame, results):
        """Visualize tracking results on the video frame."""
        for obj in results['tracked_objects']:
            x1, y1, x2, y2, track_id = map(int, obj[:5])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Ensure track_id exists in detection_states before accessing
            state_info = self.detection_states.get(track_id, {'state': 'Unknown'})
            cv2.putText(
                frame, f"ID: {track_id} State: {state_info['state']}",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )


if __name__ == "__main__":
    config_path = '../../config/config.yaml'
    model_path = '../data/weights/best_model.pth'
    video_path = '../data/pipeline_eval_data/sudden cardiac arrest tatami.webm'

    system = EmergencyDetectionSystem(config_path, model_path)
    system.process_video(video_path)
