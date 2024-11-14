import os
import json
import pandas as pd
import numpy as np
import cv2

from src.application.improved_app import DetectionUtility

# Load ground truth data
with open('ground_truth.json', 'r') as f:
    ground_truth_data = json.load(f)

# Constants for evaluation
MIN_SECS_STATIONARY_BEFORE_ALERT = 4
ACCURACY_IN_SEC_OF_ALERT_REQUIRED = 1
fps_orig = 30

# Initialize results DataFrame
results_df = pd.DataFrame(columns=['video', 'truth', 'found', 'correct', 'false', 'missed'])


class Evaluator:
    def __init__(self):
        self.util = DetectionUtility()

    def run_detection_and_classification(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        detected_alerts = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.util.yolo_model(frame, conf=0.6)
            detections = results[0].boxes

            for det in detections:
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                cls_name = self.util.yolo_model.names[int(det.cls[0])]

                if cls_name == "Person":
                    keypoints = self.util.extract_keypoints(frame[y1:y2, x1:x2])
                    if keypoints is not None:
                        prediction = self.util.classifier.predict([keypoints])[0]
                        label = self.util.label_encoder.inverse_transform([prediction])[0]
                        if label == "Need help":
                            detected_alerts.append(frame_count)
                            break

            frame_count += 1

        cap.release()
        return detected_alerts


evaluator = Evaluator()

# Run evaluation
for vid, ground_truth in ground_truth_data.items():
    video_path = f'../needs_for_application/vids/positive/{vid}'
    alerts_generated = evaluator.run_detection_and_classification(video_path)

    alerts_correct = []
    alerts_missed = []
    alerts_false = []

    for alert_frame in ground_truth:
        found = False
        for alerted_frame in alerts_generated:
            if abs((alert_frame + round(MIN_SECS_STATIONARY_BEFORE_ALERT * fps_orig)) - alerted_frame) < (
                    ACCURACY_IN_SEC_OF_ALERT_REQUIRED * fps_orig):
                found = True
                break
        if found:
            alerts_correct.append(alert_frame)
        else:
            alerts_missed.append(alert_frame)

    for alerted_frame in alerts_generated:
        was_alert = False
        for alert_frame in ground_truth:
            if abs((alert_frame + round(MIN_SECS_STATIONARY_BEFORE_ALERT * fps_orig)) - alerted_frame) < (
                    ACCURACY_IN_SEC_OF_ALERT_REQUIRED * fps_orig):
                was_alert = True
                break
        if not was_alert:
            alerts_false.append(alerted_frame)

    # Store results
    new_row = {
        'video': vid,
        'truth': ground_truth,
        'found': alerts_generated,
        'correct': alerts_correct,
        'false': alerts_false,
        'missed': alerts_missed
    }
    results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

# Calculate metrics
tot_truth = sum(len(row['truth']) for _, row in results_df.iterrows())
tot_found = sum(len(row['found']) for _, row in results_df.iterrows())
tot_correct = sum(len(row['correct']) for _, row in results_df.iterrows())
tot_missed = sum(len(row['missed']) for _, row in results_df.iterrows())
tot_false = sum(len(row['false']) for _, row in results_df.iterrows())

recall = round((tot_correct / tot_truth) * 100, 2) if tot_truth > 0 else None
precision = round((tot_correct / tot_found) * 100, 2) if tot_found > 0 else None
false_alert_rate = round((tot_false / tot_found) * 100, 2) if tot_found > 0 else None

print('Evaluation Results:')
for vid in results_df['video']:
    print(f"Video: {vid}")
print(f"Precision: {precision}%")
print(f"Recall: {recall}%")
print(f"False Alert Rate: {false_alert_rate}%")

# Save results
results_df.to_csv('evaluation_results.csv', index=False)