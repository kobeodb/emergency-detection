import os
import json
import pandas as pd
import numpy as np
import cv2
from pathlib import Path

from src.application.improved_app import DetectionUtility

# Load ground truth data
with open('ground_truth.json', 'r') as f:
    ground_truth_data = json.load(f)

# Constants for evaluation
MIN_SECS_STATIONARY_BEFORE_ALERT = 4
ACCURACY_IN_SEC_OF_ALERT_REQUIRED = 4
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
        alert_triggered = False

        while cap.isOpened() and not alert_triggered:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.util.yolo_model(frame, conf=0.6)
            detections = results[0].boxes

            for det in detections:
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                cls_name = self.util.yolo_model.names[int(det.cls[0])]

                if cls_name == "Fall Detected":
                    keypoints = self.util.extract_keypoints(frame[y1:y2, x1:x2])
                    if keypoints is not None:
                        prediction = self.util.classifier.predict([keypoints])[0]
                        label = self.util.label_encoder.inverse_transform([prediction])[0]
                        if label == "Need help":
                            detected_alerts.append(frame_count)
                            alert_triggered = True
                            break

            frame_count += 1

        cap.release()
        return detected_alerts


evaluator = Evaluator()


# p = Path('../needs_for_application/vids/Le2i Fall Dataset')
#
# def evaluate_videos(annotations_dir, videos_dir):
#     for txt_file in annotations_dir.iterdir():
#         annotation_name = txt_file.stem
#         corresponding_video = None
#         video_name = None
#
#         for video in videos_dir.iterdir():
#             if video.stem == annotation_name:
#                 corresponding_video = video
#                 video_name = video.stem
#                 break
#
#         if not corresponding_video:
#             print(f"No matching video found for annotation: {txt_file.name}")
#             continue
#
#         with txt_file.open() as f:
#             ground_truth = int(f.readline().strip())
#
#         video_path = str(corresponding_video)
#         alerts_generated = evaluator.run_detection_and_classification(video_path)
#
#         alerts_correct = []
#         alerts_missed = []
#         alerts_false = []
#
#         found = False
#         for alerted_frame in alerts_generated:
#             if abs((ground_truth + round(MIN_SECS_STATIONARY_BEFORE_ALERT * fps_orig)) - alerted_frame) <= (
#                     ACCURACY_IN_SEC_OF_ALERT_REQUIRED * fps_orig):
#                 found = True
#                 alerts_correct.append(ground_truth)
#                 break
#
#         if not found:
#             alerts_missed.append(ground_truth)
#
#         for alerted_frame in alerts_generated:
#             if not any(abs((ground_truth + round(MIN_SECS_STATIONARY_BEFORE_ALERT * fps_orig)) - alerted_frame) <= (
#                     ACCURACY_IN_SEC_OF_ALERT_REQUIRED * fps_orig) for ground_truth in [ground_truth]):
#                 alerts_false.append(alerted_frame)
#
#         video_rename = f"{videos_dir.relative_to(p)}/{video_name}"
#
#         # Store results
#         new_row = {
#             'video': video_rename,
#             'truth': [ground_truth],
#             'found': alerts_generated,
#             'correct': alerts_correct,
#             'false': alerts_false,
#             'missed': alerts_missed
#         }
#         global results_df
#         results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
#
#
# for subdir in p.glob('**/Annotation_files'):
#     annotations_dir = subdir
#     videos_dir = subdir.parent / 'Videos'
#
#     if videos_dir.exists():
#         evaluate_videos(annotations_dir, videos_dir)
#
# # Save results to a CSV file
# results_df.to_csv('evaluation_results.csv', index=False)

# Run evaluation
for vid, ground_truth in ground_truth_data.items():
    video_path = f'../needs_for_application/vids/evaluation_videos/{vid}'
    alerts_generated = evaluator.run_detection_and_classification(video_path)  # for now 1 frame (found)

    alerts_correct = []
    alerts_missed = []
    alerts_false = []

    for ground_truth_alert_frame in ground_truth:  # 1 frame (ground truth) for now
        found = False
        for alerted_frame in alerts_generated:  # 1 frame for now (found)
            if abs((ground_truth_alert_frame + round(MIN_SECS_STATIONARY_BEFORE_ALERT * fps_orig)) - alerted_frame) <= (
                    ACCURACY_IN_SEC_OF_ALERT_REQUIRED * fps_orig):
                found = True
                break
        if found:
            alerts_correct.append(ground_truth_alert_frame)
        else:
            alerts_missed.append(ground_truth_alert_frame)

    for alerted_frame in alerts_generated:
        was_alert = False
        for alert_frame in ground_truth:
            if abs((alert_frame + round(MIN_SECS_STATIONARY_BEFORE_ALERT * fps_orig)) - alerted_frame) <= (
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
tot_truth = len(results_df)
tot_found = sum(len(row['found']) for _, row in results_df.iterrows())
tot_correct = sum(len(row['correct']) for _, row in results_df.iterrows())
tot_missed = sum(len(row['missed']) for _, row in results_df.iterrows())
tot_false = sum(len(row['false']) for _, row in results_df.iterrows())

# Count cases where no ground truth and no alerts were found
correct_no_alerts = sum(1 for _, row in results_df.iterrows() if not row['truth'] and not row['found'])

tot_correct += correct_no_alerts

recall = round((tot_correct / tot_truth) * 100, 2) if tot_truth > 0 else None
precision = round((tot_correct / (tot_found + correct_no_alerts)) * 100, 2) if (tot_found + correct_no_alerts) > 0 else None
false_alert_rate = round((tot_false / tot_found) * 100, 2) if tot_found > 0 else None

print('Evaluation Results:')
for vid in results_df['video']:
    print(f"Video: {vid}")
print(f"Precision: {precision}%")
print(f"Recall: {recall}%")
print(f"False Alert Rate: {false_alert_rate}%")
print(results_df)

results_df.to_csv('evaluation_results.csv', index=False)