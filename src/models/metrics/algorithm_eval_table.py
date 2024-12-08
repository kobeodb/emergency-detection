import time
from pathlib import Path
import yaml
import json
import pandas as pd
import cv2
# from src.pipeline.inference import EmergencyDetection
from src.pipeline.main import EmergencyDetector

class EvaluationTable:
    def __init__(self, config_path, video_dir):
        self.config_path = config_path
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)

        self.model_path = self.config['model']['classifier']['model_path']
        self.pipeline = EmergencyDetector(config_path, self.model_path)
        self.video_dir = Path(video_dir)

        ground_truth_path = self.config['evaluation']['ground_truth_path']
        with open(ground_truth_path) as f:
            self.ground_truth = json.load(f)

        self.FRAME_THRESHOLD = 180  # Frames, given 30 frames per second, this equals 6 seconds
        self.results_df = pd.DataFrame(columns=['video', 'truth', 'found', 'correct', 'false', 'missed'])

    def run_inference(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        start_time = time.time()
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        detected_alerts = []
        frame_count = 0
        stop_processing = False

        while cap.isOpened():
            if stop_processing:
                break

            ret, frame = cap.read()
            if not ret:
                break

            try:
                current_time = time.time() - start_time
                frame_results = self.pipeline.process_frame(frame, current_time)

                if frame_results and isinstance(frame_results, dict):
                    for detection in frame_results.get('detections', []):
                        if detection.get('state') == 'EMERGENCY':
                            detected_alerts.append(frame_count)
                            stop_processing = True
                            break
            except cv2.error as e:
                print(f"Warning: OpenCV error in frame {frame_count}: {str(e)}")
                self.pipeline.history.clear()
                self.pipeline.dtime.clear()
                self.pipeline.mtime.clear()
                self.pipeline.last_pos.clear()
                self.pipeline.static_back.clear()
            except Exception as e:
                print(f"Warning: Unexpected error in frame {frame_count}: {str(e)}")

            frame_count += 1

        cap.release()
        return detected_alerts

    def evaluate_all_videos(self):
        for video_name, ground_truth_frame in self.ground_truth.items():
            video_path = self.video_dir / video_name
            alerts_generated = self.run_inference(video_path)

            alerts_correct = []
            alerts_false = []
            alerts_missed = []

            for gt_frame in ground_truth_frame:
                found = False
                for alert_frame in alerts_generated:
                    if abs(gt_frame - alert_frame) <= self.FRAME_THRESHOLD:
                        found = True
                        alerts_correct.append(gt_frame)
                        break
                if not found:
                    alerts_missed.append(gt_frame)

            for alert_frame in alerts_generated:
                was_correct = False
                for gt_frame in ground_truth_frame:
                    if abs(gt_frame - alert_frame) <= self.FRAME_THRESHOLD:
                        was_correct = True
                        break
                if not was_correct:
                    alerts_false.append(alert_frame)

            new_row = {
                'video': video_name,
                'truth': ground_truth_frame,
                'found': alerts_generated,
                'correct': alerts_correct,
                'false': alerts_false,
                'missed': alerts_missed
            }
            self.results_df = pd.concat([self.results_df, pd.DataFrame([new_row])], ignore_index=True)

            self.pipeline.history.clear()
            self.pipeline.dtime.clear()
            self.pipeline.mtime.clear()
            self.pipeline.last_pos.clear()
            self.pipeline.static_back.clear()

        return self._calculate_metrics()

    def _calculate_metrics(self):
        tot_truth = sum(len(row['truth']) for _, row in self.results_df.iterrows())
        tot_found = sum(len(row['found']) for _, row in self.results_df.iterrows())
        tot_correct = sum(len(row['correct']) for _, row in self.results_df.iterrows())
        tot_false = sum(len(row['false']) for _, row in self.results_df.iterrows())
        tot_missed = sum(len(row['missed']) for _, row in self.results_df.iterrows())

        metrics = {
            'precision': round((tot_correct / (tot_correct + tot_false)) * 100, 2) if tot_found > 0 else 0,
            'recall': round((tot_correct / (tot_correct + tot_missed)) * 100, 2) if tot_truth > 0 else 0,
            'false_alert_rate': round((tot_false / tot_found) * 100, 2) if tot_found > 0 else 0
        }

        results_path = Path(self.config['evaluation']['results_path'])
        self.results_df.to_csv(results_path, index=False)

        return metrics

    def print_results(self):
        """Print evaluation results in a readable format"""
        metrics = self.evaluate_all_videos()

        print('\nEvaluation Results:')
        print('-' * 50)
        print(f"Precision: {metrics['precision']}%")
        print(f"Recall: {metrics['recall']}%")
        print(f"False Alert Rate: {metrics['false_alert_rate']}%")
        print('\nResult Table:')
        print('-' * 50)

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(self.results_df)


if __name__ == '__main__':
    evaluator = EvaluationTable('../../../config/config.yaml', '../../data/pipeline_eval_data/test_videos')
    evaluator.print_results()

