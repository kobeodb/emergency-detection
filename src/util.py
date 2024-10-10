import cv2
import os

def extract_frames(video_path, output_dir, frame_rate=1):
    video = cv2.VideoCapture(video_path)
    os.makedirs(output_dir, exist_ok=True)
    frame_id = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        if frame_id % frame_rate == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_id}.jpg")
            cv2.imwrite(frame_path, frame)
        frame_id += 1
    video.release()