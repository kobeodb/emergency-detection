from typing import List

import cv2
import os

from pathlib import Path


def extract_frames(path: str, out: str, frame_rate: int = 3) -> None:
    video = cv2.VideoCapture(path)
    os.makedirs(out, exist_ok=True)
    frame_id = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        if frame_id % frame_rate == 0:
            frame_path = os.path.join(out, f"frame_{frame_id}.jpg")
            cv2.imwrite(frame_path, frame)
        frame_id += 1
    video.release()


def process_videos(videos: List[str], out: str, frame_rate: int = 1) -> None:
    for v in videos:
        name = Path(v).stem
        v_out = os.path.join(out, name)
        extract_frames(v, v_out, frame_rate)
