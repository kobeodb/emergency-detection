import sys
import cv2
import numpy as np
from pydantic import BaseModel

import ultralytics

class GetKeypoint(BaseModel):
    NOSE:           int = 0
    LEFT_EYE:       int = 1
    RIGHT_EYE:      int = 2
    LEFT_EAR:       int = 3
    RIGHT_EAR:      int = 4
    LEFT_SHOULDER:  int = 5
    RIGHT_SHOULDER: int = 6
    LEFT_ELBOW:     int = 7
    RIGHT_ELBOW:    int = 8
    LEFT_WRIST:     int = 9
    RIGHT_WRIST:    int = 10
    LEFT_HIP:       int = 11
    RIGHT_HIP:      int = 12
    LEFT_KNEE:      int = 13
    RIGHT_KNEE:     int = 14
    LEFT_ANKLE:     int = 15
    RIGHT_ANKLE:    int = 16


class DetectKeypoint:
    def __init__(self, yolov11_model='yolo11n-pose'):
        self.yolov11_model = yolov11_model
        self.get_keypoint = GetKeypoint()
        self.__load_model()

    def __load_model(self):
        self.model = ultralytics.YOLO(model=self.yolov11_model)

    def extract_keypoint(self, keypoint: np.ndarray) -> list:
        extracted_keypoints = []

        for key in self.get_keypoint.model_fields.values():
            try:
                x, y = keypoint[key.default]
                extracted_keypoints.extend([x, y])
            except IndexError:

                print(f"Keypoint is missing. Using default values (0, 0).")
                extracted_keypoints.extend([0, 0])

        return extracted_keypoints

    def get_xy_keypoint(self, results) -> list:
        result_keypoint = results.keypoints.xyn.cpu().numpy()[0]
        keypoint_data = self.extract_keypoint(result_keypoint)
        return keypoint_data

    def __call__(self, image: np.array):
        results = self.model.predict(image, save=False)[0]
        return results