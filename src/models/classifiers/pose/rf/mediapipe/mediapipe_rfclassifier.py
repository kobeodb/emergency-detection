import os
from typing import Tuple

import cv2
import joblib
import mediapipe as mp
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import optuna

from pathlib import Path



mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

LANDMARKS = [0, 11, 12, 23, 24, 25, 26]


def extract_kps(image):
    result = pose.process(image)
    kps = np.zeros(len(LANDMARKS) * 3)

    if result.pose_landmarks:
        for i, landmark_index in enumerate(LANDMARKS):
            landmark = result.pose_landmarks.landmark[landmark_index]
            kps[i * 3:i * 3 + 3] = [landmark.x, landmark.y, landmark.z]

    return kps


def augment_image(image):
    if np.random.rand() > 0.5:
        image = cv2.flip(image, 1)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

    brightness_factor = np.random.uniform(0.7, 1.3)
    hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * brightness_factor, 0, 255)

    saturation_factor = np.random.uniform(0.7, 1.3)
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255)

    augmented_image = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return augmented_image


def load_data(fd: str, nhd: str, augment=False) -> Tuple[np.array, np.array]:
    data = []
    labels = []
    for label, _dir in enumerate([fd, nhd]):
        for filename in os.listdir(_dir):
            if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
                img_path = os.path.join(_dir, filename)
                img = cv2.imread(img_path)
                kps = extract_kps(img)

                if augment:
                    for _ in range(3):
                        augmented_img = augment_image(img)
                        augmented_kps = extract_kps(augmented_img)
                        data.append(augmented_kps)
                        labels.append(label)

                data.append(kps)
                labels.append(label)

    return np.array(data), np.array(labels)


def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])

    model = RandomForestClassifier(n_estimators=n_estimators,
                                   max_depth=max_depth,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf,
                                   max_features=max_features,
                                   random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    return acc


if __name__ == '__main__':
    current_file = Path().resolve()

    pose_dir = current_file

    while pose_dir.name != 'src' and pose_dir != pose_dir.parent:
        pose_dir = pose_dir.parent

    data_path = Path(pose_dir / 'dataset')

    fine_dir = Path(data_path / 'fine')
    print(fine_dir)

    needhelp_dir = Path(data_path / 'needhelp')
    print(needhelp_dir)


    X, y = load_data(fine_dir, needhelp_dir, augment=True)
    print('loaded correctly')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print('splitted the data')

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    best_params = study.best_params

    print(f"Best hyperparameters: {study.best_params}")
    print(f"Best accuracy: {study.best_value}")

    best_rf_model = RandomForestClassifier(**best_params, random_state=42)
    best_rf_model.fit(X_train, y_train)

    y_pred = best_rf_model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Final model accuracy: {accuracy}")

    print(classification_report(y_val, y_pred))

    model_dump_path = Path(pose_dir / 'data' / 'weights' / 'media_pipe_rfclassifier' / 'best_rf_model_M.pkl')

    joblib.dump(best_rf_model, model_dump_path)