import os
from typing import Tuple

import cv2
import joblib
import mediapipe as mp
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import optuna

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)


def extract_kps(image):
    result = pose.process(image)
    kps = np.zeros(33 * 3)

    if result.pose_landmarks:
        for i, landmark in enumerate(result.pose_landmarks.landmark):  # ignore
            kps[i * 3:i * 3 + 3] = [landmark.x, landmark.y, landmark.z]

    return kps


def augment_image(image):
    if np.random.rand() > 0.5:
        image = cv2.flip(image, 1)

    angle = np.random.uniform(-30, 30)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, rotation_matrix, (w, h))

    factor = np.random.uniform(0.7, 1.3)
    image = cv2.convertScaleAbs(image, alpha=factor, beta=0)

    return image


def load_data(fd: str, nfd: str, augment=False) -> Tuple[np.array, np.array]:
    data = []
    labels = []
    for label, _dir in enumerate([fd, nfd]):
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
    fall_dir = '../../data/classification_data/fall'
    not_fall_dir = '../../data/classification_data/not-fall'

    X, y = load_data(fall_dir, not_fall_dir, augment=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    best_params = study.best_params

    print(f"Best hyperparameters: {study.best_params}")
    print(f"Best accuracy: {study.best_value}")

    best_rf_model = RandomForestClassifier(**best_params, random_state=42)
    best_rf_model.fit(X_train, y_train)

    y_pred = best_rf_model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Final model accuracy: {accuracy}")

    joblib.dump(best_rf_model, '../weights/best_rf_model_M.pkl')
