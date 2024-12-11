import os

import cv2
import joblib
import mediapipe as mp
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import optuna

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def extract_pose_keypoints(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)

    keypoints = np.zeros((33 * 3))

    if result.pose_landmarks:
        for i, landmark in enumerate(result.pose_landmarks.landmark):
            keypoints[i * 3] = landmark.x
            keypoints[i * 3 + 1] = landmark.y
            keypoints[i * 3 + 2] = landmark.z

    return np.array(keypoints).flatten()


def load_data(fall_dir, not_fall_dir):
    data = []
    labels = []
    for label, directory in enumerate([fall_dir, not_fall_dir]):
        for filename in os.listdir(directory):
            if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
                img_path = os.path.join(directory, filename)
                img = cv2.imread(img_path)
                kps = extract_pose_keypoints(img)

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

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    return accuracy  # Objective to maximize is accuracy


fall_dir = '../../data/classification_data/fall'
not_fall_dir = '../../data/classification_data/not-fall'

X, y = load_data(fall_dir, not_fall_dir)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

best_params = study.best_params

# Print the best parameters and accuracy
print(f"Best hyperparameters: {study.best_params}")
print(f"Best accuracy: {study.best_value}")

# Train the RandomForestClassifier with the best parameters
best_rf_model = RandomForestClassifier(**best_params, random_state=42)
best_rf_model.fit(X_train, y_train)

# Evaluate on the validation set
y_pred = best_rf_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Final model accuracy: {accuracy}")

joblib.dump(best_rf_model, '../weights/best_rf_model_M.pkl')
