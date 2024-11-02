import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the datasets
train_df = pd.read_csv('train_keypoints.csv')
val_df = pd.read_csv('val_keypoints.csv')

# Separate features and labels
X_train = train_df.drop(columns=['label'])
y_train = train_df['label']
X_val = val_df.drop(columns=['label'])
y_val = val_df['label']

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)

# Ensure there are multiple classes
num_classes = len(np.unique(y_train_encoded))
if num_classes < 2:
    raise ValueError("Training data must contain at least two classes.")

# Feature Engineering: Normalize by torso, calculate angles, and add temporal features
def feature_engineering(df):
    # Assume the torso keypoint is at index 11 for normalization
    torso_x = df.iloc[:, 33]  # Assuming torso x-coordinate is the 34th feature
    torso_y = df.iloc[:, 34]  # Assuming torso y-coordinate is the 35th feature

    # Normalize keypoints by torso position
    for i in range(0, len(df.columns) - 1, 3):  # Iterating over each x, y, z triplet
        df.iloc[:, i] -= torso_x
        df.iloc[:, i + 1] -= torso_y

    # Calculate angles between joints as additional features (example: shoulder-elbow-wrist)
    df['angle_shoulder_elbow'] = np.degrees(np.arctan2(df.iloc[:, 1] - df.iloc[:, 4], df.iloc[:, 0] - df.iloc[:, 3]))
    df['angle_elbow_wrist'] = np.degrees(np.arctan2(df.iloc[:, 4] - df.iloc[:, 7], df.iloc[:, 3] - df.iloc[:, 6]))
    df['angle_hip_knee_ankle'] = np.degrees(np.arctan2(df.iloc[:, 13] - df.iloc[:, 16], df.iloc[:, 12] - df.iloc[:, 15]))
    df['angle_shoulder_hip'] = np.degrees(np.arctan2(df.iloc[:, 1] - df.iloc[:, 13], df.iloc[:, 0] - df.iloc[:, 12]))

    # Placeholder for temporal features (velocity and acceleration)
    # You could calculate these using consecutive frames in your dataset
    df['velocity'] = 0  # Example placeholder, replace with actual calculation
    df['acceleration'] = 0  # Example placeholder, replace with actual calculation

    return df

# Apply feature engineering to training and validation sets
X_train = feature_engineering(X_train.copy())
X_val = feature_engineering(X_val.copy())

# Initialize XGBoost with a randomized search for hyperparameter tuning
param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [3, 6, 10, 15],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'scale_pos_weight': [1, 5, 10]  # Added for class weight adjustment
}

# Modify num_class parameter and eval_metric for compatibility
random_search = RandomizedSearchCV(XGBClassifier(use_label_encoder=False, objective='multi:softmax', num_class=num_classes, eval_metric='mlogloss', early_stopping_rounds=10), param_dist, n_iter=15, cv=3, scoring='accuracy', n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train_encoded, eval_set=[(X_val, y_val_encoded)], verbose=False)

print("Best parameters found:", random_search.best_params_)
best_model = random_search.best_estimator_

# Evaluate the best model on the validation set
y_pred_encoded = best_model.predict(X_val)
y_pred = label_encoder.inverse_transform(y_pred_encoded)
y_val_original = label_encoder.inverse_transform(y_val_encoded)

accuracy = accuracy_score(y_val_original, y_pred)
print("Validation Accuracy after tuning:", accuracy)
print("Classification Report after tuning:\n", classification_report(y_val_original, y_pred))

# Save the tuned model and label encoder
joblib.dump(best_model, '../src/models/fall_detection_model_xgb.pkl')
joblib.dump(label_encoder, '../src/models/label_encoder.pkl')
print("Tuned XGBoost model and label encoder saved.")
