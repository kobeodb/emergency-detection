import pandas as pd
import numpy as np
import xgboost
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import joblib

# Load datasets
train_df = pd.read_csv('data/points/train_keypoints.csv')
val_df = pd.read_csv('data/points/val_keypoints.csv')

# Check for NaN values in datasets
print("Training data NaN count:\n", train_df.isna().sum().sum())
print("Validation data NaN count:\n", val_df.isna().sum().sum())

# Separate features and labels
X_train, y_train = train_df.drop(columns=['label']), train_df['label']
X_val, y_val = val_df.drop(columns=['label']), val_df['label']

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)


def enhanced_feature_engineering(df):
    """
    Perform feature engineering on the dataset to add meaningful features
    by normalizing keypoints, calculating angles, and deriving distances.

    Parameters:
    df (DataFrame): Input DataFrame containing keypoint data.

    Returns:
    DataFrame: Transformed DataFrame with engineered features.
    """
    df = df.copy()

    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Normalize by torso keypoints
    torso_x, torso_y = df.iloc[:, 33], df.iloc[:, 34]
    for i in range(0, len(df.columns) - 1, 3):
        df.iloc[:, i] = df.iloc[:, i].sub(torso_x, fill_value=0)
        df.iloc[:, i + 1] = df.iloc[:, i + 1].sub(torso_y, fill_value=0)

    # Define safe calculation functions for angles and distances
    def safe_angle(x1, y1, x2, y2):
        return np.degrees(np.arctan2(y2 - y1, x2 - x1)) if not (np.isnan([x1, y1, x2, y2]).any()) else np.nan

    def safe_distance(x1, y1, x2, y2):
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) if not (np.isnan([x1, y1, x2, y2]).any()) else np.nan

    # Compute angles and distances
    df['left_shoulder_angle'] = df.apply(lambda row: safe_angle(row.iloc[0], row.iloc[1], row.iloc[3], row.iloc[4]), axis=1)
    df['right_shoulder_angle'] = df.apply(lambda row: safe_angle(row.iloc[6], row.iloc[7], row.iloc[9], row.iloc[10]), axis=1)
    df['shoulder_width'] = df.apply(lambda row: safe_distance(row.iloc[0], row.iloc[1], row.iloc[6], row.iloc[7]), axis=1)
    df['hip_width'] = df.apply(lambda row: safe_distance(row.iloc[12], row.iloc[13], row.iloc[18], row.iloc[19]), axis=1)

    # Calculate ratios
    df['shoulder_hip_ratio'] = df['shoulder_width'].div(df['hip_width']).replace([np.inf, -np.inf], np.nan)

    # Debugging information
    print("\nFeature engineering stats:")
    print("NaN values after engineering:", df.isna().sum().sum())
    print("Infinite values:", np.isinf(df.values).sum())

    return df


# Apply feature engineering
print("\nApplying feature engineering...")
X_train = enhanced_feature_engineering(X_train)
X_val = enhanced_feature_engineering(X_val)

# Preprocessing pipeline without SMOTE
preprocessing_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Apply preprocessing
X_train_preprocessed = preprocessing_pipeline.fit_transform(X_train)
X_val_preprocessed = preprocessing_pipeline.transform(X_val)

# Apply SMOTE to balance classes
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_preprocessed, y_train_encoded)

# Configure XGBoost classifier
xgb_classifier = XGBClassifier(
    use_label_encoder=False,
    objective='multi:softmax',
    num_class=len(np.unique(y_train_encoded)),
    eval_metric='mlogloss',
    random_state=42
)

# Define parameter grid for hyperparameter tuning
param_dist = {
    'n_estimators': [300, 400, 500],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
}

# Split training data for early stopping
X_train_final, X_early_stop, y_train_final, y_early_stop = train_test_split(
    X_train_balanced, y_train_balanced, test_size=0.2, random_state=42
)

# Hyperparameter tuning using RandomizedSearchCV
print("\nStarting model training...")
random_search = RandomizedSearchCV(
    xgb_classifier, param_distributions=param_dist, n_iter=15, cv=5,
    scoring='accuracy', n_jobs=-1, random_state=42, verbose=2
)
random_search.fit(X_train_final, y_train_final)

print("\nBest parameters found:", random_search.best_params_)
best_model = random_search.best_estimator_

# Final model with best parameters and early stopping
final_xgb = XGBClassifier(
    **random_search.best_params_,
    use_label_encoder=False,
    objective='multi:softmax',
    num_class=len(np.unique(y_train_encoded)),
    eval_metric='mlogloss',
    random_state=42,
    callbacks=[xgboost.callback.EarlyStopping(rounds=20, save_best=True)]  # type: ignore
)
final_xgb.fit(X_train_final, y_train_final, eval_set=[(X_early_stop, y_early_stop)], verbose=False)

# Evaluate the model
y_pred = final_xgb.predict(X_val_preprocessed)
y_pred_labels = label_encoder.inverse_transform(y_pred)
y_val_original = label_encoder.inverse_transform(y_val_encoded)

accuracy = accuracy_score(y_val_original, y_pred_labels)
print("\nValidation Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_val_original, y_pred_labels))

# Save the final model and encoder
final_model = Pipeline([
    ('preprocessor', preprocessing_pipeline),
    ('classifier', final_xgb)
])
joblib.dump(final_model, 'data/models/improved_fall_detection_model_xgb.pkl')
joblib.dump(label_encoder, 'data/models/improved_label_encoder.pkl')
print("\nImproved model and label encoder saved.")
