import torch
import cv2
import numpy as np
import joblib
import yaml
from pathlib import Path
from torchvision import models
import torch.nn as nn


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        return self.resnet(x)

def preprocess_image(image_path, input_size):
    """Preprocesses the input image for feature extraction."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (input_size, input_size))
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))  # Convert to CxHxW format
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    return image_tensor

def test_classifier(image_path, config_path, model_path):
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device(config['system']['device'])

    # Initialize FeatureExtractor
    feature_extractor = FeatureExtractor().to(device)
    feature_extractor.eval()

    # Load the Random Forest Classifier
    rf_classifier = joblib.load(model_path)

    # Preprocess the input image
    input_size = config['model']['classifier']['input_size']
    image_tensor = preprocess_image(image_path, input_size).to(device)

    # Extract features
    with torch.no_grad():
        features = feature_extractor(image_tensor).cpu().numpy()

    # Make a prediction
    prediction = rf_classifier.predict(features)
    prediction_proba = rf_classifier.predict_proba(features)

    # Output results
    class_label = "Emergency" if prediction[0] == 0 else "No Emergency"
    confidence = prediction_proba[0][0] if prediction[0] == 0 else prediction_proba[0][1]

    print(f"Prediction: {class_label}")
    print(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    # Example usage
    image_path = "../../../data/pipeline_eval_data/test_frames/Screenshot 2024-12-08 at 18.59.34.png"  # Replace with your image path
    config_path = "../../../../config/config.yaml"
    model_path = "rf_classifier.pkl"

    test_classifier(image_path, config_path, model_path)
