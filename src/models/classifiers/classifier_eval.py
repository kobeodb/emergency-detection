import torch
import cv2
import numpy as np
import yaml
from pathlib import Path
from classifier import CNN
from src.models.classifiers.notebooks.cnn2d_utils import make_model


def preprocess_image(image_path, config):
    """Preprocess an image for the model."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (
        config['model']['classifier']['input_size'],
        config['model']['classifier']['input_size']
    ))
    image = image / 255.0  # Normalize to [0, 1]
    image = np.transpose(image, (2, 0, 1))  # Convert HWC to CHW
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    return image_tensor

def load_model(model_path, config, device):
    """Load the model from a checkpoint."""

    input_size = config['model']['classifier']['input_size']
    loaded_model = make_model(trial=None, input_size=input_size).to(config['system']['device'])
    checkpoint = torch.load(model_path, map_location=config['system']['device'])
    loaded_model.load_state_dict(checkpoint['classifier_state_dict'])
    loaded_model.eval()

    return loaded_model

def predict(image_path, model, config, device):
    """Predict the label for a single image."""
    image_tensor = preprocess_image(image_path, config).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probability = output.item()
    return probability

if __name__ == "__main__":
    # Load configuration
    config_path = '../../../config/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Specify the image path
    # image_path = '../../data/pipeline_eval_data/test_frames/Screenshot 2024-12-02 at 19.11.24.png'
    image_path = '../../data/classifier_data/test/images/final1500107_jpg.rf.f39398cceb35b9c67b3bdaff083edca3.jpg'

    # Set the device
    device = config['system']['device']

    # Load the trained model
    model = load_model('notebooks/best_model_optuna.pth', config, device)

    # Predict the label
    probability = predict(image_path, model, config, device)

    print(f"Probability: {probability:.4f}")
