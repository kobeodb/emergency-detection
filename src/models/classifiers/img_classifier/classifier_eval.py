import torch
import cv2
import numpy as np
import yaml
from pathlib import Path
from classifier import CNN

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

def load_model(checkpoint_path, config, device):
    """Load the model from a checkpoint."""
    model = CNN(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['classifier_state_dict'])
    model.eval()  # Set the model to evaluation mode
    return model

def predict(image_path, model, config, device):
    """Predict the label for a single image."""
    image_tensor = preprocess_image(image_path, config).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probability = output.item()
    return probability

if __name__ == "__main__":
    # Load configuration
    config_path = '../../../../config/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Specify the image path
    image_path = '../../data/pipeline_eval_data/test_frames/Screenshot 2024-12-02 at 19.15.07.png'

    # Set the device
    device = config['system']['device']

    # Load the trained model
    model = load_model('best_model.pth', config, device)

    # Predict the label
    probability = predict(image_path, model, config, device)

    print(f"Probability: {probability:.4f}")
