import argparse
import math
import os
import cv2
import threading
from typing import Callable

from jax.example_libraries.optimizers import optimizer, momentum
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from ultralytics import YOLO
from comet_ml import start
from comet_ml.integration.pytorch import log_model
from dotenv import load_dotenv
import torch

load_dotenv()

# API_KEY = os.getenv('API_KEY')
#
# experiment = start(
#   api_key=API_KEY,
#   project_name="general",
#   workspace="kobeodb"
# )

# Constants
FONT_SCALE = 0.5
FONT_COLOR = (0, 0, 255)
FONT_THICKNESS = 1
# CLASSES = ['Fall Detected', 'Not Fall', 'Sitting', 'Walking']
CLASSES = ['Fall Detected', 'Not Fall']
TRAIN_EPOCHS = 20
IMG_SIZE = 640

#learning rate
INITIAL_LR = 0.01
MIN_LR = 0.0001


def initialize_model(weights: str) -> YOLO:
    """Initialize YOLO model with specified weights."""
    return YOLO(weights)

def train_model_with_scheduler(weights: str, data_yaml: str, epochs: int = TRAIN_EPOCHS, img_size: int = IMG_SIZE):
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    model = initialize_model(weights)
    model.to(device)

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        optimizer='SGD',
        lr0=INITIAL_LR,
        lrf= MIN_LR / INITIAL_LR,
        momentum=0.9,
        weight_decay=5e-4
    )

#     # log_model(experiment, model=model, model_name="yoloNewDataSet")



if __name__ == "__main__":

    data_yaml_path = os.path.abspath("../data/data.yaml")
    weights_path = os.path.abspath("../data/weights/yolo11n.pt")
    # weights_path = os.path.abspath("../../runs/detect/train5/weights/best.pt")

    # Train the model
    train_model_with_scheduler(weights=weights_path, data_yaml=data_yaml_path)
