import torch
from ultralytics import YOLO
import yaml


class FallDetectorTrainer:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device(self.config['system']['device'])
        self.model = YOLO(self.config['model']['detector']['weights_path'])

    def train(self):
        train_args = {
            'data': 'data.yaml',
            'epochs': self.config['training_detector']['epochs'],
            'imgsz': self.config['fall_detection_data']['img_size'],
            'device': self.config['system']['device'],
            'project': 'runs/train',
            'name': 'Fall Detector',
            'optimizer': 'Adam'
        }

        results = self.model.train(**train_args)

    def validate(self):
        val_args = {
            'data': self.config['model']['detector']['data_yaml_path'],
            'imgsz': self.config['fall_detection_data']['img_size'],
            'device': self.config['system']['device'],
        }

        metrics = self.model.val(**val_args)
        return metrics


if __name__ == "__main__":
    trainer = FallDetectorTrainer('../../../config/config.yaml')
    trainer.train()
