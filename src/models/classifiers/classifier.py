from comet_ml import Experiment
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm
import cv2
import mediapipe as mp
from src.models.metrics.metrics import calculate_metrics
from absl import logging
import os
import warnings


class FallImageDataset(Dataset):
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split

        self.data_dir = Path(config['classifier_data'][f'{split}_path'])

        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        images_dir = self.data_dir / 'images'
        labels_dir = self.data_dir / 'labels'

        for img_path in sorted(images_dir.glob('*.jpg')):
            label_path = labels_dir / f"{img_path.stem}.txt"

            if label_path.exists():
                with open(label_path, 'r') as f:
                    line = f.readline().strip().split()
                    label = int(line[0])

                    samples.append({
                        'image_path': img_path,
                        'label': label
                    })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = cv2.imread(str(sample['image_path']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (
            self.config['model']['classifier']['input_size'],
            self.config['model']['classifier']['input_size']))
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))

        label = torch.tensor(sample['label'], dtype=torch.float32)

        return {
            'image': torch.tensor(image, dtype=torch.float32),
            'label': label
        }


class CNN(nn.Module):
    def __init__(self, config):
        super().__init__()

        input_size = config['model']['classifier']['input_size']
        hidden_size = config['model']['classifier']['hidden_size']
        dropout = config['model']['classifier']['dropout']

        self.cnn = nn.Sequential(
            nn.Conv2d(3, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(hidden_size, hidden_size * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_size * 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(hidden_size * 2, hidden_size * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_size * 4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((input_size // 8) * (input_size // 8) * hidden_size * 4, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.classifier(x)
        return x


class ClassifierTrainer:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Comet ML Experiment
        # self.experiment = Experiment(
        #     api_key="Mo41H6DYUmGophP0c4VRLAo7Z",
        #     project_name="general",
        #     workspace="mxttywxtty"
        # )
        # self.experiment.log_parameters(self.config)

        self.device = self.config['system']['device']

        self.classifier = CNN(self.config).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.config['training_classifier']['learning_rate'],
            weight_decay=self.config['training_classifier']['weight_decay']
        )

        self.criterion = nn.BCELoss()

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )

        self.train_loader = DataLoader(
            FallImageDataset(self.config, 'train'),
            batch_size=self.config['training_classifier']['batch_size'],
            shuffle=True,
            num_workers=self.config['system']['num_workers']
        )

        self.valid_loader = DataLoader(
            FallImageDataset(self.config, 'valid'),
            batch_size=self.config['training_classifier']['batch_size'],
            shuffle=True,
            num_workers=self.config['system']['num_workers']
        )

    def train_epoch(self):
        self.classifier.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch in tqdm(self.train_loader, desc="Training"):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.classifier(images)
            loss = self.criterion(outputs, labels.unsqueeze(1))

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            all_preds.extend(outputs.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        all_preds_binary = (np.array(all_preds) > 0.5).astype(int)

        metrics = calculate_metrics(all_preds_binary, all_labels)
        metrics['loss'] = total_loss / len(self.train_loader)

        # self.experiment.log_metrics(metrics, step=None)

        return metrics

    def train(self):
        best_val_loss = float('inf')

        for epoch in range(self.config['training_classifier']['epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['training_classifier']['epochs']}")

            train_metrics = self.train_epoch()
            print('Training metrics:', train_metrics)

            self.scheduler.step(train_metrics['loss'])

            if train_metrics['loss'] < best_val_loss:
                best_val_loss = train_metrics['loss']
                self.save_checkpoint('best_model.pth')
                # self.experiment.log_model("best_model", "best_model.pth")

    def save_checkpoint(self, filename):
        checkpoint = {
            'classifier_state_dict': self.classifier.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, filename)


if __name__ == "__main__":
    trainer = ClassifierTrainer('../../../config/config.yaml')
    trainer.train()
