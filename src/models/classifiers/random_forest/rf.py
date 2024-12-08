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
from torchvision import models
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib

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




class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        return self.resnet(x)


class RFClassifierTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config['system']['device']
        self.feature_extractor = FeatureExtractor().to(self.device)
        self.rf_classifier = RandomForestClassifier(
            n_estimators=config['training_rf']['n_estimators'],
            max_depth=config['training_rf']['max_depth']
        )

    def extract_features(self, dataloader):
        self.feature_extractor.eval()
        all_features = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Extracting Features'):
                images = batch['image'].to(self.device)
                labels = batch['label'].cpu().numpy()

                features = self.feature_extractor(images).cpu().numpy()
                all_features.append(features)
                all_labels.append(labels)

        return np.vstack(all_features), np.hstack(all_labels)


    def train(self, train_loader, valid_loader):
        train_features, train_labels = self.extract_features(train_loader)
        valid_features, valid_labels = self.extract_features(valid_loader)

        self.rf_classifier.fit(train_features, train_labels)

        joblib.dump(self.rf_classifier, 'rf_classifier.pkl')

        predictions = self.rf_classifier.predict(valid_features)
        metrics = {
            'accuracy': accuracy_score(valid_labels, predictions),
            'precision': precision_score(valid_labels, predictions),
            'recall': recall_score(valid_labels, predictions)
        }
        print("Validation Metrics:", metrics)
        return metrics



if __name__ == "__main__":
    config_path = '../../../../config/config.yaml'

    with open(config_path) as f:
        config = yaml.safe_load(f)

    trainer = RFClassifierTrainer(config)
    train_loader = DataLoader(
        FallImageDataset(config, 'train'),
        batch_size=config['training_classifier']['batch_size'],
        shuffle=True,
    )
    valid_loader = DataLoader(
        FallImageDataset(config, 'valid'),
        batch_size=config['training_classifier']['batch_size'],
        shuffle=False,
    )
    trainer.train(train_loader, valid_loader)