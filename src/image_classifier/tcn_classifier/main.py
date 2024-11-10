import torch
from torch.utils.data import DataLoader
from dataset import PoseSequenceDataset
from model import TCNModel
from train import train_model
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
from collections import Counter

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Load datasets
    train_data = PoseSequenceDataset('../../data/dataset/train/images', '../../data/dataset/train/labels')
    val_data = PoseSequenceDataset('../../data/dataset/val/images', '../../data/dataset/val/labels')

    # for _, label in train_data:
    #     print(f"Label: {label}")

    # Compute class distribution for training set
    train_labels = [label for _, label in DataLoader(train_data, batch_size=len(train_data))]
    train_labels = torch.cat(train_labels).tolist()
    class_counts = Counter(train_labels)
    total_samples = sum(class_counts.values())
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}

    # Create a sampler for oversampling the minority class
    sample_weights = [1.0 / class_counts[int(label.item())] for _, label in train_data]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_data, batch_size=16, sampler=sampler)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

    model = TCNModel(input_size=99, num_classes=2)
    model.to(device)

    # Define weighted loss function using computed class weights
    weights = torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Define optimizer with lower learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device)

if __name__ == "__main__":
    main()
