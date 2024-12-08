import torch.nn as nn
import torch


def make_model(trial, input_size):
    initial_out_channels = 64
    num_layers = 4
    dropout = 0.2

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            in_channels = 3

            for i in range(num_layers):
                out_channels = initial_out_channels * (2 ** i)
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU())
                layers.append(nn.MaxPool2d(2, 2))
                in_channels = out_channels

            self.cnn = nn.Sequential(*layers)

            final_size = input_size // (2 ** num_layers)

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(final_size * final_size * in_channels, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            x = self.cnn(x)
            x = self.classifier(x)
            return x

    return CNN()
