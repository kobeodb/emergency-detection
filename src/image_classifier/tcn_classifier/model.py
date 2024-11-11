import torch.nn as nn

class TCNBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, dilation=1):
        super(TCNBlock, self).__init__()
        self.conv = nn.Conv1d(
            input_size,
            output_size,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation // 2
        )
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(output_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return self.batchnorm(x)

class TCNModel(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=64, kernel_size=3):
        super(TCNModel, self).__init__()
        self.tcn1 = TCNBlock(input_size, hidden_size, kernel_size)
        self.tcn2 = TCNBlock(hidden_size, hidden_size, kernel_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()  # Add ReLU activation function
        self.dropout = nn.Dropout(0.5)  # Add dropout layer

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(2)
        elif x.shape[1] == 1:
            x = x.permute(0, 2, 1)
        x = self.tcn1(x)
        x = self.relu(x)
        x = self.tcn2(x)
        x = self.dropout(x)
        x = self.fc(x.mean(dim=2))
        return x