import torch
import torch.nn as nn

from src.models.attention import HierarchicalAttention1D


class HALCCONMulticlass(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 13):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding="same")
        self.pool1 = nn.MaxPool1d(kernel_size=4)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding="same")
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding="same")
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.bn3 = nn.BatchNorm1d(256)

        length1 = input_dim // 4
        length2 = input_dim // 8
        length3 = input_dim // 16

        self.attention1 = HierarchicalAttention1D(64 * length1)
        self.attention2 = HierarchicalAttention1D(128 * length2)
        self.attention3 = HierarchicalAttention1D(256 * length3)

        self.fc = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.attention1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.attention2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.attention3(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x