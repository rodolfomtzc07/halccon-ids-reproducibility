import math
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

        length1 = self._pooled_length(input_dim, 4)
        length2 = self._pooled_length(length1, 2)
        length3 = self._pooled_length(length2, 2)

        if min(length1, length2, length3) < 1:
            raise ValueError(
                f"input_dim={input_dim} is too small for HALCCON pooling path "
                f"(lengths: {length1}, {length2}, {length3})"
            )

        self.length1 = length1
        self.length2 = length2
        self.length3 = length3

        self.attention1 = HierarchicalAttention1D(64 * length1)
        self.attention2 = HierarchicalAttention1D(128 * length2)
        self.attention3 = HierarchicalAttention1D(256 * length3)

        self.fc = nn.Linear(256 * length3, num_classes)
        self.relu = nn.ReLU()

    @staticmethod
    def _pooled_length(length: int, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1) -> int:
        if stride is None:
            stride = kernel_size
        return math.floor((length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

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

        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x