# src/models/attention.py Esto está alineado con la lógica de atención del notebook: aplana C × L, aplica una capa lineal, usa tanh, reacomoda y multiplica elemento a elemento.
import torch
import torch.nn as nn


class HierarchicalAttention1D(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.attention = nn.Linear(in_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, length = x.size()

        x_flat = torch.flatten(x, start_dim=1)
        attn_weights = torch.tanh(self.attention(x_flat))
        attn_weights = attn_weights.reshape(batch_size, channels, length)

        return x * attn_weights