"""MLP implementation of the encoder and residual blocks."""
from __future__ import annotations
import torch
from torch import Tensor, nn

def _validate_mlp(input_dim: int, output_dim: int, hidden_dim: int, depth: int, dropout: float) -> None:
    if min(input_dim, output_dim, hidden_dim, depth) < 1:
        raise ValueError("MLP dimensions and depth must be positive")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")

class MLPEncoder(nn.Module):
    """Descriptor encoder; graph encoders can be supplied as any nn.Module."""
    def __init__(self, input_dim: int, representation_dim: int, hidden_dim: int = 128, depth: int = 2, dropout: float = 0.1,) -> None:
        super().__init__()
        _validate_mlp(input_dim, representation_dim, hidden_dim, depth, dropout)
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(depth - 1):
            layers.extend((nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, representation_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)

class MLPResidual(nn.Module):
    """Learn the residual correction on top of a baseline."""
    def __init__(self, representation_dim: int, output_dim: int = 1, hidden_dim: int = 128, depth: int = 2, dropout: float = 0.1,) -> None:
        super().__init__()
        _validate_mlp(representation_dim, output_dim, hidden_dim, depth, dropout)
        layers: list[nn.Module] = []
        in_dim = representation_dim
        for _ in range(depth - 1):
            layers.extend((nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, representation: Tensor) -> Tensor:
        return self.network(representation)
