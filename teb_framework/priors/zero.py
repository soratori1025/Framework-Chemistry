"""Neutral baseline implementation."""

from __future__ import annotations

from torch import Tensor

from .base import BaselineBlock


class ZeroPrior(BaselineBlock):
    """A neutral baseline for tasks without a known additive contribution."""

    def __init__(self, output_dim: int = 1) -> None:
        super().__init__()
        if output_dim < 1:
            raise ValueError("output_dim must be positive")
        self.output_dim = output_dim

    def forward(self, features: Tensor) -> Tensor:
        return features.new_zeros((*features.shape[:-1], self.output_dim))
