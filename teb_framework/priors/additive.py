"""Frozen linear-additive baseline implementation."""
from __future__ import annotations
from collections.abc import Sequence
import torch
from torch import Tensor
from .base import BaselineBlock

class LinearAdditivePrior(BaselineBlock):
    """Frozen additive prior: x @ coefficients + intercept."""

    coefficients: Tensor
    intercept: Tensor

    def __init__(self, coefficients: Sequence[float] | Sequence[Sequence[float]], intercept: float | Sequence[float] = 0.0, feature_names: Sequence[str] | None = None,) -> None:
        super().__init__()
        weights = torch.as_tensor(coefficients, dtype=torch.float64)
        if weights.ndim == 1:
            weights = weights.unsqueeze(-1)
        if weights.ndim != 2 or weights.shape[0] == 0 or weights.shape[1] == 0:
            raise ValueError("coefficients must be a non-empty 1-D or 2-D array")
        bias = torch.as_tensor(intercept, dtype=torch.float64).reshape(-1)
        if bias.numel() == 1 and weights.shape[1] > 1:
            bias = bias.expand(weights.shape[1]).clone()
        if bias.numel() != weights.shape[1]:
            raise ValueError(
                f"intercept has {bias.numel()} values; expected {weights.shape[1]}"
            )
        if not torch.isfinite(weights).all() or not torch.isfinite(bias).all():
            raise ValueError("prior coefficients and intercept must be finite")
        if feature_names is not None and len(feature_names) != weights.shape[0]:
            raise ValueError(
                f"received {len(feature_names)} feature names for {weights.shape[0]} coefficients"
            )
        self.register_buffer("coefficients", weights)
        self.register_buffer("intercept", bias)
        self.feature_names = tuple(feature_names) if feature_names is not None else None

    @property
    def output_dim(self) -> int:
        return self.coefficients.shape[1]

    def forward(self, features: Tensor) -> Tensor:
        if features.shape[-1] != self.coefficients.shape[0]:
            raise ValueError(
                f"prior expects {self.coefficients.shape[0]} features, got {features.shape[-1]}"
            )
        weights = self.coefficients.to(device=features.device, dtype=features.dtype)
        bias = self.intercept.to(device=features.device, dtype=features.dtype)
        return features @ weights + bias
