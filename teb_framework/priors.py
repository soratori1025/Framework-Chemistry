"""Composable baseline and physics-informed prior blocks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn


class BaselineBlock(nn.Module, ABC):
    """Map a feature tensor to one or more baseline property values."""

    @abstractmethod
    def forward(self, features: Tensor) -> Tensor:
        """Return baseline values with batch dimensions preserved."""


class ZeroPrior(BaselineBlock):
    """A neutral baseline for tasks without a known additive contribution."""

    def __init__(self, output_dim: int = 1) -> None:
        super().__init__()
        if output_dim < 1:
            raise ValueError("output_dim must be positive")
        self.output_dim = output_dim

    def forward(self, features: Tensor) -> Tensor:
        return features.new_zeros((*features.shape[:-1], self.output_dim))


class LinearAdditivePrior(BaselineBlock):
    """Frozen additive prior: x @ coefficients + intercept."""

    def __init__(
        self,
        coefficients: Sequence[float] | Sequence[Sequence[float]],
        intercept: float | Sequence[float] = 0.0,
        feature_names: Sequence[str] | None = None,
    ) -> None:
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


class CallablePhysicsPrior(BaselineBlock):
    """Adapt a user-supplied differentiable physical equation as a prior block."""

    def __init__(self, function: Callable[[Tensor], Tensor], output_dim: int = 1) -> None:
        super().__init__()
        if not callable(function):
            raise TypeError("function must be callable")
        if output_dim < 1:
            raise ValueError("output_dim must be positive")
        self.function = function
        self.output_dim = output_dim

    def forward(self, features: Tensor) -> Tensor:
        result = self.function(features)
        if not isinstance(result, Tensor):
            raise TypeError("physics prior function must return a torch.Tensor")
        if result.shape != (*features.shape[:-1], self.output_dim):
            raise ValueError(
                f"physics prior returned shape {tuple(result.shape)}; expected "
                f"{(*features.shape[:-1], self.output_dim)}"
            )
        return result


class CompositePrior(BaselineBlock):
    """Sum compatible prior blocks to express additive/hybrid baselines."""

    def __init__(self, priors: Sequence[BaselineBlock]) -> None:
        super().__init__()
        if not priors:
            raise ValueError("CompositePrior requires at least one prior")
        output_dims = {getattr(prior, "output_dim", None) for prior in priors}
        if len(output_dims) != 1 or None in output_dims:
            raise ValueError("all composite priors must declare the same output_dim")
        self.priors = nn.ModuleList(priors)
        self.output_dim = output_dims.pop()

    def forward(self, features: Tensor) -> Tensor:
        result = self.priors[0](features)
        for prior in self.priors[1:]:
            result = result + prior(features)
        return result
