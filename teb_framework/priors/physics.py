"""Adapter for user-provided differentiable physical equations."""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor

from .base import BaselineBlock


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
