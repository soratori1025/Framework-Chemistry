"""Composition of compatible prior blocks."""
from __future__ import annotations
from collections.abc import Sequence
from torch import Tensor, nn
from .base import BaselineBlock

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
