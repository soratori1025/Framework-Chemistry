"""Composition model for prior-plus-residual property prediction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from ..heads import IdentityOutputHead
from ..priors import BaselineBlock


class ResidualPropertyModel(nn.Module):
    """Compose an encoder, frozen baseline, learned residual, and output head.

    ``encoder`` may be an MLP or any compatible graph/3-D encoder. For graph
    encoders, pass a tensor separately as ``baseline_features``. The residual
    is added to the baseline before the optional output transformation. Output
    blocks may return a tensor or a mapping of named tensor channels.
    """

    def __init__(
        self,
        encoder: nn.Module,
        prior: BaselineBlock,
        residual: nn.Module,
        output_head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.prior = prior
        self.residual = residual
        self.output_head = output_head or IdentityOutputHead()

    def forward(
        self,
        encoder_input: Any,
        baseline_features: Tensor | None = None,
    ) -> dict[str, Tensor]:
        representation = self.encoder(encoder_input)
        if not isinstance(representation, Tensor):
            raise TypeError("encoder must return a torch.Tensor representation")
        if baseline_features is None:
            if not isinstance(encoder_input, Tensor):
                raise ValueError(
                    "baseline_features is required when encoder_input is not a tensor"
                )
            baseline_features = encoder_input
        baseline = self.prior(baseline_features)
        residual = self.residual(representation)
        if baseline.shape != residual.shape:
            raise ValueError(
                f"prior output shape {tuple(baseline.shape)} does not match residual "
                f"shape {tuple(residual.shape)}"
            )
        prediction = self.output_head(baseline + residual)
        if isinstance(prediction, Tensor):
            return {"baseline": baseline, "residual": residual, "prediction": prediction}
        if isinstance(prediction, Mapping) and all(
            isinstance(value, Tensor) for value in prediction.values()
        ):
            return {"baseline": baseline, "residual": residual, **prediction}
        raise TypeError("output_head must return a tensor or a mapping of tensors")
