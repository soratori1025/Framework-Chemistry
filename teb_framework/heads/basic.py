"""Generic scalar, vector, and identity output blocks."""

from torch import Tensor, nn


class IdentityOutputHead(nn.Module):
    """Pass a prediction through without transforming it."""

    def forward(self, values: Tensor) -> Tensor:
        return values


class ScalarOutputHead(nn.Module):
    """Validate and squeeze a one-value-per-sample prediction."""

    def forward(self, values: Tensor) -> Tensor:
        if values.shape[-1] != 1:
            raise ValueError(f"scalar head expects a final dimension of 1, got {values.shape}")
        return values.squeeze(-1)


class VectorOutputHead(nn.Module):
    """Validate the width of a fixed-size vector prediction."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        if output_dim < 1:
            raise ValueError("output_dim must be positive")
        self.output_dim = output_dim

    def forward(self, values: Tensor) -> Tensor:
        if values.shape[-1] != self.output_dim:
            raise ValueError(
                f"vector head expects {self.output_dim} values, got {values.shape[-1]}"
            )
        return values
