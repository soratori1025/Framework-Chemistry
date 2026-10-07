"""Prior block contract."""
from __future__ import annotations
from abc import ABC, abstractmethod
from torch import Tensor, nn

class BaselineBlock(nn.Module, ABC):
    """Map a feature tensor to one or more baseline property values."""
    @abstractmethod
    def forward(self, features: Tensor) -> Tensor:
        """Return baseline values with batch dimensions preserved."""
