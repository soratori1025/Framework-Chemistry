"""Property prediction output blocks."""

from .basic import IdentityOutputHead, ScalarOutputHead, VectorOutputHead
from .nasa7 import NASA7OutputBlock

__all__ = [
    "IdentityOutputHead",
    "NASA7OutputBlock",
    "ScalarOutputHead",
    "VectorOutputHead",
]
