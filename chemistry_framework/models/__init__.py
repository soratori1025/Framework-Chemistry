"""Composable predictor models."""

from .residual import ResidualPropertyModel
from .builder import RolePredictor, build_role_predictor

__all__ = ["ResidualPropertyModel", "RolePredictor", "build_role_predictor"]
