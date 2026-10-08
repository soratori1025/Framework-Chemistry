"""Feature metadata and validated feature matrices."""
from __future__ import annotations
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
import numpy as np
FeatureCalculator = Callable[[Any], float]

@dataclass(frozen=True)
class FeatureSpec:
    """One named, independently replaceable molecular descriptor."""
    name: str
    calculator: FeatureCalculator
    domain: str
    applies_to: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    group: str | None = None
    description: str | None = None
    interpretation: str | None = None
    source: str | None = None
    definition: str | None = None
    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("feature name must not be empty")
        if not callable(self.calculator):
            raise TypeError(f"calculator for feature {self.name!r} must be callable")
        if not self.domain.strip():
            raise ValueError(f"domain for feature {self.name!r} must not be empty")
        for field_name in ("group", "description", "interpretation", "source", "definition"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"feature {field_name} must be a non-empty string when set")

@dataclass(frozen=True)
class FeatureMatrix:
    """Dense values paired with the exact feature order used to create them."""
    names: tuple[str, ...]
    values: np.ndarray
    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError(f"feature values must be 2-D, got shape {values.shape}")
        if values.shape[1] != len(self.names):
            raise ValueError(
                f"matrix has {values.shape[1]} columns but {len(self.names)} names"
            )
        if len(set(self.names)) != len(self.names):
            raise ValueError("feature names must be unique")
        if not np.isfinite(values).all():
            raise ValueError("feature matrix contains a non-finite value")
        object.__setattr__(self, "values", values)
