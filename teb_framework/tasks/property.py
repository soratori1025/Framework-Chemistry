"""Property-task metadata kept separate from chemistry and model layers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PropertyTask:
    """Describe a target and the feature routes used by its blocks."""

    name: str
    target: str
    unit: str
    representation_features: tuple[str, ...] = ()
    baseline_features: tuple[str, ...] = ()
    output_kind: str = "scalar"
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for label, value in (("name", self.name), ("target", self.target), ("unit", self.unit)):
            if not value.strip():
                raise ValueError(f"property task {label} must not be empty")
        if self.output_kind not in {"scalar", "vector", "temperature_dependent", "custom"}:
            raise ValueError(f"unsupported output_kind {self.output_kind!r}")
        if len(set(self.representation_features)) != len(self.representation_features):
            raise ValueError("representation_features must be unique")
        if len(set(self.baseline_features)) != len(self.baseline_features):
            raise ValueError("baseline_features must be unique")
