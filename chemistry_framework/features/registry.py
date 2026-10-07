"""Registry for chemistry-specific and user-defined feature calculators."""
from __future__ import annotations
from collections.abc import Sequence
from typing import Any
from .specs import FeatureSpec

class FeatureRegistry:
    """Registry for composable molecular, physical, and user-defined features."""
    def __init__(self) -> None:
        self._features: dict[str, FeatureSpec] = {}

    def register(self, feature: FeatureSpec, *, replace: bool = False) -> None:
        if feature.name in self._features and not replace:
            raise ValueError(f"feature {feature.name!r} is already registered")
        self._features[feature.name] = feature

    def get(self, name: str) -> FeatureSpec:
        try:
            return self._features[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._features)) or "<empty>"
            raise KeyError(f"unknown feature {name!r}; registered features: {available}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(self._features)

    def add_smarts(self, name: str, smarts: str, *, mode: str = "count", applies_to: Sequence[str] = (),) -> None:
        """Register a SMARTS fragment count or presence indicator."""
        if mode not in {"count", "presence"}:
            raise ValueError("SMARTS mode must be 'count' or 'presence'")
        try:
            from rdkit import Chem
        except ImportError as exc:
            raise ImportError("RDKit is required to register SMARTS features") from exc
        query = Chem.MolFromSmarts(smarts)
        if query is None:
            raise ValueError(f"invalid SMARTS pattern for feature {name!r}: {smarts!r}")

        def calculate(mol: Any) -> float:
            matches = mol.GetSubstructMatches(query, uniquify=True)
            return float(bool(matches)) if mode == "presence" else float(len(matches))

        self.register(
            FeatureSpec(
                name=name,
                calculator=calculate,
                domain="functional_group",
                applies_to=tuple(applies_to),
                metadata={"smarts": smarts, "mode": mode},
            )
        )
