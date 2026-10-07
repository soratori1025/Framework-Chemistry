"""Molecular feature definitions and feature-matrix construction."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
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

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("feature name must not be empty")
        if not callable(self.calculator):
            raise TypeError(f"calculator for feature {self.name!r} must be callable")
        if not self.domain.strip():
            raise ValueError(f"domain for feature {self.name!r} must not be empty")


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

    def add_smarts(
        self,
        name: str,
        smarts: str,
        *,
        mode: str = "count",
        applies_to: Sequence[str] = (),
    ) -> None:
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


class MoleculeFeatureBlock:
    """Turn SMILES or RDKit molecules into an ordered numeric feature matrix."""

    def __init__(self, registry: FeatureRegistry, feature_names: Sequence[str]) -> None:
        if not feature_names:
            raise ValueError("at least one feature must be selected")
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("selected feature names must be unique")
        self.registry = registry
        self.feature_names = tuple(feature_names)
        self.feature_specs = tuple(registry.get(name) for name in self.feature_names)

    def transform(self, molecules: Iterable[Any]) -> FeatureMatrix:
        try:
            from rdkit import Chem
        except ImportError as exc:
            raise ImportError("RDKit is required to parse SMILES") from exc

        rows: list[list[float]] = []
        for row_index, item in enumerate(molecules):
            mol = Chem.MolFromSmiles(item) if isinstance(item, str) else item
            if mol is None:
                raise ValueError(f"invalid molecule at row {row_index}: {item!r}")
            if not hasattr(mol, "GetAtoms"):
                raise TypeError(
                    f"row {row_index} must be a SMILES string or RDKit molecule, "
                    f"got {type(item).__name__}"
                )
            row: list[float] = []
            for spec in self.feature_specs:
                try:
                    value = float(spec.calculator(mol))
                except Exception as exc:
                    raise RuntimeError(
                        f"feature {spec.name!r} failed for molecule at row {row_index}"
                    ) from exc
                if not np.isfinite(value):
                    raise ValueError(
                        f"feature {spec.name!r} returned a non-finite value at row {row_index}"
                    )
                row.append(value)
            rows.append(row)
        return FeatureMatrix(self.feature_names, np.asarray(rows, dtype=np.float64).reshape(
            len(rows), len(self.feature_names)
        ))


def molecular_feature_registry(
    elements: Sequence[str] = ("C", "H", "N", "O", "S", "F", "Cl", "Br", "P", "I"),
) -> FeatureRegistry:
    """Create a small extensible registry of general-purpose molecular features."""
    try:
        from rdkit.Chem import rdMolDescriptors
    except ImportError as exc:
        raise ImportError("RDKit is required for built-in molecular features") from exc

    registry = FeatureRegistry()

    def atom_count(symbol: str) -> FeatureCalculator:
        return lambda mol: float(
            sum(atom.GetSymbol() == symbol for atom in mol.GetAtoms())
            if symbol != "H"
            else sum(atom.GetTotalNumHs(includeNeighbors=True) for atom in mol.GetAtoms())
        )

    for symbol in elements:
        registry.register(
            FeatureSpec(
                name=f"n_{symbol}",
                calculator=atom_count(symbol),
                domain="composition",
                metadata={"element": symbol},
            )
        )
    registry.register(
        FeatureSpec(
            "n_heavy_atoms",
            lambda mol: float(mol.GetNumHeavyAtoms()),
            domain="composition",
        )
    )
    for bond_name, bond_type in (
        ("single", "SINGLE"),
        ("double", "DOUBLE"),
        ("triple", "TRIPLE"),
        ("aromatic", "AROMATIC"),
    ):
        registry.register(
            FeatureSpec(
                f"n_{bond_name}_bonds",
                lambda mol, kind=bond_type: float(
                    sum(str(bond.GetBondType()) == kind for bond in mol.GetBonds())
                ),
                domain="topology",
            )
        )
    registry.register(
        FeatureSpec(
            "n_rings",
            lambda mol: float(rdMolDescriptors.CalcNumRings(mol)),
            domain="topology",
        )
    )
    registry.register(
        FeatureSpec(
            "n_rotatable_bonds",
            lambda mol: float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
            domain="topology",
        )
    )
    return registry
