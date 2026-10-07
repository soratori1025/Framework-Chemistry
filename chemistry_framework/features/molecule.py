"""Molecule-to-feature-matrix transformation."""
from __future__ import annotations
from collections.abc import Iterable, Sequence
from typing import Any
import numpy as np
from .registry import FeatureRegistry
from .specs import FeatureMatrix, FeatureSpec

class MoleculeFeatureBlock:
    """Turn SMILES or RDKit molecules into an ordered numeric feature matrix."""
    def __init__(self, registry: FeatureRegistry, feature_names: Sequence[str]) -> None:
        if not feature_names:
            raise ValueError("at least one feature must be selected")
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("selected feature names must be unique")
        self.registry = registry
        self.feature_names = tuple(feature_names)
        self.feature_specs: tuple[FeatureSpec, ...] = tuple(
            registry.get(name) for name in self.feature_names
        )

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
        values = np.asarray(rows, dtype=np.float64).reshape(
            len(rows), len(self.feature_names)
        )
        return FeatureMatrix(self.feature_names, values)
