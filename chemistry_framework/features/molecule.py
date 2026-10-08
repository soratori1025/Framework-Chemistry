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

    def transform(self, molecules_or_records: Iterable[Any], conditions: dict[str, tuple[float, ...]] | None = None) -> FeatureMatrix:
        try:
            from rdkit import Chem
        except ImportError as exc:
            raise ImportError("RDKit is required to parse SMILES") from exc
        from ..inputs.objects import ChemicalRecord, record_molecule

        rows: list[list[float]] = []
        # Convert inputs to a list to iterate alongside conditions
        items = list(molecules_or_records)
        
        for row_index, item in enumerate(items):
            if isinstance(item, ChemicalRecord):
                # We won't get a single mol right away because different features might need different sides
                pass
            else:
                mol = Chem.MolFromSmiles(item) if isinstance(item, str) else item
                if mol is None:
                    raise ValueError(f"invalid molecule at row {row_index}: {item!r}")
                if not hasattr(mol, "GetAtoms"):
                    raise TypeError(
                        f"row {row_index} must be a SMILES string, ChemicalRecord or RDKit molecule, "
                        f"got {type(item).__name__}"
                    )

            row: list[float] = []
            for spec in self.feature_specs:
                try:
                    if spec.domain == "condition":
                        if conditions is None:
                            raise RuntimeError("conditions were not provided to transform")
                        col = spec.metadata["column"]
                        if col not in conditions:
                            raise RuntimeError(f"condition column {col!r} not found in dataset")
                        val = conditions[col][row_index]
                        
                        # Apply transform if requested
                        transform_type = spec.metadata.get("transform", "identity")
                        if transform_type == "log":
                            val = float(np.log(val)) if val > 0 else 0.0
                        elif transform_type == "log10":
                            val = float(np.log10(val)) if val > 0 else 0.0
                        elif transform_type == "inverse":
                            val = float(1.0 / val) if val != 0 else 0.0
                        elif transform_type == "inverse_kilo":
                            val = float(1000.0 / val) if val != 0 else 0.0
                            
                        value = val
                    else:
                        if isinstance(item, ChemicalRecord):
                            side = spec.reaction_side
                            feature_mol = record_molecule(item, side)
                        else:
                            feature_mol = mol
                        value = float(spec.calculator(feature_mol))
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
