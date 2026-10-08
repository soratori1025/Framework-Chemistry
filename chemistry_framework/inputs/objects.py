"""Format-independent internal chemical records.

Every input adapter normalizes its source format into one of these objects so
that downstream layers (features, graph representations, splits) never depend
on how the user stored the chemistry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class MoleculeObject:
    """One molecule with optional 3-D coordinates and metadata.

    ``mol`` is an RDKit molecule. ``coordinates`` (shape ``[n_atoms, 3]``) is
    populated when the source format carried a geometry (SDF/MolBlock/XYZ) and
    mirrors the first RDKit conformer.
    """

    mol: Any
    smiles: str
    source_format: str
    coordinates: np.ndarray | None = None
    identifier: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def has_geometry(self) -> bool:
        return self.coordinates is not None

    @property
    def num_atoms(self) -> int:
        return int(self.mol.GetNumAtoms())

    @property
    def num_heavy_atoms(self) -> int:
        return int(self.mol.GetNumHeavyAtoms())


@dataclass(frozen=True)
class ReactionObject:
    """A reaction with reactant, product, and agent molecules.

    Row-level reaction conditions (temperature, pressure, solvent descriptors,
    ...) are stored on the dataset as aligned numeric columns rather than here,
    so that they can be used as ordinary features in any model role.
    """

    reactants: tuple[MoleculeObject, ...]
    products: tuple[MoleculeObject, ...]
    smiles: str
    source_format: str
    agents: tuple[MoleculeObject, ...] = ()
    identifier: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def combined(self, side: str) -> Any:
        """Return one RDKit molecule containing every molecule of ``side``."""
        from rdkit import Chem

        if side == "reactants":
            members = self.reactants
        elif side == "products":
            members = self.products
        elif side == "agents":
            members = self.agents
        else:
            raise ValueError("side must be reactants, products, or agents")
        if not members:
            raise ValueError(f"reaction {self.smiles!r} has no {side}")
        combined = members[0].mol
        for member in members[1:]:
            combined = Chem.CombineMols(combined, member.mol)
        return combined

    @property
    def principal_reactant(self) -> MoleculeObject:
        """Largest reactant, used as the split key for scaffold/size splits."""
        return max(self.reactants, key=lambda item: (item.num_heavy_atoms, item.smiles))


ChemicalRecord = MoleculeObject | ReactionObject


def record_split_smiles(record: ChemicalRecord) -> str:
    """SMILES used by structure-based splitters (scaffold, size, SMARTS)."""
    if isinstance(record, ReactionObject):
        return record.principal_reactant.smiles
    return record.smiles


def record_molecule(record: ChemicalRecord, side: str = "reactants") -> Any:
    """RDKit molecule used by molecular feature calculators and graph encoders."""
    if isinstance(record, ReactionObject):
        return record.combined(side)
    return record.mol


__all__ = [
    "ChemicalRecord",
    "MoleculeObject",
    "ReactionObject",
    "record_molecule",
    "record_split_smiles",
]
