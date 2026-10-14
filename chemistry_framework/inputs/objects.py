from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Mapping
import numpy as np
from rdkit import Chem

@dataclass(frozen=True)
class MoleculeObject:
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
    reactants: tuple[MoleculeObject, ...]
    products: tuple[MoleculeObject, ...]
    smiles: str
    source_format: str
    agents: tuple[MoleculeObject, ...] = ()
    identifier: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def combined(self, side: str) -> Any:
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
        return max(self.reactants, key=lambda item: (item.num_heavy_atoms, item.smiles))

@dataclass(frozen=True)
class MixtureObject:
    components: tuple[MoleculeObject, ...]
    fractions: tuple[float, ...]
    smiles: str
    source_format: str
    identifier: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("a mixture needs at least one component")
        if len(self.components) != len(self.fractions):
            raise ValueError("mixture components and fractions must have equal length")
        values = np.asarray(self.fractions, dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < 0) or values.sum() <= 0:
            raise ValueError("mixture fractions must be finite, non-negative, and not all zero")
        object.__setattr__(self, "fractions", tuple(float(v) for v in values / values.sum()))

    @property
    def principal_component(self) -> MoleculeObject:
        order = range(len(self.components))
        index = max(
            order,
            key=lambda i: (self.fractions[i], self.components[i].num_heavy_atoms, self.components[i].smiles),
        )
        return self.components[index]

    def combined(self) -> Any:
        combined = self.components[0].mol
        for member in self.components[1:]:
            combined = Chem.CombineMols(combined, member.mol)
        return combined

@dataclass(frozen=True)
class ComplexObject:
    ligand: MoleculeObject
    protein_sequence: str
    smiles: str
    source_format: str
    identifier: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

ChemicalRecord = MoleculeObject | ReactionObject | MixtureObject | ComplexObject

def record_split_smiles(record: ChemicalRecord) -> str:
    if isinstance(record, ReactionObject):
        return record.principal_reactant.smiles
    if isinstance(record, MixtureObject):
        return record.principal_component.smiles
    if isinstance(record, ComplexObject):
        return record.ligand.smiles
    return record.smiles

def record_molecule(record: ChemicalRecord, side: str = "reactants") -> Any:
    if isinstance(record, ReactionObject):
        return record.combined(side)
    if isinstance(record, MixtureObject):
        return record.combined()
    if isinstance(record, ComplexObject):
        return record.ligand.mol
    return record.mol

__all__ = [
    "ChemicalRecord",
    "ComplexObject",
    "MixtureObject",
    "MoleculeObject",
    "ReactionObject",
    "record_molecule",
    "record_split_smiles",
]
