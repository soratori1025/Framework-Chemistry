"""SMARTS-derived per-atom and per-bond graph input flags."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SMARTSFlag:
    name: str
    pattern: str

    def __post_init__(self) -> None:
        if not self.name or not self.pattern:
            raise ValueError("SMARTS flag name and pattern must not be empty")


class AtomSMARTSFlags:
    """Create one binary graph feature per SMARTS flag and atom."""

    def __init__(self, flags: Sequence[SMARTSFlag]) -> None:
        self.flags = tuple(flags)
        if len({flag.name for flag in self.flags}) != len(self.flags):
            raise ValueError("SMARTS flag names must be unique")
        try:
            from rdkit import Chem
        except ImportError as exc:
            raise ImportError("RDKit is required for SMARTS atom flags") from exc
        self._queries = tuple(Chem.MolFromSmarts(flag.pattern) for flag in self.flags)
        if any(query is None for query in self._queries):
            invalid = [
                flag.name for flag, query in zip(self.flags, self._queries) if query is None
            ]
            raise ValueError(f"invalid atom SMARTS pattern(s): {', '.join(invalid)}")

    @classmethod
    def from_config(cls, flags: Sequence[dict[str, str]]) -> AtomSMARTSFlags:
        return cls(tuple(SMARTSFlag(**flag) for flag in flags))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(flag.name for flag in self.flags)

    def __call__(self, mol: object) -> np.ndarray:
        if not hasattr(mol, "GetNumAtoms"):
            raise TypeError("AtomSMARTSFlags expects an RDKit molecule")
        values = np.zeros((mol.GetNumAtoms(), len(self.flags)), dtype=np.float32)
        for column, query in enumerate(self._queries):
            for match in mol.GetSubstructMatches(query, uniquify=True):
                values[list(match), column] = 1.0
        return values


class BondSMARTSFlags:
    """Create one binary graph feature per SMARTS flag and molecular bond."""

    def __init__(self, flags: Sequence[SMARTSFlag]) -> None:
        self.flags = tuple(flags)
        if len({flag.name for flag in self.flags}) != len(self.flags):
            raise ValueError("SMARTS flag names must be unique")
        try:
            from rdkit import Chem
        except ImportError as exc:
            raise ImportError("RDKit is required for SMARTS bond flags") from exc
        self._queries = tuple(Chem.MolFromSmarts(flag.pattern) for flag in self.flags)
        if any(query is None for query in self._queries):
            invalid = [
                flag.name for flag, query in zip(self.flags, self._queries) if query is None
            ]
            raise ValueError(f"invalid bond SMARTS pattern(s): {', '.join(invalid)}")

    @classmethod
    def from_config(cls, flags: Sequence[dict[str, str]]) -> BondSMARTSFlags:
        return cls(tuple(SMARTSFlag(**flag) for flag in flags))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(flag.name for flag in self.flags)

    def __call__(self, mol: object) -> np.ndarray:
        if not hasattr(mol, "GetNumBonds"):
            raise TypeError("BondSMARTSFlags expects an RDKit molecule")
        values = np.zeros((mol.GetNumBonds(), len(self.flags)), dtype=np.float32)
        for column, query in enumerate(self._queries):
            for match in mol.GetSubstructMatches(query, uniquify=True):
                for query_bond in query.GetBonds():
                    atom_begin = match[query_bond.GetBeginAtomIdx()]
                    atom_end = match[query_bond.GetEndAtomIdx()]
                    bond = mol.GetBondBetweenAtoms(atom_begin, atom_end)
                    if bond is not None:
                        values[bond.GetIdx(), column] = 1.0
        return values
