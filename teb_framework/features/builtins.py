"""Default, broadly applicable molecular descriptor calculators."""

from __future__ import annotations

from collections.abc import Sequence

from .registry import FeatureRegistry
from .specs import FeatureCalculator, FeatureSpec


def molecular_feature_registry(
    elements: Sequence[str] = ("C", "H", "N", "O", "S", "F", "Cl", "Br", "P", "I"),
) -> FeatureRegistry:
    """Create a registry of general-purpose molecular composition and topology features."""
    try:
        from rdkit.Chem import rdMolDescriptors
    except ImportError as exc:
        raise ImportError("RDKit is required for built-in molecular features") from exc

    registry = FeatureRegistry()

    def atom_count(symbol: str) -> FeatureCalculator:
        return lambda mol: float(
            sum(atom.GetSymbol() == symbol for atom in mol.GetAtoms())
            if symbol != "H"
            else sum(atom.GetTotalNumHs(includeNeighbors=False) for atom in mol.GetAtoms())
            + sum(atom.GetSymbol() == "H" for atom in mol.GetAtoms())
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
