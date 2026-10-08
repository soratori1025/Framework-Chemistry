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

    def add_smarts(
        self,
        name: str,
        smarts: str,
        *,
        reaction_side: str = "reactants",
        mode: str = "count",
        applies_to: Sequence[str] = (),
        group: str | None = None,
        description: str | None = None,
        interpretation: str | None = None,
        source: str = "smarts",
        definition: str | None = None,
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
                group=group,
                description=description,
                interpretation=interpretation,
                source=source,
                definition=definition or f"SMARTS {mode}: {smarts}",
                reaction_side=reaction_side,
            )
        )

    def add_condition(
        self,
        name: str,
        column: str,
        *,
        transform: str = "identity",
        group: str | None = None,
        description: str | None = None,
        interpretation: str | None = None,
        source: str = "condition",
        definition: str | None = None,
    ) -> None:
        """Register a dataset condition as a feature."""
        def calculate(mol: Any) -> float:
            raise RuntimeError("Condition features must be extracted directly from the dataset")

        self.register(
            FeatureSpec(
                name=name,
                calculator=calculate,
                domain="condition",
                metadata={"column": column, "transform": transform},
                group=group,
                description=description,
                interpretation=interpretation,
                source=source,
                definition=definition or f"Condition {column} ({transform})",
                reaction_side="dataset",
            )
        )

    def add_rdkit_descriptors(
        self,
        name: str,
        *,
        reaction_side: str = "reactants",
        names: Sequence[str] | None = None,
        families: Sequence[str] | None = None,
        group: str | None = None,
        description: str | None = None,
        interpretation: str | None = None,
        source: str = "rdkit_descriptors",
        definition: str | None = None,
    ) -> None:
        try:
            from rdkit.Chem import Descriptors
        except ImportError as exc:
            raise ImportError("RDKit is required for descriptors") from exc

        # Expand families or use provided names
        if families:
            resolved_names = []
            for family in families:
                if family == "all":
                    resolved_names.extend([n for n, _ in Descriptors.descList])
                # We could add more families like 'properties', 'fragments'
            descriptor_names = resolved_names
        else:
            descriptor_names = names or []

        available_descs = dict(Descriptors.descList)
        for desc_name in descriptor_names:
            if desc_name not in available_descs:
                continue
            
            calculator = available_descs[desc_name]
            feature_name = f"{name}.{desc_name}"
            
            self.register(
                FeatureSpec(
                    name=feature_name,
                    calculator=calculator,
                    domain="rdkit_descriptor",
                    group=group or name,
                    description=description,
                    interpretation=interpretation,
                    source=source,
                    definition=definition or desc_name,
                    reaction_side=reaction_side,
                )
            )

    def add_morgan(
        self,
        name: str,
        *,
        reaction_side: str = "reactants",
        radius: int = 2,
        n_bits: int = 1024,
        counts: bool = False,
        group: str | None = None,
        description: str | None = None,
        interpretation: str | None = None,
        source: str = "morgan",
        definition: str | None = None,
    ) -> None:
        try:
            from rdkit.Chem import AllChem
        except ImportError as exc:
            raise ImportError("RDKit is required for Morgan fingerprints") from exc

        # Create one feature per bit
        for bit in range(n_bits):
            feature_name = f"{name}.bit_{bit}"
            
            def make_calculator(b: int) -> FeatureCalculator:
                def calculate(mol: Any) -> float:
                    if counts:
                        fp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=n_bits)
                        return float(fp[b])
                    else:
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                        return float(fp.GetBit(b))
                return calculate

            self.register(
                FeatureSpec(
                    name=feature_name,
                    calculator=make_calculator(bit),
                    domain="morgan",
                    group=group or name,
                    description=description,
                    interpretation=interpretation,
                    source=source,
                    definition=definition or f"Morgan r={radius} bit {bit}",
                    reaction_side=reaction_side,
                )
            )

    def add_descriptors_3d(
        self,
        name: str,
        *,
        reaction_side: str = "reactants",
        names: Sequence[str],
        seed: int = 0,
        group: str | None = None,
        description: str | None = None,
        interpretation: str | None = None,
        source: str = "descriptors_3d",
        definition: str | None = None,
    ) -> None:
        try:
            from rdkit.Chem import Descriptors3D
        except ImportError as exc:
            raise ImportError("RDKit is required for 3D descriptors") from exc

        available_3d = {
            "Asphericity": Descriptors3D.Asphericity,
            "Eccentricity": Descriptors3D.Eccentricity,
            "InertialShapeFactor": Descriptors3D.InertialShapeFactor,
            "NPR1": Descriptors3D.NPR1,
            "NPR2": Descriptors3D.NPR2,
            "PMI1": Descriptors3D.PMI1,
            "PMI2": Descriptors3D.PMI2,
            "PMI3": Descriptors3D.PMI3,
            "RadiusOfGyration": Descriptors3D.RadiusOfGyration,
            "SpherocityIndex": Descriptors3D.SpherocityIndex,
        }

        for desc_name in names:
            if desc_name not in available_3d:
                raise ValueError(f"Unknown 3D descriptor {desc_name}")
            
            calculator = available_3d[desc_name]
            feature_name = f"{name}.{desc_name}"

            # 3D descriptors require a conformation, which is provided in MoleculeObject
            # We will handle the embedding in the transformation step if needed
            self.register(
                FeatureSpec(
                    name=feature_name,
                    calculator=calculator,
                    domain="descriptor_3d",
                    group=group or name,
                    description=description,
                    interpretation=interpretation,
                    source=source,
                    definition=definition or desc_name,
                    reaction_side=reaction_side,
                )
            )
