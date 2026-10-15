from __future__ import annotations
import importlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Mapping
import numpy as np
from .objects import ChemicalRecord, ComplexObject, MixtureObject, MoleculeObject, ReactionObject

_MAX_INLINE_FILE_BYTES = 50_000_000
_AMINO_ACIDS = frozenset("ACDEFGHIKLMNOPQRSTUVWYXBZJ")

def _rdkit() -> Any:
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError("RDKit is required for chemical input adapters") from exc
    return Chem

def _coordinates(mol: Any) -> np.ndarray | None:
    if mol.GetNumConformers() == 0:
        return None
    positions = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64)
    if positions.size == 0 or not np.any(np.abs(positions) > 1e-8):
        return None
    return positions

def _molecule(mol: Any, source_format: str, identifier: str | None = None) -> MoleculeObject:
    Chem = _rdkit()
    return MoleculeObject(
        mol=mol,
        smiles=Chem.MolToSmiles(mol, canonical=True),
        source_format=source_format,
        coordinates=_coordinates(mol),
        identifier=identifier,
    )

class InputAdapter(ABC):
    """Convert one raw dataset cell into a :class:`MoleculeObject` or reaction."""

    name: str = "abstract"
    record_kind: str = "molecule"

    def __init__(self, base_dir: Path | None = None, **options: Any) -> None:
        self.base_dir = base_dir
        self.options = dict(options)

    @abstractmethod
    def parse(self, value: str, *, row: int | None = None) -> ChemicalRecord:
        """Parse one raw value; raise ``ValueError`` for invalid chemistry."""

    def _location(self, row: int | None) -> str:
        return f" at row {row}" if row is not None else ""

    def _read_source(self, value: str, suffixes: tuple[str, ...]) -> str:
        """Return inline text, or the content of a file path inside ``base_dir``.

        File references are only followed when they end with an allowed suffix.
        Paths are resolved against the dataset directory and must stay inside
        it unless ``allow_external_paths: true`` is configured, which prevents
        a data file from silently reading arbitrary files on the machine.
        """
        candidate = value.strip()
        if "\n" in candidate or not candidate.lower().endswith(suffixes):
            return value
        path = Path(candidate).expanduser()
        base = (self.base_dir or Path.cwd()).resolve()
        resolved = (path if path.is_absolute() else base / path).resolve()
        if not self.options.get("allow_external_paths", False):
            try:
                resolved.relative_to(base)
            except ValueError as exc:
                raise ValueError(
                    f"input file {candidate!r} is outside the dataset directory {base}; "
                    "set dataset.input_options.allow_external_paths: true to allow it"
                ) from exc
        if not resolved.is_file():
            raise FileNotFoundError(f"input file not found: {resolved}")
        if resolved.stat().st_size > _MAX_INLINE_FILE_BYTES:
            raise ValueError(f"input file is too large: {resolved}")
        return resolved.read_text(encoding="utf-8")

class SMILESAdapter(InputAdapter):
    name = "smiles"

    def parse(self, value: str, *, row: int | None = None) -> MoleculeObject:
        Chem = _rdkit()
        mol = Chem.MolFromSmiles(value.strip())
        if mol is None:
            raise ValueError(f"invalid SMILES{self._location(row)}: {value!r}")
        return _molecule(mol, self.name)

class SELFIESAdapter(InputAdapter):
    name = "selfies"

    def parse(self, value: str, *, row: int | None = None) -> MoleculeObject:
        try:
            selfies = importlib.import_module("selfies")
        except ImportError as exc:
            raise ImportError("the SELFIES adapter requires the 'selfies' package") from exc
        Chem = _rdkit()
        smiles = selfies.decoder(value.strip())
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            raise ValueError(f"invalid SELFIES{self._location(row)}: {value!r}")
        return _molecule(mol, self.name)

class InChIAdapter(InputAdapter):
    name = "inchi"

    def parse(self, value: str, *, row: int | None = None) -> MoleculeObject:
        Chem = _rdkit()
        mol = Chem.MolFromInchi(value.strip())
        if mol is None:
            raise ValueError(f"invalid InChI{self._location(row)}: {value!r}")
        return _molecule(mol, self.name)

class MolBlockAdapter(InputAdapter):
    name = "molblock"
    def parse(self, value: str, *, row: int | None = None) -> MoleculeObject:
        Chem = _rdkit()
        text = self._read_source(value, (".mol", ".sdf"))
        block = text.split("$$$$", 1)[0]
        mol = Chem.MolFromMolBlock(
            block,
            removeHs=bool(self.options.get("remove_hs", False)),
        )
        if mol is None:
            raise ValueError(f"invalid MolBlock/SDF{self._location(row)}")
        return _molecule(mol, self.name)

class XYZAdapter(InputAdapter):
    name = "xyz"
    def parse(self, value: str, *, row: int | None = None) -> MoleculeObject:
        Chem = _rdkit()
        from rdkit.Chem import rdDetermineBonds

        text = self._read_source(value, (".xyz",))
        mol = Chem.MolFromXYZBlock(text)
        if mol is None:
            raise ValueError(f"invalid XYZ block{self._location(row)}")
        charge = int(self.options.get("charge", 0))
        try:
            rdDetermineBonds.DetermineBonds(mol, charge=charge)
        except Exception as exc:
            raise ValueError(
                f"could not perceive bonds from XYZ geometry{self._location(row)}"
            ) from exc

        output = Chem.RemoveHs(mol) if self.options.get("remove_hs", False) else mol
        return _molecule(output, self.name)

class ReactionSMILESAdapter(InputAdapter):
    name = "reaction_smiles"
    record_kind = "reaction"
    def parse(self, value: str, *, row: int | None = None) -> ReactionObject:
        Chem = _rdkit()
        parts = value.strip().split(">")
        if len(parts) != 3:
            raise ValueError(f"invalid reaction SMILES{self._location(row)}: {value!r}")

        def side(text: str) -> tuple[MoleculeObject, ...]:
            members: list[MoleculeObject] = []
            for fragment in (item for item in text.split(".") if item):
                mol = Chem.MolFromSmiles(fragment)
                if mol is None:
                    raise ValueError(
                        f"invalid reaction component{self._location(row)}: {fragment!r}"
                    )
                members.append(_molecule(mol, "smiles"))
            return tuple(members)

        reactants, agents, products = side(parts[0]), side(parts[1]), side(parts[2])
        if not reactants or not products:
            raise ValueError(
                f"reaction SMILES needs reactants and products{self._location(row)}"
            )
        canonical = ">".join(
            ".".join(sorted(member.smiles for member in group))
            for group in (reactants, agents, products)
        )
        return ReactionObject(
            reactants=reactants,
            products=products,
            agents=agents,
            smiles=canonical,
            source_format=self.name,
        )

class MixtureAdapter(InputAdapter):
    name = "mixture"
    record_kind = "mixture"
    def parse(self, value: str, *, row: int | None = None) -> MixtureObject:
        Chem = _rdkit()
        component_sep = str(self.options.get("component_separator", ";"))
        fraction_sep = str(self.options.get("fraction_separator", ":"))
        parts = [item.strip() for item in value.strip().split(component_sep) if item.strip()]
        if not parts:
            raise ValueError(f"empty mixture{self._location(row)}")
        components: list[MoleculeObject] = []
        fractions: list[float] = []
        explicit: list[bool] = []
        for part in parts:
            smiles_text, fraction = part, None
            if fraction_sep in part:
                head, tail = part.rsplit(fraction_sep, 1)
                try:
                    fraction = float(tail)
                    smiles_text = head
                except ValueError:
                    fraction = None
            mol = Chem.MolFromSmiles(smiles_text)
            if mol is None:
                raise ValueError(f"invalid mixture component{self._location(row)}: {smiles_text!r}")
            components.append(_molecule(mol, "smiles"))
            fractions.append(1.0 if fraction is None else fraction)
            explicit.append(fraction is not None)
        if any(explicit) and not all(explicit):
            raise ValueError(
                f"mixture{self._location(row)} must give fractions for all components or none"
            )
        canonical = component_sep.join(
            f"{component.smiles}{fraction_sep}{fraction:g}"
            for component, fraction in zip(components, fractions)
        )
        return MixtureObject(
            components=tuple(components),
            fractions=tuple(fractions),
            smiles=canonical,
            source_format=self.name,
        )

class ComplexAdapter(InputAdapter):
    name = "complex"
    record_kind = "complex"
    def parse(self, value: str, *, row: int | None = None) -> ComplexObject:
        Chem = _rdkit()
        separator = str(self.options.get("separator", "|"))
        max_length = int(self.options.get("max_sequence_length", 5000))
        if value.count(separator) != 1:
            raise ValueError(
                f"complex input{self._location(row)} must be 'SMILES{separator}SEQUENCE'"
            )
        smiles_text, sequence = (item.strip() for item in value.split(separator))
        mol = Chem.MolFromSmiles(smiles_text)
        if mol is None:
            raise ValueError(f"invalid ligand SMILES{self._location(row)}: {smiles_text!r}")
        sequence = sequence.upper()
        if not sequence or len(sequence) > max_length or set(sequence) - _AMINO_ACIDS:
            raise ValueError(
                f"invalid protein sequence{self._location(row)}: expected 1..{max_length} "
                "one-letter amino-acid codes"
            )
        ligand = _molecule(mol, "smiles")
        return ComplexObject(
            ligand=ligand,
            protein_sequence=sequence,
            smiles=f"{ligand.smiles}{separator}{sequence}",
            source_format=self.name,
        )

AdapterFactory = Callable[..., InputAdapter]
_ADAPTERS: dict[str, AdapterFactory] = {
    SMILESAdapter.name: SMILESAdapter,
    SELFIESAdapter.name: SELFIESAdapter,
    InChIAdapter.name: InChIAdapter,
    MolBlockAdapter.name: MolBlockAdapter,
    "sdf": MolBlockAdapter,
    XYZAdapter.name: XYZAdapter,
    ReactionSMILESAdapter.name: ReactionSMILESAdapter,
    MixtureAdapter.name: MixtureAdapter,
    ComplexAdapter.name: ComplexAdapter,
}

def register_input_adapter(name: str, factory: AdapterFactory, *, replace: bool = False) -> None:
    if not name or not isinstance(name, str):
        raise ValueError("adapter name must be a non-empty string")
    if name in _ADAPTERS and not replace:
        raise ValueError(f"input adapter {name!r} is already registered")
    _ADAPTERS[name] = factory

def available_input_formats() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))

def get_input_adapter(name: str, *, base_dir: Path | None = None, options: Mapping[str, Any] | None = None,) -> InputAdapter:
    try:
        factory = _ADAPTERS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown input format {name!r}; available: {', '.join(available_input_formats())}"
        ) from exc
    return factory(base_dir=base_dir, **dict(options or {}))

__all__ = [
    "ComplexAdapter",
    "InChIAdapter",
    "InputAdapter",
    "MixtureAdapter",
    "MolBlockAdapter",
    "ReactionSMILESAdapter",
    "SELFIESAdapter",
    "SMILESAdapter",
    "XYZAdapter",
    "available_input_formats",
    "get_input_adapter",
    "register_input_adapter",
]
