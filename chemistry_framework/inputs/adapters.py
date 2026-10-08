"""Input adapters that normalize chemical notations into internal records.

Adapters are registered by name so YAML configurations can select them with
``dataset.input_format``. Users can register additional adapters (for example
CIF or PDB readers) with :func:`register_input_adapter`.
"""
from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .objects import ChemicalRecord, MoleculeObject, ReactionObject

_MAX_INLINE_FILE_BYTES = 50_000_000


def _rdkit() -> Any:
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - depends on optional extra
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
    """Inline MolBlock text or a ``.mol``/``.sdf`` file (first record)."""

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
    """Inline XYZ block or ``.xyz`` file; bonds are perceived from geometry."""

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
        # RemoveHs keeps the heavy-atom conformer, so coordinates survive.
        output = Chem.RemoveHs(mol) if self.options.get("remove_hs", False) else mol
        return _molecule(output, self.name)


class ReactionSMILESAdapter(InputAdapter):
    """Reaction SMILES ``reactants>agents>products`` or ``reactants>>products``."""

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


AdapterFactory = Callable[..., InputAdapter]
_ADAPTERS: dict[str, AdapterFactory] = {
    SMILESAdapter.name: SMILESAdapter,
    SELFIESAdapter.name: SELFIESAdapter,
    InChIAdapter.name: InChIAdapter,
    MolBlockAdapter.name: MolBlockAdapter,
    "sdf": MolBlockAdapter,
    XYZAdapter.name: XYZAdapter,
    ReactionSMILESAdapter.name: ReactionSMILESAdapter,
}


def register_input_adapter(name: str, factory: AdapterFactory, *, replace: bool = False) -> None:
    """Register a user adapter (e.g. CIF/PDB) under ``dataset.input_format``."""
    if not name or not isinstance(name, str):
        raise ValueError("adapter name must be a non-empty string")
    if name in _ADAPTERS and not replace:
        raise ValueError(f"input adapter {name!r} is already registered")
    _ADAPTERS[name] = factory


def available_input_formats() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))


def get_input_adapter(
    name: str,
    *,
    base_dir: Path | None = None,
    options: Mapping[str, Any] | None = None,
) -> InputAdapter:
    try:
        factory = _ADAPTERS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown input format {name!r}; available: {', '.join(available_input_formats())}"
        ) from exc
    return factory(base_dir=base_dir, **dict(options or {}))


__all__ = [
    "InChIAdapter",
    "InputAdapter",
    "MolBlockAdapter",
    "ReactionSMILESAdapter",
    "SELFIESAdapter",
    "SMILESAdapter",
    "XYZAdapter",
    "available_input_formats",
    "get_input_adapter",
    "register_input_adapter",
]
