"""Configuration- and molecule-specific feature cache."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .molecule import MoleculeFeatureBlock
from .specs import FeatureMatrix


class FeatureCache:
    """Persist one feature row per canonical molecule and feature schema."""

    def __init__(self, directory: str | Path, configuration_fingerprint: str) -> None:
        if not configuration_fingerprint:
            raise ValueError("configuration_fingerprint must not be empty")
        self.directory = Path(directory).expanduser()
        self.configuration_fingerprint = configuration_fingerprint

    def transform(
        self,
        block: MoleculeFeatureBlock,
        molecules: list[Any] | tuple[Any, ...],
    ) -> FeatureMatrix:
        try:
            from rdkit import Chem, rdBase
        except ImportError as exc:
            raise ImportError("feature caching for molecular inputs requires RDKit") from exc
        if self.configuration_fingerprint == "":
            raise ValueError("configuration_fingerprint must not be empty")
        self.directory.mkdir(parents=True, exist_ok=True)
        rows: list[np.ndarray] = []
        feature_names = np.asarray(block.feature_names, dtype=str)
        for row_index, item in enumerate(molecules):
            mol = Chem.MolFromSmiles(item) if isinstance(item, str) else item
            if mol is None:
                raise ValueError(f"invalid molecule at row {row_index}: {item!r}")
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            cache_key = hashlib.sha256(
                json.dumps(
                    {
                        "config": self.configuration_fingerprint,
                        "features": block.feature_names,
                        "molecule": canonical_smiles,
                        "rdkit": rdBase.rdkitVersion,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            cache_file = self.directory / f"{cache_key}.npz"
            if cache_file.is_file():
                with np.load(cache_file, allow_pickle=False) as cached:
                    cached_names = tuple(str(name) for name in cached["names"].tolist())
                    row = np.asarray(cached["values"], dtype=np.float64)
                if cached_names != block.feature_names or row.shape != (len(block.feature_names),):
                    raise ValueError(f"feature cache schema mismatch: {cache_file}")
                if not np.isfinite(row).all():
                    raise ValueError(f"feature cache contains non-finite values: {cache_file}")
                rows.append(row)
                continue

            row = block.transform([mol]).values[0]
            self._write_atomic(cache_file, feature_names, row)
            rows.append(row)
        values = np.vstack(rows) if rows else np.empty((0, len(block.feature_names)))
        return FeatureMatrix(block.feature_names, values)

    @staticmethod
    def _write_atomic(path: Path, names: np.ndarray, values: np.ndarray) -> None:
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{path.stem}.", suffix=".npz", dir=path.parent, delete=False
            ) as temporary:
                temporary_path = temporary.name
                np.savez_compressed(temporary, names=names, values=values)
            os.replace(temporary_path, path)
        except Exception:
            if temporary_path is not None:
                Path(temporary_path).unlink(missing_ok=True)
            raise
