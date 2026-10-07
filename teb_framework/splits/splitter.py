"""Reproducible random, size, scaffold, and SMARTS data splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..config.schema import SplitConfig


@dataclass(frozen=True)
class DatasetSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    def __post_init__(self) -> None:
        arrays = [np.asarray(index, dtype=np.int64) for index in (self.train, self.validation, self.test)]
        combined = np.concatenate(arrays)
        if len(np.unique(combined)) != len(combined):
            raise ValueError("split indices overlap")
        if any(array.ndim != 1 for array in arrays):
            raise ValueError("split indices must be one-dimensional")
        if any(array.size == 0 for array in arrays):
            raise ValueError("train, validation, and test splits must all be non-empty")
        object.__setattr__(self, "train", arrays[0])
        object.__setattr__(self, "validation", arrays[1])
        object.__setattr__(self, "test", arrays[2])


def _partition(indices: np.ndarray, fractions: tuple[float, float, float]) -> DatasetSplit:
    n = len(indices)
    n_train = max(1, int(n * fractions[0]))
    n_validation = max(1, int(n * fractions[1]))
    if n_train + n_validation >= n:
        n_validation = max(1, n - n_train - 1)
    if n_train + n_validation >= n:
        n_train = n - 2
    if n_train < 1 or n_validation < 1 or n - n_train - n_validation < 1:
        raise ValueError("at least three records are required for train/validation/test")
    return DatasetSplit(
        indices[:n_train],
        indices[n_train:n_train + n_validation],
        indices[n_train + n_validation:],
    )


def split_indices(smiles: Sequence[str], config: SplitConfig) -> DatasetSplit:
    """Create deterministic dataset partitions without consulting target values."""
    if len(smiles) < 3:
        raise ValueError("at least three molecules are required for a three-way split")
    rng = np.random.default_rng(config.seed)
    indices = np.arange(len(smiles), dtype=np.int64)

    if config.method == "random":
        rng.shuffle(indices)
        return _partition(indices, config.fractions)
    if config.method == "size":
        try:
            from rdkit import Chem
        except ImportError as exc:
            raise ImportError("RDKit is required for size-based splitting") from exc
        sizes: list[int] = []
        for row, value in enumerate(smiles):
            mol = Chem.MolFromSmiles(value)
            if mol is None:
                raise ValueError(f"invalid SMILES at row {row}: {value!r}")
            sizes.append(mol.GetNumHeavyAtoms())
        ordered = np.asarray(sorted(indices, key=lambda i: (sizes[int(i)], int(i))), dtype=np.int64)
        return _partition(ordered, config.fractions)
    if config.method == "smarts":
        return _smarts_split(smiles, config, rng, indices)
    if config.method == "scaffold":
        return _scaffold_split(smiles, config, rng)
    raise ValueError(f"unsupported split method: {config.method}")


def _smarts_split(
    smiles: Sequence[str],
    config: SplitConfig,
    rng: np.random.Generator,
    indices: np.ndarray,
) -> DatasetSplit:
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError("RDKit is required for SMARTS-based splitting") from exc
    query = Chem.MolFromSmarts(config.test_smarts or "")
    if query is None:
        raise ValueError(f"invalid SMARTS pattern: {config.test_smarts!r}")
    matches: list[int] = []
    remainder: list[int] = []
    for row, value in enumerate(smiles):
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            raise ValueError(f"invalid SMILES at row {row}: {value!r}")
        (matches if mol.HasSubstructMatch(query) else remainder).append(row)
    if not matches or len(remainder) < 2:
        raise ValueError("SMARTS split has no matching test molecules or too few remaining molecules")
    rng.shuffle(remainder)
    relative_val = config.fractions[1] / (config.fractions[0] + config.fractions[1])
    n_val = min(len(remainder) - 1, max(1, round(len(remainder) * relative_val)))
    return DatasetSplit(
        np.asarray(remainder[n_val:], dtype=np.int64),
        np.asarray(remainder[:n_val], dtype=np.int64),
        np.asarray(matches, dtype=np.int64),
    )


def _scaffold_split(
    smiles: Sequence[str],
    config: SplitConfig,
    rng: np.random.Generator,
) -> DatasetSplit:
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError as exc:
        raise ImportError("RDKit is required for scaffold splitting") from exc
    groups: dict[str, list[int]] = {}
    for row, value in enumerate(smiles):
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            raise ValueError(f"invalid SMILES at row {row}: {value!r}")
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True)
        groups.setdefault(scaffold or f"acyclic:{Chem.MolToSmiles(mol)}", []).append(row)

    group_values = list(groups.values())
    rng.shuffle(group_values)
    group_values.sort(key=len, reverse=True)
    total = len(smiles)
    targets = np.asarray(config.fractions) * total
    bins: list[list[int]] = [[], [], []]
    for group in group_values:
        ratios = [len(bins[i]) / max(targets[i], 1.0) for i in range(3)]
        selected = int(np.argmin(ratios))
        bins[selected].extend(group)
    if any(not values for values in bins):
        raise ValueError("scaffold groups cannot populate all three requested splits")
    return DatasetSplit(*(np.asarray(values, dtype=np.int64) for values in bins))
