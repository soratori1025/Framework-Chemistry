"""Canonical dataset loading for the feature-selection benchmarks.

This module is the single entry point the recommendation benchmarks use to obtain
a real molecular design matrix. It exists so that ``run_recommendation_ablation``
(and any other experiment) can ask for a dataset *without* embedding the
descriptor pipeline, the split manifests, or the target-column conventions.

Design notes
------------
* **One manifest, one truth.** Datasets, target columns, task types and split
  files all come from ``data/moleculenet/dataset_manifest.json``; nothing is
  hard-coded here beyond the fallback chain.
* **Leakage boundary is the caller's job.** :func:`load_split` returns the full
  design matrix with the training/validation/test index arrays attached. It never
  returns a pre-subsetted matrix, and it never scales anything, so a selector
  cannot accidentally see held-out statistics through this module.
* **Fail loudly here, degrade in the caller.** Callers that must survive a
  missing dataset should catch the exception; silently returning synthetic data
  in place of a requested real dataset is what allowed a previous version of
  E-REC to report "molecular" results that were entirely synthetic.

Typical use::

    from benchmarks.datasets import load_dataset, load_split

    data = load_dataset("esol")
    subset = load_split("esol", seed=1)
    X_train = data.features[subset.train]
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "moleculenet"
MANIFEST_PATH = DATA_DIR / "dataset_manifest.json"

# Three-way protocol constants, kept here so every experiment shares them.
TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1


class DatasetUnavailableError(RuntimeError):
    """Raised when a requested dataset is missing, empty or malformed."""


@dataclass(frozen=True)
class Dataset:
    """A molecular design matrix with its target and provenance.

    Attributes:
        name: Canonical dataset key, e.g. ``"esol"``.
        features: ``(n_samples, n_features)`` float array of RDKit 2D descriptors.
        target: ``(n_samples,)`` float array; class labels are cast to 0/1 for
            classification so downstream metrics stay uniform.
        feature_names: Column names, aligned with ``features``.
        smiles: SMILES strings in row order, retained for auditing.
        is_classification: Whether the endpoint is binary classification.
        target_name: Source column name of the target.
        metadata: Raw manifest entry, kept for provenance records.
    """

    name: str
    features: np.ndarray
    target: np.ndarray
    feature_names: list[str]
    smiles: list[str]
    is_classification: bool
    target_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return int(self.features.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.features.shape[1])


@dataclass(frozen=True)
class SplitIndices:
    """Index arrays for one saved split manifest."""

    name: str
    seed: int
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.name,
            "seed": self.seed,
            "n_train": int(self.train.size),
            "n_val": int(self.val.size),
            "n_test": int(self.test.size),
        }


def _load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise DatasetUnavailableError(
            f"dataset manifest not found at {MANIFEST_PATH}; "
            "run benchmarks/prepare_moleculenet_data.py first"
        )
    with MANIFEST_PATH.open("r") as handle:
        return json.load(handle)


def available_datasets() -> list[str]:
    """Return the canonical dataset keys declared in the manifest."""
    return sorted(_load_manifest().keys())


def _encode_target(values: np.ndarray, is_classification: bool) -> np.ndarray:
    """Cast a target column to a finite float vector.

    Classification labels arrive as strings in some MoleculeNet distributions.
    They are mapped to ``0.0``/``1.0`` in sorted label order, which keeps AUROC
    and average precision well-defined without assuming a particular encoding.
    """
    if is_classification:
        as_text = pd.Series(values).astype(str)
        unique = sorted(as_text.unique())
        if len(unique) > 2:
            raise DatasetUnavailableError(
                f"classification target has {len(unique)} levels; "
                "only binary endpoints are supported"
            )
        mapping = {label: float(i) for i, label in enumerate(unique)}
        return as_text.map(mapping).to_numpy(dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def _descriptor_cache_dir() -> Path:
    """Return the cache directory for derived descriptor matrices."""
    path = DATA_DIR / ".descriptor_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _smiles_fingerprint(smiles: Sequence[str]) -> str:
    """Return a stable hash of a SMILES list.

    Keyed on contents rather than file mtime so a regenerated CSV with identical
    molecules still hits the cache, while any change to the molecule set (or its
    order) misses it. The order matters because the cached matrix has to stay
    row-aligned with the target vector.
    """
    digest = hashlib.sha256()
    digest.update(str(len(smiles)).encode("utf-8"))
    for item in smiles:
        digest.update(b"\x00")
        digest.update(item.encode("utf-8", errors="replace"))
    return digest.hexdigest()[:32]


def _descriptor_cache_path(name: str, task: str | None, smiles: Sequence[str]) -> Path:
    """Return the cache file for one (dataset, task, molecule-set) triple."""
    key = f"{name}__{task or 'default'}__{_smiles_fingerprint(smiles)}"
    return _descriptor_cache_dir() / f"{key}.npz"


def _compute_descriptors(
    name: str,
    task: str | None,
    smiles: Sequence[str],
    *,
    use_cache: bool,
) -> tuple[np.ndarray, list[str]]:
    """Derive RDKit 2D descriptors, reusing a cached matrix when possible.

    Descriptor derivation costs ~5 s per 1000 molecules, which dominates a
    multi-dataset sweep. The cache stores only *derived descriptors*, never
    targets or split indices, so it cannot leak supervision across folds.

    The matrix is stored as plain float64 and the column names in a sibling
    ``.json`` file. Names are deliberately *not* stored inside the ``.npz``:
    an object-dtype array there can only be read back with
    ``allow_pickle=True``, which would either raise (and silently disable the
    cache) or reintroduce a pickle-deserialisation hazard for no benefit.
    """
    cache_path = _descriptor_cache_path(name, task, smiles)
    names_path = cache_path.with_suffix(".names.json")
    if use_cache and cache_path.exists() and names_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                matrix = np.asarray(cached["matrix"], dtype=np.float64)
            with names_path.open("r", encoding="utf-8") as handle:
                names = json.load(handle)
            if isinstance(names, list) and matrix.shape == (len(smiles), len(names)):
                return matrix, [str(n) for n in names]
        except Exception:
            # A corrupt cache must never break a run; fall through and recompute.
            pass

    from chemistry_framework.features.pharmacophore import extract_rdkit_2d_descriptors

    matrix, feature_names = extract_rdkit_2d_descriptors(list(smiles))
    matrix = np.asarray(matrix, dtype=np.float64)
    names = [str(n) for n in feature_names]

    if use_cache:
        try:
            np.savez_compressed(cache_path, matrix=matrix)
            with names_path.open("w", encoding="utf-8") as handle:
                json.dump(names, handle)
        except Exception:
            # Caching is an optimisation; a read-only disk must not fail the run.
            pass
    return matrix, names


def load_dataset(
    name: str,
    *,
    task: str | None = None,
    drop_constant: bool = True,
    use_cache: bool = True,
) -> Dataset:
    """Load a canonical dataset as a descriptor matrix plus target.

    Args:
        name: Dataset key from ``dataset_manifest.json``.
        task: Which endpoint to use for multi-task datasets. ``None`` (default)
            selects ``targets[0]`` and is only allowed when that target has no
            missing labels. Naming the task explicitly is required for datasets
            such as tox21 that carry masked (NaN) labels per assay, because
            silently dropping masked rows changes the sample set and must be a
            deliberate, recorded choice.
        drop_constant: When ``True`` (default) columns whose standard deviation
            is zero are removed. These carry no selector signal and would
            otherwise be ranked arbitrarily by tie-breaking.
        use_cache: Reuse a previously derived descriptor matrix when the
            molecule set is unchanged. Set ``False`` to force recomputation.

    Raises:
        DatasetUnavailableError: If the dataset is unknown, the CSV is missing,
            the task is ambiguous, or no usable rows/columns remain.
    """
    manifest = _load_manifest()
    if name not in manifest:
        raise DatasetUnavailableError(
            f"unknown dataset {name!r}; available: {', '.join(sorted(manifest))}"
        )
    entry = manifest[name]

    csv_path = REPO_ROOT / entry["clean_csv"]
    if not csv_path.exists():
        raise DatasetUnavailableError(f"dataset CSV not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    smiles_col = entry["smiles_col"]
    if smiles_col not in frame.columns:
        raise DatasetUnavailableError(
            f"column {smiles_col!r} missing from {csv_path.name}; "
            "regenerate the dataset with benchmarks/prepare_moleculenet_data.py"
        )

    targets = list(entry["targets"])
    if task is not None:
        if task not in targets:
            raise DatasetUnavailableError(
                f"task {task!r} not declared for {name!r}; available: {targets}"
            )
        target_col = task
    else:
        target_col = targets[0]

    if target_col not in frame.columns:
        raise DatasetUnavailableError(
            f"target column {target_col!r} missing from {csv_path.name}"
        )

    is_classification = "classification" in entry["task_type"]

    # A masked multi-task label set cannot be used implicitly: the row count
    # would change under the caller's feet. Demand an explicit task instead.
    n_missing = int(frame[target_col].isna().sum())
    if n_missing and task is None:
        raise DatasetUnavailableError(
            f"{name!r} has {n_missing} masked labels for the default task "
            f"{target_col!r} (multi-task endpoint). Pass task=<one of {targets}> "
            "to state which assay you are evaluating; the masked rows will be "
            "dropped and that decision must be explicit."
        )

    # Rows with a missing target cannot contribute a supervised signal.
    valid = frame[target_col].notna().to_numpy()
    if not valid.all():
        frame = frame.loc[valid].reset_index(drop=True)

    smiles = [str(s) for s in frame[smiles_col].tolist()]
    target = _encode_target(frame[target_col].to_numpy(), is_classification)

    features, feature_names = _compute_descriptors(
        name, task, smiles, use_cache=use_cache
    )

    finite_rows = np.isfinite(features).all(axis=1)
    if not finite_rows.all():
        features = features[finite_rows]
        target = target[finite_rows]
        smiles = [s for s, keep in zip(smiles, finite_rows) if keep]

    if drop_constant:
        keep_cols = features.std(axis=0) > 1e-12
        features = features[:, keep_cols]
        feature_names = [n for n, keep in zip(feature_names, keep_cols) if keep]

    if features.shape[0] == 0 or features.shape[1] < 3:
        raise DatasetUnavailableError(
            f"dataset {name!r} reduced to an unusable shape {features.shape}"
        )

    return Dataset(
        name=name,
        features=features,
        target=target,
        feature_names=list(feature_names),
        smiles=smiles,
        is_classification=is_classification,
        target_name=target_col,
        metadata={
            **entry,
            "selected_task": target_col,
            "n_targets_available": len(targets),
            "n_rows_dropped_for_missing_target": n_missing,
        },
    )


def load_split(
    name: str,
    seed: int = 1,
    *,
    task: str | None = None,
) -> tuple[np.ndarray, np.ndarray, SplitIndices]:
    """Load a dataset together with one saved split manifest.

    Returns:
        ``(features, target, split)``. The target is returned separately because
        the recommendation benchmarks treat it as a first-class argument rather
        than an attribute of the design matrix.

    Note:
        Saved splits index the *full* cleaned CSV. When ``task`` causes masked
        rows to be dropped the resulting matrix is shorter than the split's
        index space, so the split is rejected rather than silently misaligned.
        Multi-task endpoints with masked labels therefore need their own
        manifest generated for the specific task.
    """
    dataset = load_dataset(name, task=task)
    manifest = _load_manifest()[name]
    expected_rows = int(manifest["n_samples"])
    if dataset.n_samples != expected_rows:
        raise DatasetUnavailableError(
            f"{name!r} loaded {dataset.n_samples} rows but the saved splits index "
            f"{expected_rows}. Splits cannot be reused after dropping rows for a "
            "masked task; regenerate them with "
            "benchmarks/prepare_moleculenet_data.py."
        )
    key = f"seed_{seed}"
    if key not in manifest.get("splits", {}):
        available = ", ".join(sorted(manifest.get("splits", {})))
        raise DatasetUnavailableError(
            f"split {key!r} unavailable for {name!r}; have: {available}"
        )
    split_file = REPO_ROOT / manifest["splits"][key]["split_file"]
    if not split_file.exists():
        raise DatasetUnavailableError(f"split manifest not found: {split_file}")

    with split_file.open("r") as handle:
        raw = json.load(handle)

    def _as_index(values: Sequence[int]) -> np.ndarray:
        arr = np.asarray(values, dtype=np.int64)
        if arr.size and (arr.min() < 0 or arr.max() >= dataset.n_samples):
            raise DatasetUnavailableError(
                f"split {split_file.name} references rows outside the loaded "
                f"matrix ({dataset.n_samples} rows); regenerate the splits"
            )
        return arr

    split = SplitIndices(
        name=name,
        seed=seed,
        train=_as_index(raw["train"]),
        val=_as_index(raw["val"]),
        test=_as_index(raw["test"]),
    )
    return dataset.features, dataset.target, split


def load_descriptor_matrix(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Convenience wrapper returning the raw ``(X, y, names)`` triple."""
    dataset = load_dataset(name)
    return dataset.features, dataset.target, list(dataset.feature_names)



