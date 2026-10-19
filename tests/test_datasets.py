"""Tests for the canonical dataset loader used by the recommendation benchmarks.

These tests encode the failure that motivated the module: a benchmark asked for
real molecules, the loader was silently unavailable, and the caller substituted
synthetic data while still reporting ``source="molecular"``. Every guard here
exists to make that class of mislabelling impossible.
"""
from __future__ import annotations

import numpy as np
import pytest

from benchmarks.datasets import (
    MANIFEST_PATH,
    Dataset,
    DatasetUnavailableError,
    SplitIndices,
    available_datasets,
    load_dataset,
    load_split,
)

pytestmark = pytest.mark.skipif(
    not MANIFEST_PATH.exists(),
    reason="MoleculeNet manifest is unavailable; dataset tests need prepared data",
)


def _smallest_available_single_task() -> str:
    """Return a small single-task dataset key that is safe to load in tests."""
    manifest_datasets = available_datasets()
    for candidate in ("freesolv", "esol", "bbbp", "bace", "clintox"):
        if candidate in manifest_datasets:
            return candidate
    return manifest_datasets[0]


def test_available_datasets_are_sorted_and_nonempty():
    names = available_datasets()
    assert names, "manifest declared no datasets"
    assert names == sorted(names)


def test_unknown_dataset_raises_rather_than_returning_none():
    with pytest.raises(DatasetUnavailableError, match="unknown dataset"):
        load_dataset("definitely_not_a_dataset")


def test_load_dataset_returns_finite_aligned_arrays():
    name = _smallest_available_single_task()
    data = load_dataset(name)

    assert isinstance(data, Dataset)
    assert data.features.ndim == 2
    assert data.features.shape[0] == data.target.size == len(data.smiles)
    assert data.features.shape[1] == len(data.feature_names)
    assert np.isfinite(data.features).all(), "loader must not leak NaN/inf"
    assert np.isfinite(data.target).all()
    assert data.n_samples == data.features.shape[0]
    assert data.n_features == data.features.shape[1]


def test_constant_columns_are_dropped():
    name = _smallest_available_single_task()
    data = load_dataset(name)
    assert (data.features.std(axis=0) > 1e-12).all()


def test_classification_target_is_binary_encoded():
    data = load_dataset("bbbp")
    assert data.is_classification
    assert set(np.unique(data.target)).issubset({0.0, 1.0})


def test_load_split_partitions_every_row_exactly_once():
    name = _smallest_available_single_task()
    features, target, split = load_split(name, seed=1)

    assert isinstance(split, SplitIndices)
    assert features.shape[0] == target.size

    combined = np.concatenate([split.train, split.val, split.test])
    assert combined.size == features.shape[0], "split must cover every row"
    assert np.unique(combined).size == features.shape[0], "no row may repeat"
    assert combined.min() >= 0
    assert combined.max() < features.shape[0]

    assert not set(split.train) & set(split.val)
    assert not set(split.train) & set(split.test)
    assert not set(split.val) & set(split.test)


def test_load_split_rejects_missing_seed():
    name = _smallest_available_single_task()
    with pytest.raises(DatasetUnavailableError, match="split"):
        load_split(name, seed=99)


def test_split_indices_stay_in_bounds_for_every_seed():
    """A split referencing rows the loader dropped is the tox21 failure mode."""
    name = _smallest_available_single_task()
    n_rows = load_dataset(name).n_samples
    for seed in (1, 2, 3):
        _features, _target, split = load_split(name, seed=seed)
        for indices in (split.train, split.val, split.test):
            assert indices.size == 0 or int(indices.max()) < n_rows


@pytest.mark.skipif("tox21" not in available_datasets(), reason="tox21 not prepared")
def test_masked_multitask_requires_explicit_task():
    """tox21 has masked per-assay labels; guessing a task would be unscientific."""
    with pytest.raises(DatasetUnavailableError, match="masked labels"):
        load_dataset("tox21")


@pytest.mark.skipif("tox21" not in available_datasets(), reason="tox21 not prepared")
def test_masked_multitask_with_explicit_task_drops_only_masked_rows():
    data = load_dataset("tox21", task="NR-AR")
    dropped = data.metadata["n_rows_dropped_for_missing_target"]
    assert dropped > 0
    assert data.features.shape[0] == data.target.size
    assert np.isfinite(data.target).all()
    assert set(np.unique(data.target)).issubset({0.0, 1.0})
    assert data.metadata["selected_task"] == "NR-AR"
    assert data.metadata["n_targets_available"] > 1


@pytest.mark.skipif("tox21" not in available_datasets(), reason="tox21 not prepared")
def test_masked_multitask_splits_are_rejected_not_misaligned():
    """Splits index the full CSV, so they cannot be reused after row dropping."""
    with pytest.raises(DatasetUnavailableError, match="cannot be reused|masked"):
        load_split("tox21", seed=1, task="NR-AR")


@pytest.mark.skipif("tox21" not in available_datasets(), reason="tox21 not prepared")
def test_unknown_task_is_rejected():
    with pytest.raises(DatasetUnavailableError, match="not declared"):
        load_dataset("tox21", task="NOT_AN_ASSAY")


# --------------------------------------------------------------------------
# Regression tests for the silent-fallback bug.
#
# run_recommendation_ablation used to import a non-existent module, swallow the
# ImportError, and fall back to the synthetic stress dataset while still
# reporting source="molecular". The tests below pin the corrected behaviour.
# --------------------------------------------------------------------------


def test_benchmark_loader_raises_instead_of_substituting_synthetic():
    from benchmarks.run_recommendation_ablation import _load_descriptor_dataset

    with pytest.raises(DatasetUnavailableError):
        _load_descriptor_dataset("not_a_real_dataset", seed=1)


def test_benchmark_rejects_dataset_outside_the_requested_set():
    from benchmarks.run_recommendation_ablation import run_ablation_on_dataset

    with pytest.raises(ValueError, match="unknown dataset"):
        run_ablation_on_dataset(
            "esol",
            datasets=("synthetic",),
            seed=1,
            k=3,
            n_splits=3,
            n_resamples=100,
        )


def test_real_dataset_run_is_labelled_molecular_with_real_shapes():
    """A real run must report molecular provenance and real descriptor counts."""
    from benchmarks.run_recommendation_ablation import run_ablation_on_dataset

    name = _smallest_available_single_task()
    run = run_ablation_on_dataset(
        name,
        datasets=(name,),
        seed=1,
        k=3,
        n_splits=3,
        n_resamples=200,
    )

    assert run["source"] == "molecular", "real data must not be mislabelled"
    assert run["dataset"] == name

    expected = load_dataset(name)
    assert run["n_samples"] == expected.n_samples
    # The synthetic stress dataset has exactly 10 columns; real descriptor
    # matrices have ~180-210. This is the assertion that would have failed
    # before the loader existed.
    assert run["n_features"] == expected.n_features
    assert run["n_features"] > 50


def test_synthetic_run_is_labelled_synthetic():
    from benchmarks.run_recommendation_ablation import run_ablation_on_dataset

    run = run_ablation_on_dataset(
        "synthetic",
        datasets=("synthetic",),
        seed=1,
        k=3,
        n_splits=3,
        n_resamples=200,
    )
    assert run["source"] == "synthetic"
    assert run["task"] is None


def test_aggregation_retains_auditable_per_run_records():
    """The artifact must keep raw runs so provenance can be checked later."""
    from benchmarks.run_recommendation_ablation import _aggregate

    runs = [
        {
            "dataset": "esol",
            "source": "molecular",
            "task": None,
            "seed": seed,
            "n_features": 191,
            "arms": {
                "scorecard": {
                    "mean_test_score": 0.5,
                    "mean_generalisation_gap": 0.01,
                    "paired_vs_baseline": None,
                }
            },
        }
        for seed in (1, 2)
    ]
    agg = _aggregate(runs)

    assert agg["n_seeds"] == 2
    assert agg["sources"] == ["molecular"]
    assert len(agg["runs"]) == 2
    assert [r["seed"] for r in agg["runs"]] == [1, 2]
    assert {r["source"] for r in agg["runs"]} == {"molecular"}


# --------------------------------------------------------------------------- #
# Sweep checkpointing and worker thread pinning
# --------------------------------------------------------------------------- #
def test_checkpoint_round_trips_and_skips_truncated_lines(tmp_path):
    """A hard kill leaves a partial final line; that must not lose the rest."""
    from benchmarks.run_recommendation_ablation import _read_checkpoint, _write_checkpoint

    path = tmp_path / "cells.jsonl"
    cells = [
        {"dataset": "esol", "seed": 1, "n_features": 191},
        {"dataset": "esol", "seed": 2, "n_features": 191},
    ]
    _write_checkpoint(path, cells)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"dataset": "esol", "seed": 3, "n_feat')

    done = _read_checkpoint(path)

    assert set(done) == {("esol", 1), ("esol", 2)}
    assert done[("esol", 2)]["n_features"] == 191


def test_checkpoint_resume_skips_completed_cells_and_matches_serial(
    tmp_path, monkeypatch
):
    """Resuming must not recompute finished cells, and must match a cold run."""
    from benchmarks import run_recommendation_ablation as runner

    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(runner, "_CHECKPOINT_NAME", "cells.jsonl")

    kwargs = dict(datasets=("synthetic",), seeds=[1, 2], k=3, n_splits=3,
                  n_resamples=100, n_jobs=2, write=False)
    cold = runner.run_recommendation_ablation(**kwargs)

    calls: list[tuple[str, int]] = []
    real_run_one = runner._run_one

    def _counting_run_one(dataset, seed, **kwargs):
        calls.append((dataset, seed))
        return real_run_one(dataset, seed, **kwargs)

    monkeypatch.setattr(runner, "_run_one", _counting_run_one)
    warm = runner.run_recommendation_ablation(**kwargs)

    assert calls == [], "resume recomputed cells that were already checkpointed"
    assert warm["per_dataset"]["synthetic"]["n_seeds"] == 2
    assert (
        warm["per_dataset"]["synthetic"]["arms"]["siren_full"]["mean_test_score"]
        == cold["per_dataset"]["synthetic"]["arms"]["siren_full"]["mean_test_score"]
    )


def test_worker_pins_blas_threads_to_one(monkeypatch):
    """Workers are the unit of parallelism; BLAS must not multiply them by cores."""
    from benchmarks.run_recommendation_ablation import _pin_blas_threads

    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.delenv(name, raising=False)

    _pin_blas_threads(1)

    import os

    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        assert os.environ[name] == "1"


# --------------------------------------------------------------------------- #
# Descriptor cache
# --------------------------------------------------------------------------- #
def test_descriptor_cache_round_trips_and_is_faster(tmp_path, monkeypatch):
    """A warm load must return the identical matrix, not merely a same-shape one.

    The cache previously stored column names as an object-dtype array, which
    cannot be read back with ``allow_pickle=False``. The resulting exception was
    swallowed by a bare ``except``, so every "cached" load silently recomputed
    the descriptors. These assertions pin both the equality and the speedup.
    """
    import time

    from benchmarks import datasets as module

    monkeypatch.setattr(module, "DATA_DIR", tmp_path)
    name = _smallest_available_single_task()

    t0 = time.perf_counter()
    cold = load_dataset(name, use_cache=False)
    cold_seconds = time.perf_counter() - t0

    # Populate the cache, then time a genuine hit. A ``use_cache=False`` call
    # never writes, so without this warm-up the "warm" timing would also be a
    # recompute and the test would be vacuous.
    load_dataset(name, use_cache=True)

    t0 = time.perf_counter()
    warm = load_dataset(name, use_cache=True)
    warm_seconds = time.perf_counter() - t0

    assert np.array_equal(cold.features, warm.features)
    assert cold.feature_names == warm.feature_names
    assert cold.target.tolist() == warm.target.tolist()
    # A recompute would take as long as the cold path; require a real saving.
    assert warm_seconds < cold_seconds, (
        f"warm load ({warm_seconds:.3f}s) did not beat cold ({cold_seconds:.3f}s); "
        "the cache is not being read"
    )


def test_descriptor_cache_names_file_is_plain_json(tmp_path, monkeypatch):
    """Column names must live in readable JSON, never inside the ``.npz``.

    Note the cache stores the *raw* descriptor output, before
    ``drop_constant`` removes zero-variance columns. So the names file is a
    superset of ``Dataset.feature_names``, and the npz has more columns than the
    returned matrix. That is deliberate: the cache must stay valid for both
    ``drop_constant`` settings.
    """
    import json

    from benchmarks import datasets as module

    monkeypatch.setattr(module, "DATA_DIR", tmp_path)
    name = _smallest_available_single_task()
    data = load_dataset(name, use_cache=True)

    name_files = sorted(tmp_path.glob(".descriptor_cache/*.names.json"))
    assert len(name_files) == 1
    with name_files[0].open("r", encoding="utf-8") as handle:
        names = json.load(handle)

    # Every surviving feature must be present in the cached name list, in order.
    survivors = [n for n in names if n in set(data.feature_names)]
    assert survivors == data.feature_names

    # The npz itself must load without pickle and hold no object array.
    npz_files = sorted(tmp_path.glob(".descriptor_cache/*.npz"))
    assert len(npz_files) == 1
    with np.load(npz_files[0], allow_pickle=False) as cached:
        assert list(cached.keys()) == ["matrix"]
        assert cached["matrix"].shape == (data.n_samples, len(names))


def test_descriptor_cache_is_independent_of_drop_constant(tmp_path, monkeypatch):
    """One cached matrix must serve both ``drop_constant`` settings."""
    from benchmarks import datasets as module

    monkeypatch.setattr(module, "DATA_DIR", tmp_path)
    name = _smallest_available_single_task()

    dropped = load_dataset(name, drop_constant=True, use_cache=True)
    kept = load_dataset(name, drop_constant=False, use_cache=True)

    # Same cache file, so the second call must not have recomputed anything.
    assert len(list(tmp_path.glob(".descriptor_cache/*.npz"))) == 1
    assert kept.n_features >= dropped.n_features
    # The surviving columns of the full matrix must match the dropped version.
    keep_mask = [n in set(dropped.feature_names) for n in kept.feature_names]
    assert list(np.asarray(kept.feature_names)[keep_mask]) == dropped.feature_names
    assert np.array_equal(kept.features[:, keep_mask], dropped.features)


def test_descriptor_cache_fingerprint_tracks_molecule_order_and_content():
    """A reordered or edited molecule list must miss the cache.

    The cached matrix is row-aligned with the target vector, so serving a stale
    matrix for a reordered SMILES list would silently misalign features from
    labels -- the worst possible failure for a supervised benchmark.
    """
    from benchmarks.datasets import _smiles_fingerprint

    base = _smiles_fingerprint(["CCO", "CCN", "c1ccccc1"])
    assert _smiles_fingerprint(["CCN", "CCO", "c1ccccc1"]) != base, "order ignored"
    assert _smiles_fingerprint(["CCO", "CCC", "c1ccccc1"]) != base, "content ignored"
    assert _smiles_fingerprint(["CCO", "CCN", "c1ccccc1"]) == base, "not deterministic"


def test_corrupt_descriptor_cache_is_recomputed_not_raised(tmp_path, monkeypatch):
    """A truncated cache file must degrade to recomputation, never to a crash."""
    from benchmarks import datasets as module

    monkeypatch.setattr(module, "DATA_DIR", tmp_path)
    name = _smallest_available_single_task()
    expected = load_dataset(name, use_cache=True)

    for npz in tmp_path.glob(".descriptor_cache/*.npz"):
        npz.write_bytes(b"not a real npz file")

    recovered = load_dataset(name, use_cache=True)
    assert np.array_equal(recovered.features, expected.features)
    assert recovered.feature_names == expected.feature_names




