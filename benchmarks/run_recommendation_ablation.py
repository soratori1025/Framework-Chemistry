"""Experiment E-REC: SIREN recommendation ablation on real molecular datasets.

This benchmark is the empirical evidence for the framework's recommendation
contribution. It answers a single, falsifiable question:

    Does a conditional, synergy- and stability-aware selector choose features
    that generalise better -- under a leakage-free protocol -- than the legacy
    marginal Spearman scorecard?

Design
------
* **Data.** Real molecular descriptors from the framework's canonical datasets
  (ESOL / FreeSolv / Lipophilicity / BBBP). The dataset loader is imported
  lazily so the module stays importable when data is unavailable.
* **Protocol.** :func:`nested_selection_evaluation` re-runs the whole selection
  inside each training fold, so no held-out row ever influences the choice.
* **Comparison.** SIREN vs. the legacy scorecard on *identical* folds, with a
  paired bootstrap confidence interval and p-value on the fold-wise difference.
* **Ablation.** To isolate which term earns the gain, SIREN is re-run with each
  objective term disabled in turn (relevance-only, no-synergy, no-stability,
  no-redundancy penalty).
* **Honesty.** All seeds, weights and fold assignments are recorded so results
  are reproducible, and the module reports a null result as readily as a
  positive one -- the paired bootstrap makes that decision data-driven rather
  than narrative.

Run::

    python benchmarks/run_recommendation_ablation.py --datasets esol --seeds 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chemistry_framework.feature_analysis.protocol import (
    nested_selection_evaluation,
    paired_bootstrap_test,
)
from chemistry_framework.feature_analysis.recommend import (
    DEFAULT_MIN_SUPPORT,
    DEFAULT_WEIGHTS,
    recommend_features,
    scorecard_ranking,
)

RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"

# Per-cell checkpoint for resumable sweeps. JSON Lines so a cell can be appended
# the instant it finishes, without rewriting the whole file.
_CHECKPOINT_NAME = "experiment_e_rec_recommendation_ablation.checkpoint.jsonl"


def _read_checkpoint(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Load completed (dataset, seed) cells from a JSONL checkpoint.

    A truncated final line -- which is what a hard kill leaves behind -- is
    skipped rather than treated as corruption of the whole file, so an
    interrupted sweep loses at most the single cell that was mid-write.
    """
    done: dict[tuple[str, int], dict[str, Any]] = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            cell = json.loads(line)
        except json.JSONDecodeError:
            continue
        done[(cell["dataset"], int(cell["seed"]))] = cell
    return done


def _append_checkpoint(path: Path, cell: dict[str, Any]) -> None:
    """Append one completed cell to the JSONL checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(cell) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_checkpoint(path: Path, cells: Sequence[dict[str, Any]]) -> None:
    """Rewrite the checkpoint from a complete set of cells."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for cell in cells:
            handle.write(json.dumps(cell) + "\n")

# Ablation arms: each name maps to the objective WEIGHTS it uses, so every arm
# is the same algorithm with one term ablated. This keeps the comparison
# apples-to-apples instead of comparing unrelated methods.
#
# ``siren_full`` is the *current* production default, which now sets
# ``redundancy = 0.0`` (see ``DEFAULT_WEIGHTS``). ``siren_legacy`` restores the
# pre-2026 default (``redundancy = 1.0``) so the old numbers stay reproducible
# and the effect of the change can be reported directly. Note that
# ``siren_no_redundancy`` and ``siren_full`` are therefore identical in this
# configuration; both arms are retained so that archived checkpoints written
# under the old default continue to aggregate correctly.
LEGACY_WEIGHTS: dict[str, float] = {**DEFAULT_WEIGHTS, "redundancy": 1.0}

# Arms differ in *two* independent ways: the objective weights and whether the
# support screen is active. Keeping them in one record means new mechanism arms
# (e.g. disabling the screen) need no change to the grid code below.
ABLATION_ARMS: dict[str, dict[str, float]] = {
    "siren_full": dict(DEFAULT_WEIGHTS),
    "siren_legacy": dict(LEGACY_WEIGHTS),
    "siren_no_synergy": {**DEFAULT_WEIGHTS, "synergy": 0.0},
    "siren_no_stability": {**DEFAULT_WEIGHTS, "stability": 0.0},
    "siren_no_redundancy": {**DEFAULT_WEIGHTS, "redundancy": 0.0},
    "siren_relevance_only": {
        "relevance": 1.0,
        "synergy": 0.0,
        "stability": 0.0,
        "redundancy": 0.0,
    },
}

# Arms that switch off the support screen, i.e. the pre-screen selector. These
# isolate the screen's effect from the weight changes above.
NO_SCREEN_ARMS: tuple[str, ...] = ("siren_no_support_screen",)



def _load_descriptor_dataset(name: str, seed: int, task: str | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a real molecular descriptor matrix through the framework pipeline.

    This deliberately *raises* when a dataset cannot be loaded. The previous
    implementation swallowed the import error and returned ``None``, which made
    the caller fall back to the synthetic stress dataset while still labelling
    the run as ``source="molecular"``. That produced entirely synthetic results
    attributed to real molecules, so the failure must be loud and attributed.

    The imports are still guarded, but a missing dependency is reported as a
    :class:`DatasetUnavailableError` rather than silently degrading.
    """
    from benchmarks.datasets import DatasetUnavailableError, load_dataset, load_split

    try:
        features, target, _split = load_split(name, seed=seed, task=task)
        dataset = load_dataset(name, task=task)
    except DatasetUnavailableError as exc:
        raise DatasetUnavailableError(
            f"cannot load real dataset {name!r} for the recommendation benchmark: {exc}"
        ) from exc
    feature_names = list(dataset.feature_names)

    features = np.asarray(features, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if features.ndim != 2 or features.shape[0] != target.size:
        raise DatasetUnavailableError(
            f"dataset {name!r} produced an inconsistent matrix/target shape: "
            f"{features.shape} vs {target.shape}"
        )
    if features.shape[1] < 3 or features.shape[0] < 20:
        raise DatasetUnavailableError(
            f"dataset {name!r} is too small for selection benchmarking: "
            f"{features.shape}"
        )
    return features, target, feature_names


def synthetic_stress_dataset(seed: int, n_samples: int = 400) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build a dataset with the exact pathologies the scorecard cannot handle.

    Construction (all columns are *planted* so the ground truth is known):

    ``x0``  marginal driver                       -- must be selected by anyone.
    ``x1``  near-duplicate of ``x0``              -- pure redundancy.
    ``x2``  suppressor: correlates with ``x0`` noise but carries independent
            signal once ``x0`` is held fixed      -- wins only conditionally.
    ``x3``  interaction partner of ``x4``         -- zero marginal information.
    ``x4``  interaction partner of ``x3``         -- zero marginal information.
    ``x5..x9`` pure noise                        -- must never be selected.

    A selector with only marginal relevance can recover ``x0`` but must fail on
    ``x2`` (suppressor) and on the ``x3``/``x4`` pair (synergy). This is the
    cleanest possible falsification test for the SIREN claims.
    """
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n_samples)
    # Suppressor structure: the suppressor and the driver share a nuisance
    # component, but the suppressor retains a unique contribution to `latent`.
    nuisance = rng.normal(size=n_samples)
    driver = latent + 0.2 * rng.normal(size=n_samples)
    duplicate = driver + 0.01 * rng.normal(size=n_samples)
    suppressor = -0.7 * driver + 0.7 * latent + 0.3 * rng.normal(size=n_samples)
    # XOR-style interaction pair: individually uninformative, jointly decisive.
    member_a = rng.normal(size=n_samples)
    sign = np.where(latent > 0, 1.0, -1.0)
    member_b = sign * member_a
    noise = [rng.normal(size=n_samples) for _ in range(5)]

    features = np.column_stack(
        [driver, duplicate, suppressor, member_a, member_b, *noise]
    )
    target = latent + 0.3 * rng.normal(size=n_samples)
    names = [
        "driver",
        "duplicate",
        "suppressor",
        "interaction_a",
        "interaction_b",
        *(f"noise_{i}" for i in range(5)),
    ]
    return features, target, names


def _arm_selector(
    weights: dict[str, float], k: int, seed: int, min_support: int = DEFAULT_MIN_SUPPORT
):
    """Build a fold-aware selector for one ablation arm."""

    def _select(X_train: np.ndarray, y_train: np.ndarray, fold: int):
        return recommend_features(
            X_train,
            y_train,
            k=k,
            weights=weights,
            stability_resamples=16,
            min_support=min_support,
            seed=seed + fold,
        )

    return _select


def _baseline_selector(k: int):
    """Legacy marginal scorecard, used as the paired comparison baseline."""

    def _select(X_train: np.ndarray, y_train: np.ndarray, fold: int):
        return scorecard_ranking(X_train, y_train, max_features=k)

    return _select



def run_ablation_on_dataset(
    name: str,
    *,
    datasets: Sequence[str],
    seed: int,
    task: str | None = None,
    k: int = 5,
    n_splits: int = 5,
    n_resamples: int = 5000,
) -> dict[str, Any]:
    """Run all ablation arms plus the baseline for one repetition.

    ``name`` selects the protocol: ``"synthetic"`` uses the planted stress
    dataset, a name in ``datasets`` uses real molecules. A real dataset that
    fails to load raises instead of silently substituting synthetic data, so the
    recorded ``source`` field can always be trusted.
    """
    if name == "synthetic":
        features, target, feature_names = synthetic_stress_dataset(seed)
        source = "synthetic"
    else:
        if name not in datasets:
            raise ValueError(
                f"unknown dataset {name!r}; pass 'synthetic' or one of {list(datasets)}"
            )
        features, target, feature_names = _load_descriptor_dataset(name, seed, task)
        source = "molecular"

    arms: dict[str, Any] = {}
    baseline_scores: list[float] | None = None
    baseline_audit = None

    # Baseline first so every arm is compared on the same folds.
    baseline_audit, baseline_scores = nested_selection_evaluation(
        features,
        target,
        selector=_baseline_selector(k),
        feature_names=feature_names,
        n_splits=n_splits,
        seed=seed,
    )
    arms["scorecard"] = {
        "mean_test_score": baseline_audit.mean_test_score,
        "std_test_score": baseline_audit.std_test_score,
        "mean_generalisation_gap": baseline_audit.mean_generalisation_gap,
        "selection_frequency": baseline_audit.selection_frequency,
        "paired_vs_baseline": None,
    }

    for arm_name, weights in ABLATION_ARMS.items():
        audit, scores = nested_selection_evaluation(
            features,
            target,
            selector=_arm_selector(weights, k, seed),
            feature_names=feature_names,
            n_splits=n_splits,
            seed=seed,
        )
        interval = paired_bootstrap_test(
            scores, baseline_scores, n_resamples=n_resamples, seed=seed
        )
        arms[arm_name] = {
            "mean_test_score": audit.mean_test_score,
            "std_test_score": audit.std_test_score,
            "mean_generalisation_gap": audit.mean_generalisation_gap,
            "selection_frequency": audit.selection_frequency,
            "paired_vs_baseline": interval.to_dict(),
        }

    # The support-screen ablation uses the *current* default weights so that the
    # only difference from ``siren_full`` is whether the screen is active.
    for arm_name in NO_SCREEN_ARMS:
        audit, scores = nested_selection_evaluation(
            features,
            target,
            selector=_arm_selector(dict(DEFAULT_WEIGHTS), k, seed, min_support=0),
            feature_names=feature_names,
            n_splits=n_splits,
            seed=seed,
        )
        interval = paired_bootstrap_test(
            scores, baseline_scores, n_resamples=n_resamples, seed=seed
        )
        arms[arm_name] = {
            "mean_test_score": audit.mean_test_score,
            "std_test_score": audit.std_test_score,
            "mean_generalisation_gap": audit.mean_generalisation_gap,
            "selection_frequency": audit.selection_frequency,
            "paired_vs_baseline": interval.to_dict(),
        }

    return {
        "dataset": name,
        "source": source,
        "task": task,
        "seed": seed,
        "k": k,
        "n_splits": n_splits,
        "n_samples": int(features.shape[0]),
        "n_features": int(features.shape[1]),
        "feature_names": feature_names,
        "arms": arms,
    }


def _aggregate(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Average arm statistics across seeds and count significant wins.

    Arms are unioned across runs rather than taken from the first run's keys.
    Checkpoints written under a different ``ABLATION_ARMS`` -- e.g. before
    ``siren_legacy`` existed, or before ``siren_full`` changed default -- are
    still aggregated, and a run missing an arm contributes nothing to that arm
    instead of raising ``KeyError``.
    """
    if not runs:
        return {}
    arm_names: list[str] = []
    for run in runs:
        for arm in run["arms"]:
            if arm not in arm_names:
                arm_names.append(arm)
    summary: dict[str, Any] = {}
    for arm in arm_names:
        present = [run["arms"][arm] for run in runs if arm in run["arms"]]
        if not present:
            continue
        means = [entry["mean_test_score"] for entry in present]
        gaps = [entry["mean_generalisation_gap"] for entry in present]
        intervals = [
            entry["paired_vs_baseline"]
            for entry in present
            if entry["paired_vs_baseline"] is not None
        ]
        summary[arm] = {
            "mean_test_score": float(np.mean(means)),
            "std_across_seeds": float(np.std(means, ddof=0)),
            "mean_generalisation_gap": float(np.mean(gaps)),
            "mean_paired_difference_vs_baseline": (
                float(np.mean([item["mean_difference"] for item in intervals]))
                if intervals
                else None
            ),
            "significant_wins_vs_baseline": int(
                sum(
                    1
                    for item in intervals
                    if item["p_value"] < 0.05 and item["ci_low"] > 0.0
                )
            ),
            "n_seed_comparisons": len(intervals),
            "seeds_present": len(present),
        }
    return {
        "arms": summary,
        "n_seeds": len(runs),
        "datasets": sorted({run["dataset"] for run in runs}),
        "sources": sorted({run["source"] for run in runs}),
        # Raw per-seed records are retained so the artifact stays independently
        # auditable: the silent-fallback bug this benchmark was fixed for could
        # only be detected because a per-run record disagreed with its label.
        "runs": list(runs),
    }


def _pin_blas_threads(threads: int = 1) -> None:
    """Pin BLAS/OpenMP thread pools to ``threads`` per process.

    The (dataset x seed) grid runs one cell per worker process, so each worker
    is already the unit of parallelism. Left alone, numpy's BLAS picks up every
    core in the machine, and ``n_jobs`` workers then spawn
    ``n_jobs x n_cores`` threads between them. Measured on hiv (41127 x 207),
    that oversubscription is catastrophic: the marginal scorecard took 0.70 s
    for a fold with one thread but had not finished after five minutes with the
    default pool. Pinning to one thread per worker is what makes the parallel
    sweep use the cores instead of fighting over them.

    The environment is only touched once per process, before the numeric work
    starts, because these variables are read at BLAS initialisation time.
    """
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = str(threads)


def _run_one(
    dataset: str,
    seed: int,
    *,
    datasets: Sequence[str],
    task: str | None,
    k: int,
    n_splits: int,
    n_resamples: int,
) -> dict[str, Any]:
    """Run one (dataset, seed) cell. Top-level so it is picklable by joblib."""
    _pin_blas_threads(1)
    return run_ablation_on_dataset(
        dataset,
        datasets=datasets,
        seed=seed,
        task=task,
        k=k,
        n_splits=n_splits,
        n_resamples=n_resamples,
    )


def run_recommendation_ablation(
    datasets: Sequence[str] = ("synthetic", "esol"),
    *,
    seeds: Sequence[int] = (1, 2, 3),
    task: str | None = None,
    k: int = 5,
    n_splits: int = 5,
    n_resamples: int = 5000,
    n_jobs: int = 1,
    write: bool = True,
) -> dict[str, Any]:
    """Run Experiment E-REC across datasets and seeds and summarise it.

    Args:
        task: Endpoint name for datasets with masked multi-task labels
            (e.g. ``"NR-AR"`` for tox21). Required for such datasets; ignored
            otherwise.
        n_jobs: Worker processes for the (dataset x seed) grid. Each cell is
            independent -- it loads its own data and uses its own RNG seed -- so
            parallelism changes wall-clock only, never results. ``1`` keeps the
            run in-process, which is what the tests use. ``-1`` uses every core.
    """
    started = time.time()
    print("=" * 75)
    print("EXPERIMENT E-REC: SIREN RECOMMENDATION ABLATION")
    print("=" * 75)

    grid = [(dataset, seed) for dataset in datasets for seed in seeds]

    # Completed cells are checkpointed as they finish. Without this, a sweep
    # that is interrupted -- or that stalls on one intractable cell, as the
    # first hiv run did after 69 of 80 cells -- throws away every cell it had
    # already computed, because the artifact is only assembled at the end.
    checkpoint = RESULTS_DIR / _CHECKPOINT_NAME
    done: dict[tuple[str, int], dict[str, Any]] = _read_checkpoint(checkpoint)
    pending = [(d, s) for (d, s) in grid if (d, s) not in done]
    if done:
        print(f"  resuming: {len(done)} cell(s) already checkpointed, "
              f"{len(pending)} to run", flush=True)

    def _record(cell: dict[str, Any]) -> dict[str, Any]:
        _append_checkpoint(checkpoint, cell)
        return cell

    cells: list[dict[str, Any]] = [done[key] for key in grid if key in done]
    if pending:
        if n_jobs == 1:
            for dataset, seed in pending:
                print(f"  running {dataset} seed={seed} ...", flush=True)
                cells.append(
                    _record(
                        _run_one(
                            dataset, seed, datasets=datasets, task=task, k=k,
                            n_splits=n_splits, n_resamples=n_resamples,
                        )
                    )
                )
        else:
            from joblib import Parallel, delayed

            # Deliberately not sharing a seed stream across workers: each cell
            # seeds its own RNG from its own seed, so the parallel and serial
            # runs produce bit-identical artifacts and results stay reproducible
            # either way.
            #
            # Cells are checkpointed as they complete rather than after the
            # whole batch: `Parallel` returns only once every worker is done,
            # so one intractable cell (hiv costs ~25 min per cell) would
            # otherwise hold every finished cell hostage behind it.
            # `return_as="generator_unordered"` matters: it yields each cell as
            # soon as it finishes, so the checkpoint advances even while the
            # slowest cell (hiv) is still running. The default return order
            # buffers results until every task is done, which would defeat the
            # per-cell checkpointing entirely.
            cells.extend(
                _record(cell)
                for cell in Parallel(
                    n_jobs=n_jobs, backend="loky", verbose=10,
                    return_as="generator_unordered",
                )(
                    delayed(_run_one)(
                        dataset, seed, datasets=datasets, task=task, k=k,
                        n_splits=n_splits, n_resamples=n_resamples,
                    )
                    for dataset, seed in pending
                )
            )
            _write_checkpoint(checkpoint, cells)

    all_runs: list[dict[str, Any]] = []
    per_dataset: dict[str, Any] = {}
    for dataset in datasets:
        runs = [cell for cell in cells if cell["dataset"] == dataset]
        all_runs.extend(runs)
        per_dataset[dataset] = _aggregate(runs)
        summary = per_dataset[dataset]["arms"]
        print(f"\n  {dataset.upper()} ({runs[0]['source']}, n={runs[0]['n_samples']}, "
              f"p={runs[0]['n_features']})")
        for arm, stats in summary.items():
            print(
                f"    {arm:22s} test={stats['mean_test_score']:+.4f} "
                f"gap={stats['mean_generalisation_gap']:+.4f} "
                f"sig_wins={stats['significant_wins_vs_baseline']}/"
                f"{stats['n_seed_comparisons']}"
            )

    payload = {
        "experiment": "E-REC",
        "description": "SIREN vs marginal scorecard under leakage-free nested selection",
        "k": k,
        "n_splits": n_splits,
        "n_resamples": n_resamples,
        "seeds": list(seeds),
        "n_jobs": n_jobs,
        "ablation_arms": ABLATION_ARMS,
        "per_dataset": per_dataset,
        "overall": _aggregate(all_runs),
        "runtime_minutes": (time.time() - started) / 60.0,
    }

    if write:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / "experiment_e_rec_recommendation_ablation.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="E-REC SIREN recommendation ablation")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["synthetic", "esol"],
        help="Dataset names; use 'synthetic' for the planted stress test.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument(
        "--task",
        default=None,
        help=(
            "Endpoint name for masked multi-task datasets, e.g. 'NR-AR' for "
            "tox21. Required for such datasets because dropping masked rows "
            "changes the sample set and must be an explicit choice."
        ),
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help=(
            "Worker processes over the (dataset x seed) grid. Each cell is "
            "independent and seeded separately, so results are identical to a "
            "serial run. Use -1 for all cores."
        ),
    )
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    run_recommendation_ablation(
        datasets=args.datasets,
        seeds=args.seeds,
        task=args.task,
        k=args.k,
        n_splits=args.splits,
        n_resamples=args.resamples,
        n_jobs=args.n_jobs,
        write=not args.no_write,
    )


if __name__ == "__main__":
    main()
