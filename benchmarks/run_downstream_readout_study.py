"""E-DOWNSTREAM: does the selected subspace improve a real downstream model?

E-REC evaluates a selector with a single ordinary-least-squares readout and an
internal k-fold. That protocol isolates selection quality, but it cannot answer
the question a practitioner asks: *if I train the model I actually use on these
features, do I get a better model?*

This study closes that gap. For every dataset and scaffold split it

1. selects ``k`` descriptors on the development rows (train + validation) with
   each selector arm, including an ``all_features`` reference;
2. fits two readouts on identical rows -- a linear model (ridge / logistic) and
   a nonlinear ensemble (random forest);
3. evaluates on the held-out scaffold split.

The scaffold split is the leakage boundary: the test molecules share no Bemis-
Murcko scaffold with the development molecules, which is the realistic
generalisation setting and the one used in Paper 1.

The nonlinear readout is the key comparison. A descriptor that only matters in
combination with another cannot help a linear model even when it is selected
correctly; a random forest can exploit that interaction. If SIREN's conditional
and synergy terms are doing real work, the gap should appear there.

Usage
-----
    python benchmarks/run_downstream_readout_study.py --seeds 1 2 3 4 5
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chemistry_framework.feature_analysis.recommend import (
    recommend_features,
    scorecard_ranking,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

ARM_WEIGHTS: dict[str, dict[str, float] | None] = {
    "all_features": None,
    "scorecard": None,
    "siren_full": {"relevance": 1.0, "synergy": 0.5, "stability": 0.5, "redundancy": 0.0},
    "siren_relevance_only": {"relevance": 1.0, "synergy": 0.0, "stability": 0.0, "redundancy": 0.0},
    "siren_no_synergy": {"relevance": 1.0, "synergy": 0.0, "stability": 0.5, "redundancy": 0.0},
}


def _readouts(is_classification: bool, seed: int) -> dict[str, Any]:
    """Return the linear and nonlinear readouts for one endpoint type."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if is_classification:
        return {
            "linear": make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, C=1.0, random_state=seed),
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=2,
                n_jobs=1,
                random_state=seed,
            ),
        }
    return {
        "linear": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "random_forest": RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            n_jobs=1,
            random_state=seed,
        ),
    }


def _evaluate(
    model: Any,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    is_classification: bool,
) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    )

    model.fit(X_fit, y_fit)
    prediction = model.predict(X_test)
    if is_classification:
        try:
            probability = model.predict_proba(X_test)[:, 1]
            auroc = float(roc_auc_score(y_test, probability))
        except Exception:
            auroc = float("nan")
        return {
            "auroc": auroc,
            "accuracy": float(accuracy_score(y_test, prediction)),
        }
    return {
        "rmse": float(math.sqrt(mean_squared_error(y_test, prediction))),
        "mae": float(mean_absolute_error(y_test, prediction)),
        "r2": float(r2_score(y_test, prediction)),
    }


def _select(
    arm: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    names: Sequence[str],
    k: int,
    seed: int,
) -> list[int]:
    if arm == "all_features":
        return list(range(X_train.shape[1]))
    weights = ARM_WEIGHTS[arm]
    if weights is None:
        report = scorecard_ranking(X_train, y_train, feature_names=names, max_features=k)
    else:
        report = recommend_features(
            X_train, y_train, feature_names=names, k=k, weights=weights, seed=seed
        )
    return report.selected_indices()


def run(
    datasets: Sequence[str],
    seeds: Sequence[int],
    k: int = 5,
) -> dict[str, Any]:
    from benchmarks.datasets import load_split

    cells: list[dict[str, Any]] = []
    for dataset_name in datasets:
        features, target, split = load_split(dataset_name, seed=int(seeds[0]))
        is_classification = bool(np.unique(target).size <= 2)
        names = [f"column_{index}" for index in range(features.shape[1])]
        try:
            from benchmarks.datasets import load_dataset

            names = list(load_dataset(dataset_name).feature_names)
        except Exception:
            pass

        for seed in seeds:
            features, target, split = load_split(dataset_name, seed=int(seed))
            development = np.concatenate([split.train, split.val])
            test = split.test
            X_dev, y_dev = features[development], target[development]
            X_test, y_test = features[test], target[test]
            print(
                f"\n{dataset_name} seed={seed}: dev={X_dev.shape}, test={X_test.shape}",
                flush=True,
            )
            for arm in ARM_WEIGHTS:
                selected = _select(arm, X_dev, y_dev, names, k, seed)
                if not selected:
                    print(f"  {arm:22s} skipped (no features selected)", flush=True)
                    continue
                chosen_names = [names[index] for index in selected]
                for readout_name, model in _readouts(is_classification, seed).items():
                    metrics = _evaluate(
                        model,
                        X_dev[:, selected],
                        y_dev,
                        X_test[:, selected],
                        y_test,
                        is_classification,
                    )
                    cell = {
                        "dataset": dataset_name,
                        "seed": int(seed),
                        "arm": arm,
                        "readout": readout_name,
                        "k": k,
                        "n_selected": len(selected),
                        "selected": chosen_names,
                        "is_classification": is_classification,
                        "n_development": int(X_dev.shape[0]),
                        "n_test": int(X_test.shape[0]),
                        **metrics,
                    }
                    cells.append(cell)
                    headline = (
                        f"auroc={metrics['auroc']:.4f}"
                        if is_classification
                        else f"rmse={metrics['rmse']:.4f} r2={metrics['r2']:+.4f}"
                    )
                    print(f"  {arm:22s} {readout_name:14s} {headline}", flush=True)
    return {
        "experiment": "E-DOWNSTREAM",
        "description": (
            "Downstream model quality on scaffold-held-out molecules for "
            "descriptors selected by each arm"
        ),
        "k": k,
        "seeds": list(seeds),
        "datasets": list(datasets),
        "arms": list(ARM_WEIGHTS),
        "cells": cells,
    }


def summarise(payload: dict[str, Any]) -> dict[str, Any]:
    """Mean/std per (dataset, arm, readout), plus paired deltas vs scorecard.

    The headline metric is endpoint-specific: AUROC for classification and RMSE
    for regression. A single combined metric across both endpoint types would be
    meaningless, so every summary is keyed by dataset and carries its own metric
    and direction.
    """
    from collections import defaultdict

    metric_by_dataset: dict[str, str] = {}
    for cell in payload["cells"]:
        metric_by_dataset[cell["dataset"]] = (
            "auroc" if cell["is_classification"] else "rmse"
        )

    buckets: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for cell in payload["cells"]:
        metric = metric_by_dataset[cell["dataset"]]
        value = cell.get(metric)
        if value is not None and np.isfinite(value):
            buckets[(cell["dataset"], cell["arm"], cell["readout"])].append(
                float(value)
            )

    per_dataset: dict[str, Any] = {}
    for (dataset, arm, readout), values in sorted(buckets.items()):
        array = np.asarray(values, dtype=float)
        per_dataset.setdefault(dataset, {}).setdefault(readout, {})[arm] = {
            "mean": float(array.mean()),
            "std": float(array.std()),
            "n": int(array.size),
        }

    # Paired deltas against the scorecard on identical seeds.
    by_key: dict[tuple[str, str, str, int], float] = {}
    for cell in payload["cells"]:
        metric = metric_by_dataset[cell["dataset"]]
        value = cell.get(metric)
        if value is not None and np.isfinite(value):
            by_key[
                (cell["dataset"], cell["readout"], cell["arm"], cell["seed"])
            ] = float(value)

    deltas: dict[str, Any] = {}
    for dataset in sorted({key[0] for key in by_key}):
        metric = metric_by_dataset[dataset]
        higher_is_better = metric == "auroc"
        for readout in sorted({key[1] for key in by_key}):
            baseline = {
                key[3]: value
                for key, value in by_key.items()
                if key[0] == dataset and key[1] == readout and key[2] == "scorecard"
            }
            if not baseline:
                continue
            for arm in sorted({key[2] for key in by_key}):
                if arm == "scorecard":
                    continue
                paired = [
                    by_key[(dataset, readout, arm, seed)] - baseline[seed]
                    for seed in sorted(baseline)
                    if (dataset, readout, arm, seed) in by_key
                ]
                if not paired:
                    continue
                array = np.asarray(paired, dtype=float)
                deltas.setdefault(dataset, {}).setdefault(readout, {})[arm] = {
                    "metric": metric,
                    "higher_is_better": higher_is_better,
                    "mean_delta": float(array.mean()),
                    "std_delta": float(array.std()),
                    "n_paired": int(array.size),
                    "wins": int(
                        (array > 0).sum() if higher_is_better else (array < 0).sum()
                    ),
                }

    return {
        "metric_by_dataset": metric_by_dataset,
        "per_dataset": per_dataset,
        "paired_delta_vs_scorecard": deltas,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="E-DOWNSTREAM readout study")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["esol", "freesolv", "lipophilicity", "bbbp", "bace"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    payload = run(args.datasets, args.seeds, k=args.k)
    payload["summary"] = summarise(payload)

    if not args.no_write:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / "experiment_e_downstream_readout.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
