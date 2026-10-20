"""E-LEAK: how much optimism does selection leakage buy?

Feature-selection papers routinely report a score obtained by

1. ranking/selecting features on the *full* dataset, then
2. cross-validating or hold-out testing a model built from those features.

Step 1 already saw the held-out rows -- including their labels -- so the
reported score is not an estimate of generalisation. This study measures the
size of that optimism directly and compares it across selectors.

Three protocols are run on identical scaffold splits:

``nested``
    Selection uses development rows only (train + validation); the model is
    fitted on development rows and scored on the held-out scaffolds.
``leaky``
    Selection uses every row (development + test); the model is fitted on
    development rows and scored on the same held-out scaffolds. The only
    difference from ``nested`` is that the selector saw the test rows.
``circular``
    Selection uses development rows, then the model is fitted and scored on the
    *same* development rows. This is the in-sample ceiling and quantifies how
    much of a reported score can be pure fit.

The honest estimate is ``nested``. ``leaky - nested`` is the optimism induced by
the selection step alone; ``circular - nested`` is the optimism of reporting an
in-sample score.

Usage
-----
    python benchmarks/run_selection_leakage_study.py --seeds 1 2 3 4 5
"""
from __future__ import annotations

import argparse
import json
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
    "scorecard": None,
    "siren_full": {"relevance": 1.0, "synergy": 0.5, "stability": 0.5, "redundancy": 0.0},
}


def _select(
    arm: str,
    X: np.ndarray,
    y: np.ndarray,
    names: Sequence[str],
    k: int,
    seed: int,
) -> list[int]:
    weights = ARM_WEIGHTS[arm]
    if weights is None:
        report = scorecard_ranking(X, y, feature_names=names, max_features=k)
    else:
        report = recommend_features(
            X, y, feature_names=names, k=k, weights=weights, seed=seed
        )
    return report.selected_indices()


def _spearman_score(
    X_fit: np.ndarray, y_fit: np.ndarray, X_eval: np.ndarray, y_eval: np.ndarray
) -> float:
    """OLS readout scored by Spearman association, matching the E-REC protocol."""
    from scipy import stats

    if X_fit.size == 0 or X_eval.shape[1] == 0:
        return 0.0
    design = np.column_stack([X_fit, np.ones(X_fit.shape[0])])
    coefficients, *_ = np.linalg.lstsq(design, y_fit, rcond=None)
    prediction = np.column_stack([X_eval, np.ones(X_eval.shape[0])]) @ coefficients
    rho, _ = stats.spearmanr(prediction, y_eval)
    return float(rho) if np.isfinite(rho) else 0.0


def run(datasets: Sequence[str], seeds: Sequence[int], k: int = 5) -> dict[str, Any]:
    from benchmarks.datasets import load_dataset, load_split

    cells: list[dict[str, Any]] = []
    for dataset_name in datasets:
        try:
            names = list(load_dataset(dataset_name).feature_names)
        except Exception:
            names = []
        for seed in seeds:
            features, target, split = load_split(dataset_name, seed=int(seed))
            if not names:
                names = [f"column_{index}" for index in range(features.shape[1])]
            development = np.concatenate([split.train, split.val])
            test = split.test
            X_dev, y_dev = features[development], target[development]
            X_test, y_test = features[test], target[test]
            X_all, y_all = features, target

            print(f"\n{dataset_name} seed={seed}", flush=True)
            for arm in ARM_WEIGHTS:
                selected_nested = _select(arm, X_dev, y_dev, names, k, seed)
                selected_leaky = _select(arm, X_all, y_all, names, k, seed)

                nested = _spearman_score(
                    X_dev[:, selected_nested], y_dev, X_test[:, selected_nested], y_test
                )
                leaky = _spearman_score(
                    X_dev[:, selected_leaky], y_dev, X_test[:, selected_leaky], y_test
                )
                circular = _spearman_score(
                    X_dev[:, selected_nested], y_dev, X_dev[:, selected_nested], y_dev
                )
                shared = len(set(selected_nested) & set(selected_leaky))
                cells.append(
                    {
                        "dataset": dataset_name,
                        "seed": int(seed),
                        "arm": arm,
                        "k": k,
                        "nested": nested,
                        "leaky": leaky,
                        "circular": circular,
                        "leaky_minus_nested": leaky - nested,
                        "circular_minus_nested": circular - nested,
                        "overlap_nested_vs_leaky": shared,
                        "jaccard_nested_vs_leaky": (
                            shared / len(set(selected_nested) | set(selected_leaky))
                            if set(selected_nested) | set(selected_leaky)
                            else 0.0
                        ),
                    }
                )
                print(
                    f"  {arm:18s} nested={nested:+.4f} leaky={leaky:+.4f} "
                    f"circular={circular:+.4f} overlap={shared}/{k}",
                    flush=True,
                )
    return {
        "experiment": "E-LEAK",
        "description": (
            "Optimism induced by selecting features before the evaluation split"
        ),
        "k": k,
        "seeds": list(seeds),
        "datasets": list(datasets),
        "protocols": ["nested", "leaky", "circular"],
        "cells": cells,
    }


def summarise(payload: dict[str, Any]) -> dict[str, Any]:
    from collections import defaultdict

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for cell in payload["cells"]:
        buckets[(cell["dataset"], cell["arm"])].append(cell)

    summary: dict[str, Any] = {}
    for (dataset, arm), cells in sorted(buckets.items()):
        def agg(key: str) -> dict[str, float]:
            values = np.asarray([cell[key] for cell in cells], dtype=float)
            values = values[np.isfinite(values)]
            return {
                "mean": float(values.mean()) if values.size else float("nan"),
                "std": float(values.std()) if values.size else float("nan"),
            }

        summary.setdefault(dataset, {})[arm] = {
            "n_seeds": len(cells),
            "nested": agg("nested"),
            "leaky": agg("leaky"),
            "circular": agg("circular"),
            "leaky_minus_nested": agg("leaky_minus_nested"),
            "circular_minus_nested": agg("circular_minus_nested"),
            "jaccard_nested_vs_leaky": agg("jaccard_nested_vs_leaky"),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="E-LEAK selection-leakage study")
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
        path = RESULTS_DIR / "experiment_e_leakage.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved: {path}")

    print("\nLeakage summary (mean over seeds):")
    for dataset, arms in payload["summary"].items():
        for arm, stats in arms.items():
            print(
                f"  {dataset:16s} {arm:18s} "
                f"nested={stats['nested']['mean']:+.3f} "
                f"leaky={stats['leaky']['mean']:+.3f} "
                f"gap={stats['leaky_minus_nested']['mean']:+.3f} "
                f"circular_gap={stats['circular_minus_nested']['mean']:+.3f}"
            )


if __name__ == "__main__":
    main()
