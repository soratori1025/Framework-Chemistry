"""E-SYNTH: ground-truth recovery of suppressor and synergy features.

The E-REC benchmark asks whether a selector generalises better. This experiment
asks a different, complementary question that a real dataset cannot answer
because the truth is unknown: *does the selector recover the features that are
provably informative?*

Each scenario constructs a feature matrix with known structure and a target that
depends on a designated subset. The legacy marginal scorecard and SIREN are then
run on identical data and their selected sets are scored against the planted
truth.

Scenarios
---------
``marginal``
    One strongly predictive column plus noise. A sanity check: both methods
    must find it.
``duplicate``
    The predictive column is duplicated. The duplicate carries no additional
    information, so a selector should spend only one of its ``k`` slots on the
    pair.
``suppression``
    Three noisy proxies of a latent signal. Each proxy alone is only moderately
    correlated with the target, and the proxies are strongly correlated with
    each other, so the scorecard's hard ``|r|`` prune discards the second and
    third proxies even though they carry independent information about the
    latent signal. This is classical multiple-regression suppression.
``xor``
    The target is the sign of a product of two independent columns. Neither
    column is marginally predictive; both are jointly predictive. A marginal
    scorecard cannot see this by construction.
``masked_synergy``
    One marginal driver plus an interaction partner that is pure noise on its
    own and informative only through the driver.
``noise``
    The target is independent of every column. Any selected feature is a false
    discovery; this calibrates the selector's false-positive behaviour.

Two families are run: fully synthetic Gaussian designs, and planted signals on
the *real* ESOL descriptor matrix, which preserves the correlation structure and
marginal distributions of 191 RDKit descriptors while still giving a known
truth.

Usage
-----
    python benchmarks/run_synthetic_recovery_study.py --seeds 1 2 3 4 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# Make ``chemistry_framework`` and ``benchmarks`` importable when this file is
# executed directly (``python benchmarks/<script>.py``) rather than as a module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chemistry_framework.feature_analysis.recommend import (
    recommend_features,
    scorecard_ranking,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Selector configurations compared on identical data. ``siren_full`` is the
# production default; the ablations isolate which objective term earns each
# recovery. Every arm uses the same k and the same random seed.
ARMS: dict[str, dict[str, float] | None] = {
    "scorecard": None,
    "siren_full": {"relevance": 1.0, "synergy": 0.5, "stability": 0.5, "redundancy": 0.0},
    "siren_relevance_only": {"relevance": 1.0, "synergy": 0.0, "stability": 0.0, "redundancy": 0.0},
    "siren_no_synergy": {"relevance": 1.0, "synergy": 0.0, "stability": 0.5, "redundancy": 0.0},
    "siren_no_stability": {"relevance": 1.0, "synergy": 0.5, "stability": 0.0, "redundancy": 0.0},
}


def _standardise(column: np.ndarray) -> np.ndarray:
    std = column.std()
    if std < 1e-12:
        return np.zeros_like(column)
    return (column - column.mean()) / std


def _gaussian_design(
    scenario: str, seed: int, n_samples: int = 600
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """Return (X, y, names, truth) for a synthetic Gaussian scenario."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n_samples)
    noise_columns = rng.normal(size=(n_samples, 4))

    if scenario == "marginal":
        signal = latent + 0.35 * rng.normal(size=n_samples)
        X = np.column_stack([signal, noise_columns])
        y = latent
        names = ["signal", "n0", "n1", "n2", "n3"]
        truth = {"informative": ["signal"], "redundant": [], "synergy": []}

    elif scenario == "duplicate":
        signal = latent + 0.35 * rng.normal(size=n_samples)
        X = np.column_stack([signal, signal.copy(), noise_columns[:, 0], noise_columns[:, 1]])
        y = latent
        names = ["signal", "signal_duplicate", "n0", "n1"]
        truth = {"informative": ["signal"], "redundant": ["signal_duplicate"], "synergy": []}

    elif scenario == "suppression":
        # Three noisy proxies of one latent signal. Pairwise |r| is ~0.89, above
        # the scorecard's 0.8 prune threshold, so it keeps one proxy and drops
        # the two complements that carry independent information about the
        # latent signal.
        proxies = [
            latent + 0.35 * rng.normal(size=n_samples) for _ in range(3)
        ]
        X = np.column_stack(proxies + [noise_columns[:, 0]])
        y = latent + 0.15 * rng.normal(size=n_samples)
        names = ["proxy_a", "proxy_b", "proxy_c", "n0"]
        truth = {"informative": ["proxy_a", "proxy_b", "proxy_c"], "redundant": [], "synergy": []}

    elif scenario == "xor":
        a = rng.normal(size=n_samples)
        b = rng.normal(size=n_samples)
        y = np.sign(a * b) + 0.1 * rng.normal(size=n_samples)
        # Enough noise columns that picking both interaction members by chance
        # is unlikely: with k=2 out of 12 columns the expected recall is ~0.03.
        extra = rng.normal(size=(n_samples, 10))
        X = np.column_stack([a, b, extra])
        names = ["xor_a", "xor_b"] + [f"n{index}" for index in range(10)]
        truth = {
            "informative": ["xor_a", "xor_b"],
            "redundant": [],
            "synergy": ["xor_a", "xor_b"],
            "k": 2,
        }

    elif scenario == "masked_synergy":
        driver = latent + 0.2 * rng.normal(size=n_samples)
        partner_source = rng.normal(size=n_samples)
        partner = partner_source * np.sign(latent)
        X = np.column_stack([driver, partner_source, partner, noise_columns[:, 0]])
        y = latent
        names = ["driver", "partner_source", "partner", "n0"]
        truth = {
            "informative": ["driver", "partner"],
            "redundant": [],
            "synergy": ["partner"],
        }

    elif scenario == "noise":
        X = np.column_stack([latent, noise_columns])
        y = rng.normal(size=n_samples)
        names = ["c0", "n0", "n1", "n2", "n3"]
        truth = {"informative": [], "redundant": [], "synergy": []}

    else:  # pragma: no cover - guarded by the CLI choices
        raise ValueError(f"unknown scenario: {scenario}")

    return X, y, names, truth


def _real_descriptor_design(
    scenario: str, seed: int, n_samples: int = 800
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """Plant a known signal on the real ESOL descriptor matrix.

    Rows are subsampled and the two chosen descriptor columns are standardised
    before the target is constructed, so the planted structure is exact in the
    standardised space while the remaining 189 columns keep their real
    correlation structure.
    """
    from benchmarks.datasets import load_dataset

    dataset = load_dataset("esol")
    features = np.asarray(dataset.features, dtype=float)
    names_all = list(dataset.feature_names)
    rng = np.random.default_rng(seed)
    rows = rng.choice(features.shape[0], size=min(n_samples, features.shape[0]), replace=False)
    matrix = features[rows]

    correlations = np.corrcoef(matrix, rowvar=False)
    correlations = np.nan_to_num(np.abs(correlations), nan=1.0)
    np.fill_diagonal(correlations, 0.0)

    if scenario == "real_synergy":
        # Synergy is only meaningful between descriptors that are not already
        # linearly redundant, so pick a near-orthogonal pair.
        viable = [
            (i, j)
            for i in range(matrix.shape[1])
            for j in range(i + 1, matrix.shape[1])
            if correlations[i, j] < 0.15
        ]
    elif scenario == "real_suppression":
        # Suppression needs the scorecard's prune to fire: the pair must be
        # strongly correlated, but below the duplicate screen's cut-off.
        viable = [
            (i, j)
            for i in range(matrix.shape[1])
            for j in range(i + 1, matrix.shape[1])
            if 0.80 <= correlations[i, j] < 0.99
        ]
    else:  # pragma: no cover
        raise ValueError(f"unknown scenario: {scenario}")

    if not viable:
        viable = [(0, 1)]
    first, second = viable[int(rng.integers(len(viable)))]
    a = _standardise(matrix[:, first])
    b = _standardise(matrix[:, second])

    if scenario == "real_synergy":
        y = a * b + 0.3 * rng.normal(size=matrix.shape[0])
        truth = {
            "informative": [names_all[first], names_all[second]],
            "redundant": [],
            "synergy": [names_all[first], names_all[second]],
            "k": 2,
        }
    else:
        y = 0.7 * a + 0.7 * b + 0.2 * rng.normal(size=matrix.shape[0])
        truth = {
            "informative": [names_all[first], names_all[second]],
            "redundant": [],
            "synergy": [],
        }

    return matrix, y, names_all, truth


def _select(
    arm: str, X: np.ndarray, y: np.ndarray, names: Sequence[str], k: int, seed: int
) -> list[str]:
    weights = ARMS[arm]
    if weights is None:
        report = scorecard_ranking(X, y, feature_names=names, max_features=k)
    else:
        report = recommend_features(
            X, y, feature_names=names, k=k, weights=weights, seed=seed
        )
    return report.selected_names()


def _score_selection(
    selected: Sequence[str], truth: dict[str, Any]
) -> dict[str, float]:
    informative = set(truth["informative"])
    redundant = set(truth["redundant"])
    synergy = set(truth["synergy"])
    chosen = list(selected)
    unique_informative = chosen and len(set(chosen) & informative) > 0
    recovered = len(set(chosen) & informative)
    return {
        "n_selected": float(len(chosen)),
        "recovered_informative": float(recovered),
        "recall_informative": (
            float(recovered / len(informative)) if informative else float("nan")
        ),
        "recovered_synergy": float(len(set(chosen) & synergy)),
        "selected_redundant": float(len(set(chosen) & redundant)),
        # A duplicate wastes a slot: it is counted as a false positive for the
        # purpose of "distinct informative features recovered".
        "distinct_informative": float(len(set(chosen) & informative)),
        "first_position_informative": float(
            1.0 if chosen and chosen[0] in informative else 0.0
        ),
        "any_informative": float(1.0 if unique_informative else 0.0),
    }


def run(scenarios: Sequence[str], seeds: Sequence[int], k: int = 3) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for scenario in scenarios:
        for seed in seeds:
            if scenario.startswith("real_"):
                X, y, names, truth = _real_descriptor_design(scenario, seed)
            else:
                X, y, names, truth = _gaussian_design(scenario, seed)
            # Some scenarios need a specific budget: a pure interaction pair is
            # only meaningful when the selector may take exactly two features.
            scenario_k = int(truth.get("k", k))
            for arm in ARMS:
                selected = _select(arm, X, y, names, scenario_k, seed)
                scores = _score_selection(selected, truth)
                cells.append(
                    {
                        "scenario": scenario,
                        "seed": int(seed),
                        "arm": arm,
                        "k": scenario_k,
                        "selected": list(selected),
                        **scores,
                    }
                )
                print(
                    f"  {scenario:18s} seed={seed} {arm:22s} "
                    f"recall={scores['recall_informative']:.2f} "
                    f"synergy={scores['recovered_synergy']:.0f} "
                    f"dup={scores['selected_redundant']:.0f} "
                    f"selected={selected}",
                    flush=True,
                )
    return {
        "experiment": "E-SYNTH",
        "description": (
            "Ground-truth recovery of informative, redundant and synergistic "
            "features on synthetic and real-descriptor planted-signal designs"
        ),
        "k": k,
        "seeds": list(seeds),
        "scenarios": list(scenarios),
        "arms": list(ARMS),
        "cells": cells,
    }


def summarise(payload: dict[str, Any]) -> dict[str, Any]:
    """Aggregate mean/std per (scenario, arm)."""
    from collections import defaultdict

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for cell in payload["cells"]:
        buckets[(cell["scenario"], cell["arm"])].append(cell)
    metrics = (
        "recall_informative",
        "recovered_synergy",
        "selected_redundant",
        "first_position_informative",
    )
    summary: dict[str, Any] = {}
    for (scenario, arm), cells in buckets.items():
        entry = {}
        for metric in metrics:
            values = np.asarray([cell[metric] for cell in cells], dtype=float)
            values = values[np.isfinite(values)]
            entry[metric] = (
                {"mean": float(values.mean()), "std": float(values.std())}
                if values.size
                else {"mean": float("nan"), "std": float("nan")}
            )
        summary.setdefault(scenario, {})[arm] = {
            "n_seeds": len(cells),
            **entry,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="E-SYNTH ground-truth recovery")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=[
            "marginal",
            "duplicate",
            "suppression",
            "xor",
            "masked_synergy",
            "noise",
            "real_synergy",
            "real_suppression",
        ],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    payload = run(args.scenarios, args.seeds, k=args.k)
    payload["summary"] = summarise(payload)

    if not args.no_write:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / "experiment_e_synth_recovery.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved: {path}")

    print("\nSummary (mean over seeds):")
    for scenario, arms in payload["summary"].items():
        print(f"\n  {scenario}")
        for arm, stats in arms.items():
            recall = stats["recall_informative"]["mean"]
            synergy = stats["recovered_synergy"]["mean"]
            dup = stats["selected_redundant"]["mean"]
            first = stats["first_position_informative"]["mean"]
            print(
                f"    {arm:22s} recall={recall:.2f} synergy={synergy:.2f} "
                f"dup={dup:.2f} first={first:.2f}"
            )


if __name__ == "__main__":
    main()
