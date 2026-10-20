"""E-CASES: which real descriptors are suppressors or synergy partners?

The synthetic study proves that SIREN can recover planted structure. This script
asks whether the same structure exists in the molecular benchmarks, and names
the descriptors involved so the result can be checked chemically.

For each dataset the selector is run once on the development rows of a scaffold
split (train + validation of ``seed`` 1; the held-out test molecules are never
touched) with a generous budget. Every nomination carries the full decomposition
of its objective:

``marginal_mi``
    Unconditional mutual information -- what the legacy scorecard sees.
``relevance``
    Conditional mutual information given the context.
``suppression_gain``
    ``relevance - marginal_mi``. Positive values are suppressors: descriptors
    whose signal is *masked* by the other descriptors and that a marginal
    ranking cannot promote.
``synergy``
    Mean interaction-information lift against the selected context.
``stability``
    Resample persistence in ``[0, 1]``.
``redundancy``
    Duplicate mass against the context.

Usage
-----
    python benchmarks/export_suppression_cases.py --datasets esol freesolv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chemistry_framework.feature_analysis.recommend import recommend_features

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_WEIGHTS = {
    "relevance": 1.0,
    "synergy": 0.5,
    "stability": 0.5,
    "redundancy": 0.0,
}


def run(
    datasets: Sequence[str], seed: int = 1, budget: int = 15
) -> dict[str, Any]:
    from benchmarks.datasets import load_dataset, load_split

    payload: dict[str, Any] = {
        "experiment": "E-CASES",
        "description": (
            "Descriptor-level decomposition of SIREN nominations on development "
            "rows, highlighting suppressors and synergy partners"
        ),
        "seed": seed,
        "budget": budget,
        "datasets": {},
    }
    for name in datasets:
        dataset = load_dataset(name)
        _, _, split = load_split(name, seed=seed)
        development = np.concatenate([split.train, split.val])
        X = np.asarray(dataset.features, dtype=float)[development]
        y = np.asarray(dataset.target, dtype=float)[development]
        names = list(dataset.feature_names)

        report = recommend_features(
            X,
            y,
            feature_names=names,
            k=budget,
            weights=DEFAULT_WEIGHTS,
            seed=seed,
        )
        nominations = [
            {
                "rank": item.rank,
                "name": item.name,
                "objective": item.objective,
                "relevance": item.relevance,
                "synergy": item.synergy,
                "stability": item.stability,
                "redundancy": item.redundancy,
                "marginal_mi": item.marginal_mi,
                "suppression_gain": item.suppression_gain,
                "dominant_term": item.dominant_term,
            }
            for item in report.nominations
        ]
        suppressors = sorted(
            [item for item in nominations if item["suppression_gain"] > 0.0],
            key=lambda item: -item["suppression_gain"],
        )
        synergistic = sorted(
            [item for item in nominations if item["synergy"] > 0.0],
            key=lambda item: -item["synergy"],
        )
        payload["datasets"][name] = {
            "n_development": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_selected": len(nominations),
            "n_screened": len(report.screened_features),
            "screened_examples": list(report.screened_features[:10]),
            "nominations": nominations,
            "suppressors": suppressors,
            "synergy_candidates": synergistic,
        }
        print(
            f"\n{name}: {X.shape[0]} development rows, {X.shape[1]} descriptors, "
            f"{len(suppressors)} suppressor(s), {len(synergistic)} synergy candidate(s)",
            flush=True,
        )
        for item in suppressors[:5]:
            print(
                f"  SUPPRESSOR {item['name']:44s} "
                f"marginal={item['marginal_mi']:.3f} "
                f"conditional={item['relevance']:.3f} "
                f"gain={item['suppression_gain']:+.3f} "
                f"rank={item['rank']}",
                flush=True,
            )
        for item in synergistic[:5]:
            print(
                f"  SYNERGY    {item['name']:44s} "
                f"synergy={item['synergy']:+.3f} rank={item['rank']}",
                flush=True,
            )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="E-CASES suppressor export")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["esol", "freesolv", "lipophilicity", "bbbp", "bace"],
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--budget", type=int, default=15)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    payload = run(args.datasets, seed=args.seed, budget=args.budget)
    if not args.no_write:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / "experiment_e_suppression_cases.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
