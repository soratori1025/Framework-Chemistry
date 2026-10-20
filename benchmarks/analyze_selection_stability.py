"""E-STABILITY: how reproducible is each selector's chosen feature set?

A feature selector that returns a different five descriptors on every resample is
not usable evidence, even if its average held-out score is acceptable. This
script turns the per-seed ``selection_frequency`` maps produced by the E-REC
ablation into three stability measures.

Definitions
-----------
For one (dataset, seed, arm), let ``f_j`` be the fraction of the ``k``-fold
outer splits in which feature ``j`` was selected. Because every fold selects
exactly ``k`` features, ``sum_j f_j = k``.

``max_frequency``
    ``max_j f_j``. How often the single most reliable feature is chosen. A
    value of ``1.0`` means one feature is selected in every fold.
``concentration``
    ``sum_j f_j ** 2 / sum_j f_j``. The probability that two folds agree on a
    randomly drawn selected slot. ``1 / k`` is the value expected under uniform
    random selection; ``1.0`` means every fold chose the same set.
``n_consistent``
    Number of features selected in at least half of the folds.

These are computed per seed and then averaged, so the reported mean carries a
between-seed standard deviation.

Usage
-----
    python benchmarks/analyze_selection_stability.py \
        --results benchmarks/results/experiment_e_rec_recommendation_ablation.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _seed_metrics(frequencies: dict[str, float], k: int) -> dict[str, float]:
    values = np.asarray([value for value in frequencies.values()], dtype=float)
    values = values[np.isfinite(values)]
    total = float(values.sum())
    if total <= 0:
        return {
            "max_frequency": 0.0,
            "concentration": 0.0,
            "n_consistent": 0.0,
            "uniform_baseline": 1.0 / max(k, 1),
        }
    return {
        "max_frequency": float(values.max()),
        "concentration": float((values**2).sum() / total),
        "n_consistent": float((values >= 0.5).sum()),
        "uniform_baseline": 1.0 / max(k, 1),
    }


def analyse(payload: dict[str, Any]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    k = int(payload.get("k", 5))
    for dataset, block in payload.get("per_dataset", {}).items():
        runs = block.get("runs", [])
        arms = sorted(
            {arm for run in runs for arm in run.get("arms", {})}
        )
        results[dataset] = {"k": k, "uniform_baseline": 1.0 / max(k, 1), "arms": {}}
        for arm in arms:
            per_seed: list[dict[str, float]] = []
            pooled: dict[str, list[float]] = {}
            for run in runs:
                arm_block = run.get("arms", {}).get(arm)
                if not arm_block:
                    continue
                frequencies = arm_block.get("selection_frequency") or {}
                if not frequencies:
                    continue
                per_seed.append(_seed_metrics(frequencies, k))
                for name, value in frequencies.items():
                    pooled.setdefault(name, []).append(float(value))
            if not per_seed:
                continue
            summary: dict[str, Any] = {"n_seeds": len(per_seed)}
            for metric in ("max_frequency", "concentration", "n_consistent"):
                values = np.asarray([item[metric] for item in per_seed], dtype=float)
                summary[metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                }
            summary["core_features"] = [
                {"name": name, "frequency": float(np.mean(values))}
                for name, values in sorted(
                    pooled.items(), key=lambda item: -float(np.mean(item[1]))
                )
                if float(np.mean(values)) >= 0.5
            ]
            results[dataset]["arms"][arm] = summary
    return results


def to_latex(results: dict[str, Any]) -> str:
    """Emit a compact LaTeX table: dataset x arm stability."""
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Dataset & Selector & Max frequency & Concentration & Consistent \\",
        r"\midrule",
    ]
    for dataset, block in results.items():
        arms = block["arms"]
        for position, (arm, stats) in enumerate(arms.items()):
            label = dataset.replace("_", r"\_") if position == 0 else ""
            lines.append(
                f"{label} & {arm.replace('_', chr(92) + '_')} & "
                f"{stats['max_frequency']['mean']:.2f} & "
                f"{stats['concentration']['mean']:.2f} & "
                f"{stats['n_consistent']['mean']:.1f} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="E-STABILITY analysis")
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS_DIR / "experiment_e_rec_recommendation_ablation.json",
    )
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    results = analyse(payload)

    if not args.no_write:
        output = RESULTS_DIR / "experiment_e_stability.json"
        output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        tex = RESULTS_DIR / "paper_tables" / "tab_selection_stability.tex"
        tex.parent.mkdir(parents=True, exist_ok=True)
        tex.write_text(to_latex(results), encoding="utf-8")
        print(f"Saved: {output}")
        print(f"Saved: {tex}")

    for dataset, block in results.items():
        print(f"\n{dataset} (k={block['k']}, uniform={block['uniform_baseline']:.2f})")
        for arm, stats in block["arms"].items():
            print(
                f"  {arm:24s} max={stats['max_frequency']['mean']:.2f} "
                f"conc={stats['concentration']['mean']:.3f} "
                f"consistent={stats['n_consistent']['mean']:.1f} "
                f"core={[item['name'] for item in stats['core_features'][:3]]}"
            )


if __name__ == "__main__":
    main()
