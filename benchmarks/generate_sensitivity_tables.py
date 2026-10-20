"""Render the supplementary tables for the two sensitivity analyses.

Two fragments:

* ``tab_hp_sensitivity.tex`` -- the descriptor effect as a function of encoder
  width (one-factor-at-a-time, FreeSolv, three scaffold splits).
* ``tab_redundancy.tex`` -- the recommender as a function of the redundancy
  threshold used by the greedy pruning step ($K=20$ fixed).

Reads ``hyperparameter_sensitivity.json`` and ``redundancy_sensitivity.json``;
writes .tex fragments into ``manuscript/supplementary/fragments``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "benchmarks" / "results"
OUT = REPO_ROOT / "manuscript" / "supplementary" / "fragments"

LABEL = {"esol": "ESOL", "freesolv": "FreeSolv", "bbbp": "BBBP"}


def _mean_sd(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def table_hyperparameter() -> tuple[str, dict[str, Any]]:
    payload = json.loads((RESULTS / "hyperparameter_sensitivity.json").read_text())
    rows: list[str] = []
    summary: dict[str, Any] = {}
    for dataset, block in payload["datasets"].items():
        per_width: dict[int, dict[str, list[float]]] = {}
        for run in block["runs"]:
            per_width.setdefault(int(run["hidden_dim"]), {}).setdefault(
                run["route"], []
            ).append(float(run["test_score"]))
        for width in sorted(per_width):
            graph = per_width[width].get("graph", [])
            hybrid = per_width[width].get("hybrid", [])
            if not graph or not hybrid:
                continue
            g_mean, g_sd = _mean_sd(graph)
            h_mean, h_sd = _mean_sd(hybrid)
            change = 100.0 * (g_mean - h_mean) / g_mean
            improved = sum(1 for g, h in zip(graph, hybrid) if h < g)
            summary[f"{dataset}/{width}"] = {
                "graph_mean": g_mean,
                "graph_sd": g_sd,
                "hybrid_mean": h_mean,
                "hybrid_sd": h_sd,
                "change_pct": change,
                "splits_improved": improved,
                "n_splits": len(graph),
            }
            rows.append(
                f"{width} & {g_mean:.3f} $\\pm$ {g_sd:.3f} & "
                f"{h_mean:.3f} $\\pm$ {h_sd:.3f} & {change:+.1f} & "
                f"{improved}/{len(graph)} \\\\"
            )

    lines = [
        "\\begin{table}[H]",
        "\\renewcommand{\\thetable}{S21}",
        "\\centering",
        "\\small",
        "\\caption{One-factor-at-a-time sensitivity of the descriptor effect to "
        "encoder width. FreeSolv, scaffold splits 1--3, all other settings as in "
        "the routing experiment. A positive change means the descriptors help: "
        "the hybrid RMSE is lower than graph only. Splits improved counts the "
        "splits in which the hybrid was the more accurate of the two.}",
        "\\label{tab:hp-sensitivity}",
        "\\begin{tabular}{lrrcr}",
        "\\toprule",
        "Width & Graph only & Hybrid & Change (\\%) & Splits improved \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines), summary


def table_redundancy() -> tuple[str, dict[str, Any]]:
    payload = json.loads((RESULTS / "redundancy_sensitivity.json").read_text())
    thresholds = [float(t) for t in payload["thresholds"]]
    datasets = list(payload["datasets"])

    # score[dataset][threshold] = list over splits
    scores: dict[str, dict[float, list[float]]] = {}
    jaccard: dict[str, dict[float, list[float]]] = {}
    for dataset in datasets:
        block = payload["datasets"][dataset]
        for run in block["runs"]:
            threshold = float(run["threshold"])
            scores.setdefault(dataset, {}).setdefault(threshold, []).append(
                float(run["score"])
            )
            if run.get("jaccard_vs_0.85") is not None:
                jaccard.setdefault(dataset, {}).setdefault(threshold, []).append(
                    float(run["jaccard_vs_0.85"])
                )

    summary: dict[str, Any] = {}
    rows: list[str] = []
    for threshold in thresholds:
        cells: list[str] = []
        all_j: list[float] = []
        for dataset in datasets:
            values = scores.get(dataset, {}).get(threshold, [])
            if not values:
                cells.append("--")
                continue
            mean, sd = _mean_sd(values)
            cells.append(f"{mean:.3f} $\\pm$ {sd:.3f}")
            all_j.extend(jaccard.get(dataset, {}).get(threshold, []))
            summary[f"{dataset}/{threshold:g}"] = {"mean": mean, "sd": sd}
        j_mean = float(np.mean(all_j)) if all_j else 1.0
        rows.append(
            f"{threshold:.2f} & {j_mean:.2f} & " + " & ".join(cells) + " \\\\"
        )

    header_cells = " & ".join(f"{LABEL.get(d, d)} ({_unit(d)})" for d in datasets)
    lines = [
        "\\begin{table}[H]",
        "\\renewcommand{\\thetable}{S20}",
        "\\centering",
        "\\small",
        "\\caption{Sensitivity of the recommender to the redundancy threshold. "
        "The greedy pruning step drops a feature when its absolute correlation "
        "with an already-selected feature exceeds the threshold; $K=20$ and all "
        "other settings are fixed. Jaccard is the mean overlap with the set "
        "selected at the default threshold of 0.85, over splits and datasets "
        "(1.00 at the default by construction). Values are mean $\\pm$ "
        "population SD over the ten scaffold splits.}",
        "\\label{tab:redundancy}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        f"Threshold & Jaccard vs 0.85 & {header_cells} \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines), summary


def _unit(dataset: str) -> str:
    return "AUROC" if dataset in {"bbbp", "bace"} else "RMSE"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    hp_tex, hp_summary = table_hyperparameter()
    (OUT / "tab_hp_sensitivity.tex").write_text(hp_tex, encoding="utf-8")
    print("wrote tab_hp_sensitivity.tex")
    for key, value in hp_summary.items():
        print(
            f"  {key}: graph {value['graph_mean']:.3f}+-{value['graph_sd']:.3f} "
            f"hybrid {value['hybrid_mean']:.3f}+-{value['hybrid_sd']:.3f} "
            f"{value['change_pct']:+.1f}% ({value['splits_improved']}/"
            f"{value['n_splits']})"
        )

    red_tex, red_summary = table_redundancy()
    (OUT / "tab_redundancy.tex").write_text(red_tex, encoding="utf-8")
    print("wrote tab_redundancy.tex")
    for key, value in red_summary.items():
        print(f"  {key}: {value['mean']:.3f}+-{value['sd']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
