"""Render the supplementary tables added for the robustness revision.

Two fragments:

* ``tab_random_control.tex`` -- the high-repetition random control: the
  recommender against 50 random subsets per split and budget, in all five
  datasets, with a rank-percentile $p$-value combined over splits (Fisher).
* ``tab_robustness.tex`` -- the distribution-free checks: agreement between the
  Nadeau--Bengio corrected interval and the exact sign-flip permutation test,
  the Friedman/Nemenyi block test over the six selectors, and the routing
  comparisons against the matched random control.

Reads ``random_control_*.json`` and ``statistical_robustness.json``; writes .tex
fragments into ``manuscript/supplementary/fragments``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "benchmarks" / "results"
OUT = REPO_ROOT / "manuscript" / "supplementary" / "fragments"

DATASETS = ["esol", "freesolv", "lipophilicity", "bbbp", "bace"]
LABEL = {"esol": "ESOL", "freesolv": "FreeSolv", "lipophilicity": "Lipophilicity",
         "bbbp": "BBBP", "bace": "BACE"}
UNITS = {"esol": "RMSE", "freesolv": "RMSE", "lipophilicity": "RMSE",
         "bbbp": "AUROC", "bace": "AUROC"}
K_LEVELS = [5, 10, 20, 50]

#: Table numbers in the supplementary file.  The fragment is \input at a fixed
#: position, and the SI numbers its tables explicitly, so the number is recorded
#: here next to the table it belongs to.
SI_NUMBER_RANDOM_CONTROL = "\\renewcommand{\\thetable}{S5}"
SI_NUMBER_ROBUSTNESS = "\\renewcommand{\\thetable}{S19}"


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def signed(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


def tex(text: str) -> str:
    """Escape a selector name for LaTeX text mode."""
    return text.replace("_", "\\_")


def p_value(p: float) -> str:
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def table_random_control() -> str:
    lines: list[str] = []
    lines.append("\\begin{table}[H]")
    lines.append(SI_NUMBER_RANDOM_CONTROL)
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{llrrrrr}")
    lines.append("\\toprule")
    lines.append("Dataset & $K$ & Recommended & Control & $\\bar\\Delta$ & Mean percentile & $p_{\\mathrm{Fisher}}$ \\\\")
    lines.append("\\midrule")
    current = None
    for dataset in DATASETS:
        path = RESULTS / f"random_control_{dataset}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        is_cls = payload["is_classification"]
        better = 1.0 if is_cls else -1.0
        for k in K_LEVELS:
            cell = payload["summary"][str(k)]
            delta = better * (cell["recommended_mean"] - cell["random_mean"])
            if current is not None and current != dataset:
                lines.append("\\midrule")
            current = dataset
            lines.append(
                f"{LABEL[dataset]} & {k} & {fmt(cell['recommended_mean'])} & "
                f"{fmt(cell['random_mean'])} & "
                f"{signed(delta)} & "
                f"{cell['mean_rank_percentile']:.2f} & "
                f"{p_value(cell['p_rank_fisher'])} \\\\"
            )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(
        "\\caption{High-repetition random control for the recommender, in every dataset "
        "and at every budget. Each cell draws 50 independent random descriptor subsets of "
        "size $K$ from the same training partition, trains the same 100-tree Random Forest "
        "and evaluates on the same test partition; the control column is the mean of the 50 "
        "draws. $\\bar\\Delta$ is signed so that a positive value means the recommended "
        "subset was better (lower RMSE or higher AUROC). \\emph{Mean percentile} is the "
        "average over the ten splits of the fraction of random subsets that the "
        "recommendation beat; 0.5 is chance. $p_{\\mathrm{Fisher}}$ combines the ten "
        "per-split rank $p$-values (resolution $1/51$) with Fisher's method and is not "
        "adjusted across the 20 cells. The first ten draws of each cell reproduce the "
        "ten-draw control of Table~\\ref{tab:rec-random} exactly.}\n"
        "\\label{tab:random-control}"
    )
    lines.append("\\end{table}")
    return "\n".join(lines)


def table_robustness() -> str:
    payload = json.loads((RESULTS / "statistical_robustness.json").read_text())
    perm = payload["permutation_tests"]
    fried = payload["selector_friedman"]
    routing = payload["routing_permutation"]
    random_ctl = payload.get("random_control_rank_tests", {})

    lines: list[str] = []
    lines.append("\\begin{table}[H]")
    lines.append(SI_NUMBER_ROBUSTNESS)
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\textbf{(a) Corrected interval versus exact permutation test}\\par\\smallskip")
    lines.append("\\begin{tabular}{lrrr}")
    lines.append("\\toprule")
    lines.append("View & Comparisons & Significant & Agreement \\\\")
    lines.append("\\midrule")
    lines.append(
        f"Nadeau--Bengio interval, Holm within family & {perm['n_comparisons']} & "
        f"{perm['n_significant_holm_nadeau']} & --- \\\\"
    )
    lines.append(
        f"Exact sign-flip permutation, Holm within family & {perm['n_comparisons']} & "
        f"{perm['n_significant_holm_permutation']} & {perm['decision_agreement']} \\\\"
    )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}\\par\\medskip")
    lines.append("\\textbf{(b) Friedman block test over the six selectors}\\par\\smallskip")
    lines.append("\\begin{tabular}{lr}")
    lines.append("\\toprule")
    lines.append("Quantity & Value \\\\")
    lines.append("\\midrule")
    lines.append(f"Blocks (dataset--budget cells) & {fried['n_blocks']} \\\\")
    lines.append(f"Friedman $\\chi^2$ & {fried['friedman_chi2']:.2f} \\\\")
    lines.append(f"$p$ & {p_value(fried['friedman_p'])} \\\\")
    lines.append(
        f"Nemenyi critical difference ($\\alpha=0.05$) & "
        f"{fried['nemenyi_critical_difference_alpha_0.05']:.2f} ranks \\\\"
    )
    best = min(fried["mean_ranks"], key=fried["mean_ranks"].get)
    framework_rank = fried["mean_ranks"].get("framework")
    lines.append(
        f"Best mean rank & \\texttt{{{tex(best)}}} ({fried['mean_ranks'][best]:.2f}) \\\\"
    )
    lines.append(
        f"Framework mean rank & \\texttt{{{tex('framework')}}} ({framework_rank:.2f}) \\\\"
    )
    lines.append(
        f"Gap to the best selector & {abs(fried['mean_ranks'][best] - framework_rank):.2f} ranks \\\\"
    )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}\\par\\medskip")
    lines.append(
        "\\textbf{(c) Routing routes versus the matched random control (permutation)}\\par\\smallskip"
    )
    lines.append("\\begin{tabular}{llrrr}")
    lines.append("\\toprule")
    lines.append("Dataset & Route & $\\bar\\Delta$ & Wins & $p$ (Holm) \\\\")
    lines.append("\\midrule")
    for dataset in DATASETS:
        rows = routing.get(dataset, {})
        for route, label in (("hybrid_rec", "Recommended"), ("hybrid_spearman", "Spearman"),
                             ("hybrid_mi", "MI")):
            if route not in rows:
                continue
            row = rows[route]
            holm = row.get("holm_p_permutation_within_family")
            holm_text = f"{holm:.3f}" if holm is not None else "--"
            lines.append(
                f"{LABEL[dataset]} & {label} & {signed(row['delta_positive_is_better'], 4)} & "
                f"{row['wins']}/10 & {holm_text} \\\\"
            )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    extra = ""
    if random_ctl.get("cells_total"):
        extra = (
            f" In the extended random control, {random_ctl['cells_with_fisher_p_below_0.05']} of "
            f"{random_ctl['cells_total']} dataset--budget cells had a Fisher-combined rank "
            f"$p<0.05$ (Table~\\ref{{tab:random-control}})."
        )
    lines.append(
        "\\caption{Distribution-free checks. (a) The exact sign-flip permutation test "
        "(all $2^{10}$ sign assignments, two-sided) applied to every paired comparison of "
        "the paper and Holm-adjusted within the same families, compared with the "
        "Nadeau--Bengio corrected interval used as the primary analysis. Agreement counts "
        "the comparisons on which the two views reach the same significance decision. "
        "(b) The six selectors as a randomised block design over the 20 dataset--budget "
        "cells of Table~\\ref{tab:sel-full}. (c) Each descriptor route against the matched "
        "random control of the routing experiment, with the same permutation test; $p$ is "
        "Holm-adjusted across the 15 route--dataset comparisons." + extra + "}\n"
        "\\label{tab:robustness}"
    )
    lines.append("\\end{table}")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tab_random_control.tex").write_text(table_random_control() + "\n", encoding="utf-8")
    (OUT / "tab_robustness.tex").write_text(table_robustness() + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'tab_random_control.tex'}")
    print(f"wrote {OUT / 'tab_robustness.tex'}")


if __name__ == "__main__":
    main()
