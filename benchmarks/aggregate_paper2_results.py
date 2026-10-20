"""Collect every Paper-2 experiment into one results file and LaTeX tables.

The experiments run separately (``run_recommendation_ablation.py`` etc.) and
each writes its own JSON. This script reads whatever is present, prints a
human-readable summary, and emits the tables the manuscript includes. Missing
experiments are reported as "not run" rather than failing, so the file can be
regenerated while the sweeps are still in progress.

Usage
-----
    python benchmarks/aggregate_paper2_results.py
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).resolve().parent / "results"
TABLE_DIR = RESULTS_DIR / "paper_tables"

ARM_LABELS = {
    "scorecard": "Scorecard",
    "siren_full": "SIREN (full)",
    "siren_legacy": "SIREN (legacy redundancy)",
    "siren_no_synergy": "SIREN $-$ synergy",
    "siren_no_stability": "SIREN $-$ stability",
    "siren_no_redundancy": "SIREN $-$ redundancy",
    "siren_relevance_only": "SIREN $-$ synergy $-$ stability",
    "siren_no_support_screen": "SIREN $-$ support screen",
    "all_features": "All features",
    "siren_relevance_only_no_screen": "SIREN$-$synergy$-$stability",
}


def _label(arm: str) -> str:
    return ARM_LABELS.get(arm, arm.replace("_", " "))


def _load(name: str) -> dict[str, Any] | None:
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "--"
    return f"{value:.{digits}f}"


def _tex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


# --------------------------------------------------------------------------- #
# E-REC
# --------------------------------------------------------------------------- #
def er_rec_section(payload: dict[str, Any]) -> tuple[list[str], str | None]:
    per_dataset = payload.get("per_dataset", {})
    datasets = [ds for ds in per_dataset if per_dataset[ds].get("arms")]
    if not datasets:
        return ["## E-REC\n\n_not run_"], None

    arms = sorted({arm for ds in datasets for arm in per_dataset[ds]["arms"]})
    lines = [
        "## E-REC — nested selection on the molecular benchmarks",
        "",
        f"k = {payload.get('k')}, {payload.get('n_splits')} outer folds, "
        f"seeds {payload.get('seeds')}.",
        "",
    ]
    header = "| Dataset | " + " | ".join(_label(a) for a in arms) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(arms) + 1))
    for ds in datasets:
        cells = []
        for arm in arms:
            stats = per_dataset[ds]["arms"].get(arm)
            if not stats:
                cells.append("--")
                continue
            cells.append(
                f"{_fmt(stats['mean_test_score'])} ± "
                f"{_fmt(stats['std_across_seeds'], 2)}"
            )
        lines.append(f"| {ds} | " + " | ".join(cells) + " |")

    lines += ["", "Paired difference against the scorecard (same folds):", ""]
    lines.append("| Dataset | " + " | ".join(_label(a) for a in arms[1:]) + " |")
    lines.append("|" + "---|" * len(arms))
    for ds in datasets:
        cells = []
        for arm in arms[1:]:
            stats = per_dataset[ds]["arms"].get(arm, {})
            delta = stats.get("mean_paired_difference_vs_baseline")
            wins = stats.get("significant_wins_vs_baseline")
            total = stats.get("n_seed_comparisons")
            cells.append(
                f"{_fmt(delta, 4)} ({wins}/{total})" if delta is not None else "--"
            )
        lines.append(f"| {ds} | " + " | ".join(cells) + " |")

    tex = [
        r"\begin{tabular}{l" + "c" * len(arms) + "}",
        r"\toprule",
        "Dataset & " + " & ".join(_label(a) for a in arms) + r" \\",
        r"\midrule",
    ]
    for ds in datasets:
        cells = []
        for arm in arms:
            stats = per_dataset[ds]["arms"].get(arm)
            cells.append(
                "--"
                if not stats
                else f"{_fmt(stats['mean_test_score'])} ({_fmt(stats['std_across_seeds'], 2)})"
            )
        tex.append(_tex_escape(ds) + " & " + " & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    return lines, "\n".join(tex)


# --------------------------------------------------------------------------- #
# E-SYNTH
# --------------------------------------------------------------------------- #
def synth_section(payload: dict[str, Any]) -> tuple[list[str], str | None]:
    summary = payload.get("summary", {})
    if not summary:
        return ["## E-SYNTH\n\n_not run_"], None
    scenarios = list(summary)
    arms = sorted({arm for sc in summary.values() for arm in sc})
    lines = [
        "## E-SYNTH — ground-truth recovery",
        "",
        "Recall of the planted informative features (mean over seeds).",
        "",
        "| Scenario | " + " | ".join(_label(a) for a in arms) + " |",
        "|" + "---|" * (len(arms) + 1),
    ]
    for scenario in scenarios:
        cells = []
        for arm in arms:
            stats = summary[scenario].get(arm, {}).get("recall_informative", {})
            cells.append(_fmt(stats.get("mean")))
        lines.append(f"| {scenario} | " + " | ".join(cells) + " |")

    tex = [
        r"\begin{tabular}{l" + "c" * len(arms) + "}",
        r"\toprule",
        "Scenario & " + " & ".join(_label(a) for a in arms) + r" \\",
        r"\midrule",
    ]
    for scenario in scenarios:
        cells = [
            _fmt(summary[scenario].get(arm, {}).get("recall_informative", {}).get("mean"))
            for arm in arms
        ]
        tex.append(_tex_escape(scenario) + " & " + " & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    return lines, "\n".join(tex)


# --------------------------------------------------------------------------- #
# E-LEAK
# --------------------------------------------------------------------------- #
def leak_section(payload: dict[str, Any]) -> tuple[list[str], str | None]:
    summary = payload.get("summary", {})
    if not summary:
        return ["## E-LEAK\n\n_not run_"], None
    lines = [
        "## E-LEAK — optimism from selection leakage",
        "",
        "Held-out Spearman under three protocols; gap is `leaky - nested`.",
        "",
        "| Dataset | Selector | Nested | Leaky | Gap | Circular gap | Jaccard |",
        "|---|---|---|---|---|---|---|",
    ]
    for dataset, arms in summary.items():
        for arm, stats in arms.items():
            lines.append(
                f"| {dataset} | {_label(arm)} | "
                f"{_fmt(stats['nested']['mean'])} | "
                f"{_fmt(stats['leaky']['mean'])} | "
                f"{_fmt(stats['leaky_minus_nested']['mean'], 4)} | "
                f"{_fmt(stats['circular_minus_nested']['mean'], 4)} | "
                f"{_fmt(stats['jaccard_nested_vs_leaky']['mean'], 2)} |"
            )

    tex = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Dataset & Selector & Nested & Leaky & $\Delta_{\mathrm{leak}}$ & Jaccard \\",
        r"\midrule",
    ]
    for dataset, arms in summary.items():
        for arm, stats in arms.items():
            tex.append(
                f"{_tex_escape(dataset)} & {_label(arm)} & "
                f"{_fmt(stats['nested']['mean'])} & "
                f"{_fmt(stats['leaky']['mean'])} & "
                f"{_fmt(stats['leaky_minus_nested']['mean'], 4)} & "
                f"{_fmt(stats['jaccard_nested_vs_leaky']['mean'], 2)} \\\\"
            )
        tex.append(r"\midrule")
    tex[-1] = r"\bottomrule"
    tex.append(r"\end{tabular}")
    return lines, "\n".join(tex)


# --------------------------------------------------------------------------- #
# E-DOWNSTREAM
# --------------------------------------------------------------------------- #
def downstream_section(payload: dict[str, Any]) -> tuple[list[str], str | None]:
    summary = payload.get("summary", {})
    per_dataset = summary.get("per_dataset", {})
    if not per_dataset:
        return ["## E-DOWNSTREAM\n\n_not run_"], None
    metric_by_dataset = summary.get("metric_by_dataset", {})
    lines = [
        "## E-DOWNSTREAM — scaffold-held-out readouts",
        "",
        "Metric per dataset is R² / AUROC (higher is better).",
        "",
        "| Dataset | Readout | " + " | ".join(
            _label(a) for a in sorted({a for ds in per_dataset.values() for r in ds.values() for a in r})
        ) + " |",
    ]
    arms = sorted(
        {a for ds in per_dataset.values() for r in ds.values() for a in r}
    )
    lines.append("|" + "---|" * (len(arms) + 2))
    for dataset, readouts in per_dataset.items():
        for readout, arms_stats in readouts.items():
            cells = [
                f"{_fmt(arms_stats[a]['mean'])} ± {_fmt(arms_stats[a]['std'], 2)}"
                if a in arms_stats
                else "--"
                for a in arms
            ]
            lines.append(f"| {dataset} | {readout} | " + " | ".join(cells) + " |")
    lines += ["", "metric: " + ", ".join(f"{d}={m}" for d, m in metric_by_dataset.items())]

    tex = [
        r"\begin{tabular}{ll" + "c" * len(arms) + "}",
        r"\toprule",
        "Dataset & Readout & " + " & ".join(_label(a) for a in arms) + r" \\",
        r"\midrule",
    ]
    for dataset, readouts in per_dataset.items():
        for readout, arms_stats in readouts.items():
            cells = [
                _fmt(arms_stats[a]["mean"]) if a in arms_stats else "--"
                for a in arms
            ]
            tex.append(
                f"{_tex_escape(dataset)} & {_tex_escape(readout)} & "
                + " & ".join(cells)
                + r" \\"
            )
        tex.append(r"\midrule")
    tex[-1] = r"\bottomrule"
    tex.append(r"\end{tabular}")
    return lines, "\n".join(tex)


# --------------------------------------------------------------------------- #
# E-STABILITY (already aggregated by analyze_selection_stability.py)
# --------------------------------------------------------------------------- #
def stability_section(payload: dict[str, Any]) -> list[str]:
    if not payload:
        return ["## E-STABILITY\n\n_not run_"]
    lines = [
        "## E-STABILITY — reproducibility of the selected set",
        "",
        "| Dataset | Selector | Max frequency | Concentration | Consistent |",
        "|---|---|---|---|---|",
    ]
    for dataset, block in payload.items():
        for arm, stats in block["arms"].items():
            lines.append(
                f"| {dataset} | {_label(arm)} | "
                f"{_fmt(stats['max_frequency']['mean'], 2)} | "
                f"{_fmt(stats['concentration']['mean'], 3)} | "
                f"{_fmt(stats['n_consistent']['mean'], 1)} |"
            )
    return lines


# --------------------------------------------------------------------------- #
# E-CASES
# --------------------------------------------------------------------------- #
def cases_section(payload: dict[str, Any], top: int = 3) -> tuple[list[str], str | None]:
    datasets = payload.get("datasets", {})
    if not datasets:
        return ["## E-CASES\n\n_not run_"], None
    lines = [
        "## E-CASES — named suppressors on the molecular benchmarks",
        "",
        "Descriptors whose association with the target is masked by the selected",
        "context: marginal MI is near zero while conditional MI is not.",
        "",
        "| Dataset | Descriptor | Marginal | Conditional | Gain |",
        "|---|---|---|---|---|",
    ]
    tex = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Dataset & Descriptor & $I(X;y)$ & $I(X;y\mid S)$ & Gain \\",
        r"\midrule",
    ]
    for dataset, block in datasets.items():
        suppressors = block.get("suppressors", [])[:top]
        for position, item in enumerate(suppressors):
            label = _tex_escape(dataset) if position == 0 else ""
            name = item["name"].replace("rdkit_2d.", "")
            lines.append(
                f"| {dataset if position == 0 else ''} | `{name}` | "
                f"{_fmt(item['marginal_mi'])} | {_fmt(item['relevance'])} | "
                f"{_fmt(item['suppression_gain'], 3)} |"
            )
            tex.append(
                f"{label} & \\texttt{{{_tex_escape(name)}}} & "
                f"{_fmt(item['marginal_mi'])} & {_fmt(item['relevance'])} & "
                f"{_fmt(item['suppression_gain'])} \\\\"
            )
        tex.append(r"\midrule")
    tex[-1] = r"\bottomrule"
    tex.append(r"\end{tabular}")
    return lines, "\n".join(tex)


def main() -> None:
    global RESULTS_DIR, TABLE_DIR
    parser = argparse.ArgumentParser(description="Aggregate Paper-2 results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory holding the experiment JSON files",
    )
    args = parser.parse_args()

    RESULTS_DIR = args.results_dir
    TABLE_DIR = RESULTS_DIR / "paper_tables"

    er_rec = _load("experiment_e_rec_recommendation_ablation.json")
    synth = _load("experiment_e_synth_recovery.json")
    leak = _load("experiment_e_leakage.json")
    downstream = _load("experiment_e_downstream_readout.json")
    stability = _load("experiment_e_stability.json")
    cases = _load("experiment_e_suppression_cases.json")

    blocks: list[str] = ["# Paper 2 — collected results", ""]
    tables: dict[str, str] = {}

    for name, payload, fn, texname in (
        ("E-REC", er_rec, er_rec_section, "tab_er_rec.tex"),
        ("E-SYNTH", synth, synth_section, "tab_e_synth.tex"),
        ("E-CASES", cases, cases_section, "tab_e_cases.tex"),
        ("E-LEAK", leak, leak_section, "tab_e_leak.tex"),
        ("E-DOWNSTREAM", downstream, downstream_section, "tab_e_downstream.tex"),
    ):
        if payload is None:
            blocks += [f"## {name}", "", "_not run_", ""]
            continue
        lines, tex = fn(payload)
        blocks += lines + [""]
        if tex:
            tables[texname] = tex

    blocks += stability_section(stability) + [""]

    text = "\n".join(blocks)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "PAPER2_RESULTS.md").write_text(text, encoding="utf-8")
    for name, tex in tables.items():
        (TABLE_DIR / name).write_text(tex, encoding="utf-8")
    print(text)
    print(f"\nSaved: {RESULTS_DIR / 'PAPER2_RESULTS.md'}")
    print(f"Tables: {', '.join(sorted(tables)) or 'none'}")


if __name__ == "__main__":
    main()
