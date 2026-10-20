"""Print every number the manuscript quotes, straight from the result files.

The manuscript text and the released artefacts must not drift apart, so this
script re-derives the headline numbers from ``benchmarks/results`` (or any
snapshot passed with ``--results-dir``) and prints them grouped by manuscript
table:

* Tables 3/4  canonical baselines, per model and dataset
* Table 8    role ablation (modularity) means
* Table 5    selector mean ranks, best count, and subset stability
* Table 7    harmonised descriptor routing
* Figure 2   descriptor effect, as relative changes of the endpoint means
* external   Chemprop v2.3.1 comparison
* sensitivity duplicate-controlled BBBP analysis
* statistics paired differences with corrected intervals, Holm-adjusted
             p-values and minimum detectable differences

Run::

    python benchmarks/paper_statistics.py --results-dir <dir>
    python benchmarks/report_manuscript_numbers.py --results-dir <dir>

The selector block is a regression check: run against the archived five-split
snapshot it reproduces the selector table of that snapshot (RF impurity
importance 2.20 overall, 10 best cells, Jaccard 0.64), which shows that the
ranking convention still matches the archived table.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = REPO_ROOT / "benchmarks" / "results"

DATASETS = ["esol", "freesolv", "lipophilicity", "bbbp", "bace"]
SELECTOR_DATASETS = ["esol", "freesolv", "lipophilicity", "bbbp", "bace"]
BUDGETS = ["5", "10", "20", "50"]
PRIMARY_METRIC = {
    "esol": "rmse",
    "freesolv": "rmse",
    "lipophilicity": "rmse",
    "bbbp": "roc_auc",
    "bace": "roc_auc",
}
LOWER_IS_BETTER = {"rmse", "mae"}
DISPLAY = {
    "esol": "ESOL",
    "freesolv": "FreeSolv",
    "lipophilicity": "Lipophilicity",
    "bbbp": "BBBP",
    "bace": "BACE",
    "bbbp_sensitivity": "BBBP (cleaned)",
}
SELECTOR_DISPLAY = {
    "framework": "Framework recommender",
    "spearman": "Spearman top-K",
    "mutual_info": "Mutual information",
    "mrmr": "MI-Pearson",
    "rf_importance": "RF impurity importance",
    "l1": "L1 (lasso / L1-logistic)",
}


def load(results_dir: Path, filename: str):
    path = results_dir / filename
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def mean_std(values) -> tuple[float, float]:
    array = np.asarray([float(v) for v in values], dtype=float)
    if array.size == 0:
        return float("nan"), float("nan")
    return float(array.mean()), float(array.std())


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# LaTeX emitters for the main-text tables
# ---------------------------------------------------------------------------
# The manuscript tables are typeset by hand, so this block prints ready-to-paste
# replacements.  Each emitter reproduces the archived table exactly when it is
# run against the archived five-split results, which is the check that the row
# order, bold rule, and number format still match the archived manuscript.
TABLE2_SPEC = [
    ("Reference models", [
        ("Dummy", ("baselines", "dummy")),
        ("ECFP4 + Random Forest", ("baselines", "ecfp4_rf")),
        ("RDKit 2D + Random Forest", ("baselines", "rdkit_rf")),
        ("Descriptor MLP", ("baselines", "descriptor_mlp")),
    ]),
    ("Chemprop v2.3.1", [
        ("Graph-only D-MPNN", ("chemprop", "chemprop_graph")),
        ("D-MPNN + descriptors ($x_d$)", ("chemprop", "chemprop_hybrid")),
    ]),
    ("Framework", [
        (r"Graph only (\cfg{residual\_only})", ("modularity", "residual_only")),
        (r"Graph + descriptors (\cfg{direct\_hybrid})", ("modularity", "direct_hybrid")),
        (r"Prior + graph residual (\cfg{prior\_residual})", ("modularity", "prior_residual")),
    ]),
]
TABLE4_SPEC = [
    (r"\cfg{prior\_only}", "prior_only"),
    (r"\cfg{direct\_only}", "direct_only"),
    (r"\cfg{residual\_only}", "residual_only"),
    (r"\cfg{direct\_hybrid}", "direct_hybrid"),
    (r"\cfg{prior\_residual}", "prior_residual"),
]
ROUTING_COLUMNS = [
    ("Graph only", "residual_only"),
    ("All 210", "direct_hybrid"),
    ("Recommender", "hybrid_rec"),
    ("Spearman", "hybrid_spearman"),
    ("MI", "hybrid_mi"),
    ("Random", "hybrid_random"),
]


def _model_stats(results_dir: Path, dataset: str, source: str, key: str) -> tuple[float, float]:
    metric = PRIMARY_METRIC[dataset]
    if source == "baselines":
        payload = load(results_dir, f"baselines_{dataset}_aggregated.json")
        entry = payload["aggregated"][key][metric]
        return float(entry["mean"]), float(entry["std"])
    if source == "modularity":
        payload = load(results_dir, f"modularity_ablation_{dataset}.json")
        return mean_std(metric_values(payload["raw_results"][key], metric))
    if source == "chemprop":
        payload = load(results_dir, f"chemprop_v2_{dataset}.json")
        return mean_std(metric_values(payload["raw_results"][key], metric))
    raise ValueError(source)


def _seeds_of(results_dir: Path, dataset: str, source: str) -> object:
    if source == "baselines":
        payload = load(results_dir, f"baselines_{dataset}_aggregated.json")
    elif source == "modularity":
        payload = load(results_dir, f"modularity_ablation_{dataset}.json")
    else:
        payload = load(results_dir, f"chemprop_v2_{dataset}.json")
    seeds = payload.get("seeds")
    if seeds is None and payload.get("raw_runs"):
        seeds = sorted({r.get("seed") for r in payload["raw_runs"]})
    if seeds is None:
        for records in payload.get("raw_results", {}).values():
            seeds = sorted({r.get("seed") for r in records if isinstance(r, dict)})
            break
    return seeds


def _latex_cell(mean: float, std: float, bold: bool) -> str:
    body = f"{mean:.3f}\\pm{std:.3f}"
    return f"$\\mathbf{{{body}}}$" if bold else f"${body}$"


def emit_performance_tables(results_dir: Path) -> None:
    section("LaTeX: Tables 3 (regression) and 4 (classification)")
    for group, datasets, ncols in (
        ("regression", ["esol", "freesolv", "lipophilicity"], 4),
        ("classification", ["bbbp", "bace"], 3),
    ):
        print()
        print(f"% ---- Table {2 if group == 'regression' else 3}: {group} ----")
        # means per row
        rows = []
        for section_title, entries in TABLE2_SPEC:
            rows.append(("section", section_title, None))
            for label, (source, key) in entries:
                stats = {d: _model_stats(results_dir, d, source, key) for d in datasets}
                rows.append(("row", label, stats))
        # bold rule: best mean per dataset column
        best = {}
        for dataset in datasets:
            values = [stats[dataset][0] for kind, _, stats in rows if kind == "row"]
            best[dataset] = max(values) if PRIMARY_METRIC[dataset] not in LOWER_IS_BETTER else min(values)
        for kind, label, stats in rows:
            if kind == "section":
                print(f"\\multicolumn{{{ncols}}}{{@{{}}l}}{{\\emph{{{label}}}}} \\\\")
                continue
            cells = []
            for dataset in datasets:
                mean, std = stats[dataset]
                gap = 5e-4  # ties at the displayed precision
                bold = abs(mean - best[dataset]) <= gap
                cells.append(_latex_cell(mean, std, bold))
            print(f"{label} & " + " & ".join(cells) + " \\\\")
        seeds_line = ", ".join(f"{DISPLAY[d]}={_seeds_of(results_dir, d, 'baselines')}" for d in datasets)
        print(f"% seeds: {seeds_line}")


def emit_ablation_table(results_dir: Path) -> None:
    section("LaTeX: Table 8 (role ablation)")
    datasets = DATASETS
    stats = {}
    for config, key in TABLE4_SPEC:
        stats[config] = {d: _model_stats(results_dir, d, "modularity", key) for d in datasets}
    best = {}
    for dataset in datasets:
        values = [stats[config][dataset][0] for config, _ in TABLE4_SPEC]
        best[dataset] = max(values) if PRIMARY_METRIC[dataset] not in LOWER_IS_BETTER else min(values)
    print()
    for config, _ in TABLE4_SPEC:
        cells = []
        for dataset in datasets:
            mean, std = stats[config][dataset]
            bold = abs(mean - best[dataset]) <= 5e-4
            cells.append(_latex_cell(mean, std, bold))
        print(f"{config} & " + " & ".join(cells) + " \\\\")
    print(f"% seeds (ESOL): {_seeds_of(results_dir, 'esol', 'modularity')}")


def emit_selector_table(results_dir: Path) -> None:
    section("LaTeX: Table 5 (selectors)")
    studies = {
        d: load(results_dir, f"multivariate_feature_study_{d}.json") for d in SELECTOR_DATASETS
    }
    studies = {d: p for d, p in studies.items() if p is not None}
    if not studies:
        print("(missing)")
        return
    ranks = {selector: {} for selector in studies[next(iter(studies))]["summary"]}
    best_count = {selector: 0 for selector in ranks}
    jaccard = {selector: [] for selector in ranks}
    for dataset, payload in studies.items():
        metric = payload["primary_metric"]
        lower = metric in LOWER_IS_BETTER
        per_selector_ranks = {selector: [] for selector in payload["summary"]}
        for k in BUDGETS:
            pairs = [
                (selector, payload["summary"][selector][k]["mean"])
                for selector in payload["summary"]
                if payload["summary"][selector].get(k)
            ]
            order = sorted(pairs, key=lambda item: item[1], reverse=not lower)
            position = 1
            index = 0
            while index < len(order):
                j = index
                while j + 1 < len(order) and math.isclose(order[j + 1][1], order[index][1], abs_tol=1e-12):
                    j += 1
                average_rank = (position + position + (j - index)) / 2.0
                for m in range(index, j + 1):
                    per_selector_ranks[order[m][0]].append(average_rank)
                position += j - index + 1
                index = j + 1
            best_count[order[0][0]] += 1
        for selector in payload["summary"]:
            ranks[selector][dataset] = float(np.mean(per_selector_ranks[selector]))
            jaccard[selector].append(
                float(np.mean([payload["summary"][selector][k]["stability_jaccard"] for k in BUDGETS]))
            )
    order = sorted(ranks, key=lambda s: np.mean([ranks[s].get(d, np.nan) for d in SELECTOR_DATASETS]))
    print()
    print("  & ESOL & FreeSolv & Lipo. & BBBP & BACE & Overall & Best & Jaccard \\\\")
    for selector in order:
        per_dataset = [ranks[selector].get(d, float("nan")) for d in SELECTOR_DATASETS]
        overall = float(np.mean(per_dataset))
        cells = " & ".join(f"{v:.1f}" for v in per_dataset)
        label = SELECTOR_DISPLAY.get(selector, selector).replace("MI-Pearson", "MI--Pearson")
        if selector == "framework":
            label = "Framework recommender"
        if selector == "spearman":
            label = "Spearman top-$K$"
        print(
            f"{label} & {cells} & {overall:.2f} & {best_count[selector]:d} & "
            f"{float(np.mean(jaccard[selector])):.2f} \\\\"
        )


def emit_routing_table(results_dir: Path) -> None:
    section("LaTeX: Table 7 (routing)")
    stats = {}
    for dataset in DATASETS:
        payload = load(results_dir, f"evidence_routing_{dataset}.json")
        if payload is None:
            print(f"{DISPLAY[dataset]:<15} (missing)")
            continue
        metric = payload["primary_metric"]
        stats[dataset] = {
            key: mean_std(metric_values(payload["summary"][key]["per_split"], metric))
            for _, key in ROUTING_COLUMNS
        }
    print()
    for dataset in DATASETS:
        if dataset not in stats:
            continue
        metric = PRIMARY_METRIC[dataset]
        lower = metric in LOWER_IS_BETTER
        values = [stats[dataset][key][0] for _, key in ROUTING_COLUMNS]
        best = min(values) if lower else max(values)
        arrow = "RMSE $\\downarrow$" if lower else "AUROC $\\uparrow$"
        cells = []
        for _, key in ROUTING_COLUMNS:
            mean, std = stats[dataset][key]
            cells.append(_latex_cell(mean, std, abs(mean - best) <= 5e-4))
        print(f"{DISPLAY[dataset]} & {arrow} & " + " & ".join(cells) + " \\\\")


def emit_figure1_coordinates(results_dir: Path) -> None:
    section("LaTeX: Figure 2 coordinates (descriptor effect, percent change)")
    framework = []
    chemprop = []
    for dataset in DATASETS:
        metric = PRIMARY_METRIC[dataset]
        lower = metric in LOWER_IS_BETTER
        graph = _model_stats(results_dir, dataset, "modularity", "residual_only")[0]
        hybrid = _model_stats(results_dir, dataset, "modularity", "direct_hybrid")[0]
        framework.append(((graph - hybrid) if lower else (hybrid - graph)) / graph * 100.0)
        c_graph = _model_stats(results_dir, dataset, "chemprop", "chemprop_graph")[0]
        c_hybrid = _model_stats(results_dir, dataset, "chemprop", "chemprop_hybrid")[0]
        chemprop.append(
            ((c_graph - c_hybrid) if lower else (c_hybrid - c_graph)) / c_graph * 100.0
        )
    names = ["ESOL", "FreeSolv", "Lipophilicity", "BBBP", "BACE"]
    fmt = lambda values: " ".join(f"({n},{v:.2f})" for n, v in zip(names, values))
    print()
    print(f"\\addplot[fill=blue!55!black, draw=none] coordinates {{{fmt(framework)}}};")
    print(f"\\addplot[fill=orange!80!black, draw=none] coordinates {{{fmt(chemprop)}}};")


def _n_test_of(dataset: str, seed: int) -> int:
    """Test-partition size of a split, read from the split manifest when the
    range-shift record predates the ``n_test_molecules`` field."""
    split_path = REPO_ROOT / "benchmarks" / "splits" / f"{dataset}_seed_{seed}.json"
    data = json.loads(split_path.read_text(encoding="utf-8"))
    return len(data["test"])


def emit_si_shift_table(results_dir: Path) -> None:
    """SI Table S2(b): BBBP split rows plus the summary sentence numbers."""
    section("LaTeX: SI Table S2(b) (BBBP range shift vs role models)")
    shift = load(results_dir, "ood_shift_all_splits.json")
    ablation = load(results_dir, "modularity_ablation_bbbp.json")
    baselines = load(results_dir, "baselines_bbbp_aggregated.json")
    if shift is None or ablation is None or "bbbp" not in shift:
        print("(missing ood_shift_all_splits.json / modularity_ablation_bbbp.json)")
        return

    rows = {int(row["split"]): row for row in shift["bbbp"]}
    roles = {
        r"\cfg{direct\_only}": "direct_only",
        r"\cfg{direct\_hybrid}": "direct_hybrid",
        r"\cfg{residual\_only}": "residual_only",
        r"\cfg{prior\_residual}": "prior_residual",
    }
    role_scores = {}
    for name, key in roles.items():
        records = ablation["raw_results"][key]
        seeds = ablation.get("seeds") or list(range(1, len(records) + 1))
        role_scores[name] = {
            int(seed): float(value) for seed, value in zip(seeds, metric_values(records, "roc_auc"))
        }
    mlp_scores = {
        int(run["seed"]): float(run["metrics"]["roc_auc"])
        for run in baselines.get("raw_runs", [])
        if run.get("model") == "descriptor_mlp"
    }

    print()
    for seed in sorted(rows):
        row = rows[seed]
        n_test = int(row["n_test_molecules"]) if "n_test_molecules" in row else _n_test_of("bbbp", seed)
        n_affected = round(row["pct_mols_affected"] / 100.0 * n_test)
        cells = [
            str(seed),
            f"{row['n_ood_descriptors']} ({row['pct_ood_descriptors']:.1f}\\%)",
            f"{n_affected} ({row['pct_mols_affected']:.1f}\\%)",
        ]
        for name in roles:
            value = role_scores[name].get(seed)
            cells.append(f"{value:.4f}" if value is not None else "--")
        print(" & ".join(cells) + " \\\\")

    counts = np.array([rows[s]["n_ood_descriptors"] for s in sorted(rows)], dtype=float)
    affected = np.array([rows[s]["pct_mols_affected"] for s in sorted(rows)], dtype=float)
    print(f"% BBBP OOD descriptors {counts.mean():.1f} +/- {counts.std():.1f}, "
          f"molecules affected {affected.mean():.1f} +/- {affected.std():.1f}%")

    # Summary numbers for the "across all splits" sentence (SI, scorecard section).
    for dataset in ("esol", "lipophilicity", "bbbp"):
        if dataset not in shift:
            continue
        records = shift[dataset]
        n = np.array([r["n_ood_descriptors"] for r in records], dtype=float)
        pct_desc = np.array([r["pct_ood_descriptors"] for r in records], dtype=float)
        pct_mol = np.array([r["pct_mols_affected"] for r in records], dtype=float)
        print(
            f"% {dataset:<15} descriptors {n.mean():.1f} +/- {n.std():.1f} "
            f"({pct_desc.mean():.1f} +/- {pct_desc.std():.1f}% of 210), "
            f"molecules affected {pct_mol.mean():.1f} +/- {pct_mol.std():.1f}%"
        )

    # Weakest-split contrast: the split where both descriptor-MLP models are weakest.
    direct_only_key = r"\cfg{direct\_only}"
    if mlp_scores and role_scores[direct_only_key]:
        combined = {
            seed: (mlp_scores[seed] + role_scores[direct_only_key][seed]) / 2.0
            for seed in sorted(rows)
            if seed in mlp_scores and seed in role_scores[direct_only_key]
        }
        if combined:
            weakest = min(combined, key=combined.get)
            others = [rows[s]["n_ood_descriptors"] for s in sorted(rows) if s != weakest]
            print(
                f"% weakest descriptor-MLP split: {weakest} "
                f"(mean AUROC {combined[weakest]:.4f}, baseline {mlp_scores[weakest]:.4f}, "
                f"direct_only {role_scores[direct_only_key][weakest]:.4f})"
            )
            print(
                f"%   OOD descriptors on that split {rows[weakest]['n_ood_descriptors']}, "
                f"mean of the other {len(others)} splits {np.mean(others):.1f}"
            )


def metric_values(records, metric):
    """Primary-metric values from a list of record dicts or a list of floats."""
    out = []
    for record in records:
        if isinstance(record, (int, float)) and not isinstance(record, bool):
            out.append(float(record))
        elif isinstance(record, dict) and metric in record:
            out.append(float(record[metric]))
    return out


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------
def print_baselines(results_dir: Path) -> None:
    section("Canonical baselines (Tables 3 and 4): mean +/- population SD over seeds")
    for dataset in DATASETS:
        payload = load(results_dir, f"baselines_{dataset}_aggregated.json")
        if payload is None:
            print(f"{DISPLAY[dataset]:<15} (missing)")
            continue
        metric = PRIMARY_METRIC[dataset]
        seeds = payload.get("seeds")
        print(f"{DISPLAY[dataset]}  [{metric}, seeds={seeds}]")
        aggregated = payload.get("aggregated", {})
        rows = []
        for model, stats in aggregated.items():
            if isinstance(stats, dict) and metric in stats and isinstance(stats[metric], dict):
                rows.append((model, stats[metric].get("mean"), stats[metric].get("std", float("nan"))))
            elif isinstance(stats, dict) and metric in stats:
                rows.append((model, stats[metric], float("nan")))
            elif isinstance(stats, dict) and "mean" in stats:
                rows.append((model, stats["mean"], stats.get("std", float("nan"))))
        for model, mean, std in sorted(rows, key=lambda item: item[1]):
            print(f"    {model:<28} {mean:8.4f} +/- {std:.4f}")


# ---------------------------------------------------------------------------
# role ablation
# ---------------------------------------------------------------------------
def print_modularity(results_dir: Path) -> None:
    section("Role ablation (Table 8): mean +/- population SD over seeds")
    configs = ["prior_only", "direct_only", "residual_only", "direct_hybrid", "prior_residual"]
    for dataset in DATASETS:
        payload = load(results_dir, f"modularity_ablation_{dataset}.json")
        if payload is None:
            print(f"{DISPLAY[dataset]:<15} (missing)")
            continue
        metric = PRIMARY_METRIC[dataset]
        seeds = payload.get("seeds")
        print(f"{DISPLAY[dataset]}  [{metric}, seeds={seeds}]")
        for config in configs:
            records = payload["raw_results"].get(config, [])
            mean, std = mean_std(metric_values(records, metric))
            print(f"    {config:<20} {mean:8.4f} +/- {std:.4f}")


# ---------------------------------------------------------------------------
# selector study (Table 5)
# ---------------------------------------------------------------------------
def print_selectors(results_dir: Path) -> None:
    section("Selector study (Table 5): mean rank per dataset, overall, best cells, Jaccard")
    studies = {}
    for dataset in SELECTOR_DATASETS:
        payload = load(results_dir, f"multivariate_feature_study_{dataset}.json")
        if payload is not None:
            studies[dataset] = payload
    if not studies:
        print("(missing)")
        return

    rank_tables = {}
    best_count = {}
    jaccard = {}
    for dataset, payload in studies.items():
        metric = payload["primary_metric"]
        lower = metric in LOWER_IS_BETTER
        selectors = sorted(payload["summary"])
        for selector in selectors:
            best_count.setdefault(selector, 0)
            jaccard.setdefault(selector, [])
        ranks = {selector: [] for selector in selectors}
        for k in BUDGETS:
            scores = []
            for selector in selectors:
                entry = payload["summary"][selector].get(k)
                if entry is None:
                    continue
                scores.append((selector, entry["mean"]))
            if not scores:
                continue
            order = sorted(scores, key=lambda item: item[1], reverse=not lower)
            # average ranks for ties
            position = 1
            index = 0
            while index < len(order):
                j = index
                while j + 1 < len(order) and math.isclose(order[j + 1][1], order[index][1], abs_tol=1e-12):
                    j += 1
                average_rank = (position + position + (j - index)) / 2.0
                for m in range(index, j + 1):
                    ranks[order[m][0]].append(average_rank)
                position += j - index + 1
                index = j + 1
            best_count[order[0][0]] += 1
        for selector in selectors:
            jaccard[selector].append(
                np.mean([payload["summary"][selector][k]["stability_jaccard"] for k in BUDGETS])
            )
        rank_tables[dataset] = {selector: float(np.mean(ranks[selector])) for selector in selectors}

    selectors = sorted({s for table in rank_tables.values() for s in table})
    header = f"{'Selector':<26}" + "".join(f"{DISPLAY[d]:>9}" for d in SELECTOR_DATASETS)
    header += f"{'Overall':>9}{'Best':>6}{'Jaccard':>9}"
    print(header)
    print("-" * len(header))
    rows = []
    for selector in selectors:
        per_dataset = [rank_tables[d].get(selector, float("nan")) for d in SELECTOR_DATASETS]
        overall = float(np.mean(per_dataset))
        rows.append((selector, per_dataset, overall))
    for selector, per_dataset, overall in sorted(rows, key=lambda item: item[2]):
        label = SELECTOR_DISPLAY.get(selector, selector)
        cells = "".join(f"{value:9.2f}" for value in per_dataset)
        print(f"{label:<26}{cells}{overall:9.2f}{best_count.get(selector, 0):6d}{np.mean(jaccard.get(selector, [np.nan])):9.2f}")

    print()
    print("Per-dataset, per-budget means (primary metric):")
    for dataset, payload in studies.items():
        metric = payload["primary_metric"]
        print(f"  {DISPLAY[dataset]} ({metric})")
        print("      " + f"{'Selector':<26}" + "".join(f"{'K=' + k:>10}" for k in BUDGETS))
        for selector in sorted(payload["summary"]):
            label = SELECTOR_DISPLAY.get(selector, selector)
            cells = ""
            for k in BUDGETS:
                entry = payload["summary"][selector].get(k)
                cells += f"{entry['mean']:10.4f}" if entry else f"{'--':>10}"
            print(f"      {label:<26}{cells}")


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------
def print_routing(results_dir: Path) -> None:
    section("Harmonised descriptor routing (Table 7): mean +/- population SD over splits")
    for dataset in DATASETS:
        payload = load(results_dir, f"evidence_routing_{dataset}.json")
        if payload is None:
            print(f"{DISPLAY[dataset]:<15} (missing)")
            continue
        metric = PRIMARY_METRIC[dataset]
        summary = payload["summary"]
        seeds = payload.get("seeds")
        print(f"{DISPLAY[dataset]}  [{metric}, seeds={seeds}]")
        for route, entry in summary.items():
            per_split = entry["per_split"]
            mean, std = mean_std(metric_values(per_split, metric))
            print(f"    {route:<20} {mean:8.4f} +/- {std:.4f}   (K*={entry.get('best_k')})")


# ---------------------------------------------------------------------------
# external comparison
# ---------------------------------------------------------------------------
def print_external(results_dir: Path) -> None:
    section("Chemprop v2.3.1 comparison: mean +/- population SD over seeds")
    for dataset in DATASETS:
        payload = load(results_dir, f"chemprop_v2_{dataset}.json")
        if payload is None:
            print(f"{DISPLAY[dataset]:<15} (missing)")
            continue
        metric = PRIMARY_METRIC[dataset]
        seeds = payload.get("seeds")
        print(f"{DISPLAY[dataset]}  [{metric}, seeds={seeds}]")
        for model, records in payload["raw_results"].items():
            mean, std = mean_std(metric_values(records, metric))
            print(f"    {model:<20} {mean:8.4f} +/- {std:.4f}")


# ---------------------------------------------------------------------------
# sensitivity
# ---------------------------------------------------------------------------
def print_sensitivity(results_dir: Path) -> None:
    section("Duplicate-controlled BBBP sensitivity analysis: mean +/- SD over splits")
    payload = load(results_dir, "bbbp_sensitivity_study.json")
    if payload is not None:
        print(f"framework study (seeds={payload.get('seeds')})")
        for model, entry in payload["summary"].items():
            mean, std = mean_std(metric_values(entry["per_split"], "roc_auc"))
            print(f"    {model:<20} {mean:8.4f} +/- {std:.4f}")
    payload = load(results_dir, "chemprop_bbbp_sensitivity.json")
    if payload is not None:
        print(f"chemprop study (seeds={payload.get('seeds')})")
        for model, entry in payload["summary"].items():
            values = entry["values"] if "values" in entry else entry["per_split"]
            mean, std = mean_std(metric_values(values, "roc_auc"))
            print(f"    {model:<20} {mean:8.4f} +/- {std:.4f}")


# ---------------------------------------------------------------------------
# corrected statistics
# ---------------------------------------------------------------------------
def print_statistics(results_dir: Path) -> None:
    payload = load(results_dir, "paper_statistics.json")
    if payload is None:
        print()
        print("paper_statistics.json not found - run benchmarks/paper_statistics.py first")
        return
    comps = payload["comparisons"]
    section(f"Corrected statistics ({payload['n_comparisons']} comparisons)")
    print(f"primary interval : {payload['primary_interval']}")
    print(f"multiplicity     : {payload['multiplicity']}")
    for family in ("internal", "external", "rec_random", "sensitivity", "routing", "selectors"):
        rows = [c for c in comps if c["table"] == family]
        if not rows:
            continue
        print()
        print(f"-- family {family} (m within dataset as noted) --")
        for c in rows:
            flag = ""
            holm = c["holm_p"]
            if holm is not None and holm < 0.05:
                flag = "  <-- survives Holm"
            print(
                f"  {DISPLAY.get(c['dataset'], c['dataset']):<15} {c['label_a']:<42} vs "
                f"{c['label_b']:<42} d={c['delta']:+.4f} "
                f"[{c['nadeau_ci'][0]:+.3f},{c['nadeau_ci'][1]:+.3f}] "
                f"p={c['nadeau_p']:.4f} pH={holm if holm is not None else float('nan'):.4f} "
                f"wins={c['wins']}/{c['k']} MDE={c['mde']:.3f}{flag}"
            )

    section("Minimum detectable difference at 80% power (hybrid vs RDKit 2D + RF)")
    for c in comps:
        if (
            c["table"] == "internal"
            and c["label_a"] == "Hybrid"
            and c["label_b"] == "RDKit 2D + RF"
        ):
            print(
                f"  {DISPLAY.get(c['dataset'], c['dataset']):<15} observed {c['delta']:+.4f}  "
                f"MDE {c['mde']:.4f}  SD {c['sd_pop']:.4f}  k={c['k']}"
            )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Directory holding the benchmark result JSON files",
    )
    parser.add_argument(
        "--only",
        default="all",
        help="Comma-separated subset of: baselines,modularity,selectors,routing,external,sensitivity,stats,latex",
    )
    args = parser.parse_args(argv)
    results_dir: Path = args.results_dir
    wanted = set(args.only.split(",")) if args.only != "all" else {
        "baselines", "modularity", "selectors", "routing", "external", "sensitivity", "stats", "latex"
    }

    print(f"results dir: {results_dir}")
    if "baselines" in wanted:
        print_baselines(results_dir)
    if "modularity" in wanted:
        print_modularity(results_dir)
    if "selectors" in wanted:
        print_selectors(results_dir)
    if "routing" in wanted:
        print_routing(results_dir)
    if "external" in wanted:
        print_external(results_dir)
    if "sensitivity" in wanted:
        print_sensitivity(results_dir)
    if "stats" in wanted:
        print_statistics(results_dir)
    if "latex" in wanted:
        emit_performance_tables(results_dir)
        emit_ablation_table(results_dir)
        emit_selector_table(results_dir)
        emit_routing_table(results_dir)
        emit_figure1_coordinates(results_dir)
        emit_si_shift_table(results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
