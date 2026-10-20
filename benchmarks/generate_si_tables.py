"""Render the Additional-file tables that changed with the corrected statistics.

The tables in Additional file 1 are generated here from the same objects that
``paper_statistics.py`` builds, so that the numbers in the manuscript and the
numbers in the repository cannot drift apart.  Every table environment written
by this script can be pasted into ``additional_file_1.tex`` unchanged.

Run::

    python benchmarks/paper_statistics.py        # refresh paper_statistics.json
    python benchmarks/generate_si_tables.py      # refresh the LaTeX fragments

Both scripts accept ``--results-dir`` (and this one ``--out-dir``), so the same
commands reproduce the tables from an archived snapshot directory::

    python benchmarks/paper_statistics.py --results-dir <snapshot>
    python benchmarks/generate_si_tables.py --results-dir <snapshot>
    python benchmarks/validate_paper_statistics.py --statistics <snapshot>/paper_statistics.json
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import paper_statistics  # noqa: E402
from benchmarks.paper_statistics import (  # noqa: E402
    PRIMARY_METRIC,
    Comparison,
    build_external,
    build_internal,
    build_recommended_vs_reference,
    build_routing,
    build_selectors,
    build_sensitivity,
    holm_adjust,
)

METRIC_LABEL = {"rmse": "RMSE", "roc_auc": "AUROC"}

#: Display names that must match the manuscript.
DISPLAY = {
    "esol": "ESOL",
    "freesolv": "FreeSolv",
    "lipophilicity": "Lipophilicity",
    "bbbp": "BBBP",
    "bace": "BACE",
    "bbbp_sensitivity": "BBBP (cleaned)",
}


def p_math(p: float, holm: bool = False, marker: str = "") -> str:
    if not math.isfinite(p):
        return "n/a"
    body = "$<0.001$" if p < 0.001 else f"${p:.3f}$"
    return body + marker


def p_cell(c: Comparison) -> str:
    """Holm-adjusted p with a significance marker inside the math group."""
    p = c.holm_p if c.holm_p is not None else 1.0
    if p < 0.01:
        star = "^{**}"
    elif p < 0.05:
        star = "^{*}"
    else:
        star = ""
    if not math.isfinite(p):
        return "n/a"
    if p < 0.001:
        return f"$<0.001{star}$"
    return f"${p:.3f}{star}$" if star else f"${p:.3f}$"


def p_raw(c: Comparison) -> str:
    return "$<0.001$" if c.nadeau_p < 0.001 else f"${c.nadeau_p:.3f}$"


def delta_cell(c: Comparison, digits: int = 3, sd_digits: int = 4) -> str:
    return f"${c.delta:+.{digits}f}\\pm{c.sd_pop:.{sd_digits}f}$"


def delta_only(value: float, digits: int = 3) -> str:
    return f"${value:+.{digits}f}$"


def ci(interval: tuple[float, float], digits: int = 3) -> str:
    return f"$[{interval[0]:.{digits}f},{interval[1]:.{digits}f}]$"


def excludes_zero(interval: tuple[float, float]) -> bool:
    """True when a two-sided interval lies entirely away from zero."""
    low, high = interval
    return (low > 0.0 and high > 0.0) or (low < 0.0 and high < 0.0)


def join_english(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + ", and " + items[-1]


def comparison_name(c: Comparison, with_dataset: bool = True) -> str:
    name = f"{c.label_a} vs {c.label_b}"
    return f"{name} on {DISPLAY[c.dataset]}" if with_dataset else name


def significant(comps: list[Comparison]) -> list[Comparison]:
    return [c for c in comps if (c.holm_p if c.holm_p is not None else 1.0) < 0.05]


def verdict(comps: list[Comparison]) -> str:
    """One sentence naming the comparisons that survive Holm correction."""
    survivors = significant(comps)
    if not survivors:
        return (
            "No comparison in this table survives Holm correction, so every entry "
            "is reported as inconclusive."
        )
    named = join_english([comparison_name(c) for c in survivors])
    return (
        f"The comparisons that survive Holm correction are {named}; every other "
        "entry is reported as inconclusive."
    )


# ---------------------------------------------------------------------------
def descriptor_effect_sentence() -> str:
    """Direction of descriptor augmentation, recomputed from the result files.

    Reads the framework role ablation and the Chemprop v2 rerun so that the
    statement cannot drift away from the archived numbers.
    """
    from benchmarks import paper_statistics as ps

    improved, worsened, inconsistent = [], [], []
    for dataset, metric in PRIMARY_METRIC.items():
        lower_is_better = metric in ("rmse", "mae")

        def mean_of(records: list[dict[str, float]]) -> float:
            values = [float(record[metric]) for record in records if metric in record]
            return sum(values) / len(values) if values else float("nan")

        ablation = ps.load_json(f"modularity_ablation_{dataset}.json")
        chemprop = ps.load_json(f"chemprop_v2_{dataset}.json")
        framework_delta = mean_of(ablation["raw_results"]["direct_hybrid"]) - mean_of(
            ablation["raw_results"]["residual_only"]
        )
        chemprop_delta = mean_of(chemprop["raw_results"]["chemprop_hybrid"]) - mean_of(
            chemprop["raw_results"]["chemprop_graph"]
        )
        framework_gain = -framework_delta if lower_is_better else framework_delta
        chemprop_gain = -chemprop_delta if lower_is_better else chemprop_delta
        if framework_gain > 0 and chemprop_gain > 0:
            improved.append(DISPLAY[dataset])
        elif framework_gain < 0 and chemprop_gain < 0:
            worsened.append(DISPLAY[dataset])
        else:
            inconsistent.append(DISPLAY[dataset])

    parts = []
    if improved:
        parts.append(f"raises the mean score in both implementations on {join_english(improved)}")
    if worsened:
        parts.append(f"lowers it in both on {join_english(worsened)}")
    if inconsistent:
        parts.append(f"disagrees between implementations on {join_english(inconsistent)}")
    return "Adding the 210 descriptors " + join_english(parts) + "."


def table_internal(comps: list[Comparison]) -> str:
    rows: list[str] = []
    table_comps = [c for c in comps if c.table == "internal"]
    for dataset in PRIMARY_METRIC:
        metric = METRIC_LABEL[PRIMARY_METRIC[dataset]]
        members = [c for c in table_comps if c.dataset == dataset]
        for index, c in enumerate(members):
            label = f"{DISPLAY[dataset]} ({metric})" if index == 0 else ""
            rows.append(
                f"{label} & {c.label_a} vs {c.label_b} & {delta_cell(c)} & {ci(c.naive_ci)} & "
                f"{ci(c.nadeau_ci)} & {p_raw(c)} & {p_cell(c)} & {c.wins}/{c.k} \\\\"
            )
        rows.append("\\midrule")
    if rows:
        rows.pop()

    k = table_comps[0].k if table_comps else 0
    m = len(table_comps)
    close_naive = [c for c in table_comps if excludes_zero(c.naive_ci) and not excludes_zero(c.nadeau_ci)]
    if close_naive:
        extra = (
            "In the nominal interval "
            + join_english([comparison_name(c) for c in close_naive])
            + (" exclude" if len(close_naive) > 1 else " excludes")
            + " zero, but not after correction."
        )
    else:
        extra = ""

    body = "\n".join(rows)
    return rf"""\begin{{table}}[H]
\caption{{Internal paired comparisons on the main models. Hybrid: \cfg{{direct\_hybrid}}; graph only: \cfg{{residual\_only}}; prior/residual: \cfg{{prior\_residual}}. All entries are paired on the same {k} saved splits, and win counts are out of {k}. $\bar\Delta$ is the mean paired difference $m_A-m_B$ with the population SD of the differences; wins count the splits on which $A$ was better. Both intervals are two-sided $t$ intervals with {k - 1} degrees of freedom: the nominal interval uses the paired standard error, and the corrected interval inflates the variance by $1/k+n_{{\mathrm{{test}}}}/n_{{\mathrm{{train}}}}$ to reflect the overlap of the training partitions (Nadeau--Bengio). $p$ is the corrected two-sided $p$-value and $p_{{\mathrm{{H}}}}$ is Holm-adjusted within this table ($m={m}$); $^{{*}}p_{{\mathrm{{H}}}}<0.05$, $^{{**}}p_{{\mathrm{{H}}}}<0.01$. {verdict(table_comps)} {extra}}}\label{{tab:int}}
\centering\scriptsize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{@{{}}llccccccc@{{}}}}
\toprule
Dataset (metric) & Comparison ($A$ vs $B$) & $\bar\Delta\pm\mathrm{{SD}}$ & Nominal 95\% interval & Corrected 95\% interval & $p$ & $p_{{\mathrm{{H}}}}$ & Wins of $A$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def table_external(comps: list[Comparison]) -> str:
    rows: list[str] = []
    table_comps = [c for c in comps if c.table == "external"]
    for dataset in PRIMARY_METRIC:
        metric = METRIC_LABEL[PRIMARY_METRIC[dataset]]
        for c in [c for c in table_comps if c.dataset == dataset]:
            label = c.label_a.replace("Framework ", "")
            rows.append(
                f"{DISPLAY[dataset]} ({metric}) & {label} & {delta_cell(c)} & {ci(c.naive_ci)} & "
                f"{ci(c.nadeau_ci)} & {p_raw(c)} & {p_cell(c)} & {c.wins}/{c.k} \\\\"
            )
    body = "\n".join(rows)
    m = len(table_comps)
    k = table_comps[0].k if table_comps else 0
    return rf"""\begin{{table}}[H]
\caption{{Framework minus Chemprop v2.3.1 on matched splits, for the graph only model and for the graph model with the 210 descriptors. Conventions follow Table~\ref{{tab:int}}; all entries are paired on the same {k} splits and $p_{{\mathrm{{H}}}}$ is Holm-adjusted within this table ($m={m}$). {verdict(table_comps)} {descriptor_effect_sentence()} Because a comparison between the two implementations at this sample size is underpowered, the agreement is descriptive.}}\label{{tab:ext}}
\centering\scriptsize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{@{{}}llcccccc@{{}}}}
\toprule
Dataset (metric) & Model pair & $\bar\Delta\pm\mathrm{{SD}}$ & Nominal 95\% interval & Corrected 95\% interval & $p$ & $p_{{\mathrm{{H}}}}$ & Wins of framework \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def table_rec_random(comps: list[Comparison]) -> str:
    rows_random: list[str] = []
    rows_full: list[str] = []
    panel: dict[str, dict[str, Comparison]] = {"random": {}, "full": {}}
    for dataset in ["esol", "freesolv", "bbbp"]:
        members = [c for c in comps if c.table == "rec_random" and c.dataset == dataset]
        for index, k in enumerate(["5", "10", "20", "50"]):
            label = DISPLAY[dataset] if index == 0 else ""
            c_random = next(
                c for c in members
                if c.label_a == f"Recommended K={k}" and c.label_b == f"Random K={k}"
            )
            c_full = next(
                c for c in members
                if c.label_a == f"Recommended K={k}"
                and c.label_b == "Full 210-descriptor pool"
            )
            panel["random"][f"{dataset}|{k}"] = c_random
            panel["full"][f"{dataset}|{k}"] = c_full
            rows_random.append(
                f"{label} & {k} & {delta_only(c_random.delta)} & {ci(c_random.nadeau_ci)} & "
                f"{p_cell(c_random)} & {c_random.wins}/{c_random.k} \\\\"
            )
            rows_full.append(
                f"{label} & {k} & {delta_only(c_full.delta)} & {ci(c_full.nadeau_ci)} & "
                f"{p_cell(c_full)} & {c_full.wins}/{c_full.k} \\\\"
            )
        rows_random.append("\\midrule")
        rows_full.append("\\midrule")
    rows_random.pop()
    rows_full.pop()

    def panel_cells(entries: dict[str, Comparison]) -> tuple[str, str, str]:
        """(recommended better, recommended worse, budgets with no separation)."""
        better, worse, flat = [], [], []
        for key, c in entries.items():
            dataset, k = key.split("|")
            cell = f"{DISPLAY[dataset]} $K={k}$"
            if (c.holm_p if c.holm_p is not None else 1.0) >= 0.05:
                flat.append(cell)
            elif (c.delta < 0) == (c.metric in ("rmse", "mae")):
                better.append(cell)
            else:
                worse.append(cell)
        return join_english(better), join_english(worse), len(flat)

    better_r, worse_r, flat_r = panel_cells(panel["random"])
    better_f, worse_f, flat_f = panel_cells(panel["full"])
    k_random = next(iter(panel["random"].values())).k

    random_sentence = (
        "Against random subsets of the same size, the recommended subsets score significantly "
        + (f"better at {better_r}" if better_r else "better nowhere")
        + (f" and significantly worse at {worse_r}" if worse_r else " and nowhere worse")
        + f"; the remaining {flat_r} of the 12 dataset-budget cells do not separate. "
    )
    full_sentence = (
        "Against the full 210-descriptor pool, the corrected differences are "
        + (f"favourable to the recommended subsets at {better_f}" if better_f else "nowhere favourable")
        + (f" and unfavourable at {worse_f}" if worse_f else " and nowhere unfavourable")
        + f", leaving {flat_f} of 12 cells inconclusive."
    )

    def header() -> str:
        return (
            "\\toprule\nDataset & $K$ & $\\bar\\Delta$ & Corrected 95\\% interval & "
            "$p_{\\mathrm{H}}$ & Wins of recommended \\\\\n\\midrule"
        )

    return rf"""\begin{{table}}[H]
\caption{{Framework-recommended top-$K$ subsets versus random $K$-subsets (ten draws per split, averaged) and versus the full 210-descriptor pool, with a Random Forest downstream. All entries are paired on the same {k_random} splits. $\bar\Delta$ is the mean paired difference, recommended minus the comparator, so a negative value in panel (a) and a positive value in panel (b) mean that the recommended subset scores worse than the comparator (RMSE for ESOL and FreeSolv; AUROC for BBBP). $p_{{\mathrm{{H}}}}$ is Holm-adjusted within each dataset across the eight comparisons of the two panels ($m=8$); $^{{*}}p_{{\mathrm{{H}}}}<0.05$, $^{{**}}p_{{\mathrm{{H}}}}<0.01$. {random_sentence}{full_sentence} Random subsets average ten draws per split, so differences against them carry the draw noise of the comparator.}}\label{{tab:rec-random}}
\centering\scriptsize
\setlength{{\tabcolsep}}{{4pt}}
\textbf{{(a) Recommended versus random subsets}}\par\smallskip
\begin{{tabular}}{{@{{}}llcccc@{{}}}}
{header()}
{chr(10).join(rows_random)}
\bottomrule
\end{{tabular}}

\medskip\textbf{{(b) Recommended versus the full 210-descriptor pool}}\par\smallskip
\begin{{tabular}}{{@{{}}llcccc@{{}}}}
{header()}
{chr(10).join(rows_full)}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def table_mde(comps: list[Comparison]) -> str:
    rows = []
    k = 0
    for dataset in PRIMARY_METRIC:
        c = next(
            c
            for c in comps
            if c.table == "internal"
            and c.dataset == dataset
            and c.label_a == "Hybrid"
            and c.label_b == "RDKit 2D + RF"
        )
        k = c.k
        rows.append(
            f"{DISPLAY[dataset]} & {METRIC_LABEL[c.metric]} & {c.sd_pop:.3f} & {c.mde:.3f} \\\\"
        )
    body = "\n".join(rows)
    return rf"""\begin{{table}}[H]
\caption{{Minimum detectable difference of the hybrid-versus-Random-Forest comparison at 80\% power with {k} splits, two-sided $\alpha=0.05$, computed from the corrected (Nadeau--Bengio) standard error and reported in the units of the primary metric. Where the detectable difference exceeds the observed difference (Table~\ref{{tab:int}}), the study cannot separate the two models, which is the quantitative reason why those comparisons are reported as inconclusive rather than as evidence of equivalence.}}\label{{tab:mde}}
\centering\footnotesize
\begin{{tabular}}{{@{{}}llcc@{{}}}}
\toprule
Dataset & Metric & SD of paired differences & Minimum detectable difference \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def table_sensitivity_stats(comps: list[Comparison]) -> str:
    table_comps = [c for c in comps if c.table == "sensitivity"]
    rows = []
    for c in table_comps:
        rows.append(
            f"{c.label_a} $-$ {c.label_b} & {delta_cell(c, 4, 4)} & {ci(c.nadeau_ci, 4)} & "
            f"{p_raw(c)} & {p_cell(c)} & {c.wins}/{c.k} \\\\"
        )
    body = "\n".join(rows)
    k = table_comps[0].k if table_comps else 0
    m = len(table_comps)
    return rf"""\begin{{table}}[H]
\caption{{Paired statistics for the duplicate-controlled BBBP sensitivity analysis ({k} new scaffold manifests; the same manifests are used for every model). $\bar\Delta$ is $A$ minus $B$ in AUROC; $p_{{\mathrm{{H}}}}$ is Holm-adjusted within this table ($m={m}$). {verdict(table_comps)} Because the cleaned dataset and its split manifests differ from the primary benchmark, the ranking seen here cannot be attributed to duplicate leakage alone.}}\label{{tab:bbbp-sensitivity-stats}}
\centering\scriptsize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{@{{}}lccccr@{{}}}}
\toprule
Comparison ($A$ vs $B$) & $\bar\Delta\pm\mathrm{{SD}}$ & Corrected 95\% interval & $p$ & $p_{{\mathrm{{H}}}}$ & Wins of $A$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def table_routing_stats(comps: list[Comparison]) -> str:
    table_comps = [c for c in comps if c.table == "routing"]
    rows = []
    for dataset in PRIMARY_METRIC:
        for c in [c for c in table_comps if c.dataset == dataset]:
            rows.append(
                f"{DISPLAY[dataset]} & {c.label_b} & {delta_cell(c, 3, 3)} & {ci(c.nadeau_ci)} & "
                f"{p_raw(c)} & {p_cell(c)} & {c.wins}/{c.k} \\\\"
            )
        rows.append("\\midrule")
    if rows:
        rows.pop()
    body = "\n".join(rows)
    k = table_comps[0].k if table_comps else 0
    non_survivors = [c for c in table_comps if (c.holm_p if c.holm_p is not None else 1.0) >= 0.05]
    if non_survivors:
        smallest_p = min(non_survivors, key=lambda c: c.nadeau_p)
        closest = (
            "The smallest corrected $p$-value in this table belongs to "
            f"{comparison_name(smallest_p)} ($p={smallest_p.nadeau_p:.3f}$, "
            f"$p_{{\\mathrm{{H}}}}={smallest_p.holm_p:.3f}$)."
        )
    else:
        closest = ""
    return rf"""\begin{{table}}[H]
\caption{{Paired comparisons for the harmonised descriptor routing: the recommender route minus each reference route, on the {k} splits of Table~\ref{{tab:routing-si}}. Negative values on the regression tasks and positive values on the classification tasks mean that the recommender route is better. $p_{{\mathrm{{H}}}}$ is Holm-adjusted within each dataset ($m=5$); $^{{*}}p_{{\mathrm{{H}}}}<0.05$, $^{{**}}p_{{\mathrm{{H}}}}<0.01$. {verdict(table_comps)} {closest}}}\label{{tab:routing-stats}}
\centering\scriptsize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{@{{}}llccccr@{{}}}}
\toprule
Dataset & Recommender minus & $\bar\Delta\pm\mathrm{{SD}}$ & Corrected 95\% interval & $p$ & $p_{{\mathrm{{H}}}}$ & Wins \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def table_selectors_stats(comps: list[Comparison]) -> str:
    table_comps = [c for c in comps if c.table == "selectors"]
    rows = []
    for dataset in PRIMARY_METRIC:
        flagged = sorted(
            (c for c in table_comps if c.dataset == dataset
             and (c.holm_p or 1.0) < 0.05),
            key=lambda c: c.holm_p or 1.0,
        )
        if not flagged:
            rows.append(
                f"{DISPLAY[dataset]} & \\multicolumn{{6}}{{l}}{{no comparison survives the correction}} \\\\"
            )
            continue
        for c in flagged:
            rows.append(
                f"{DISPLAY[dataset]} & {c.label_a} vs {c.label_b} & {delta_cell(c, 3, 3)} & "
                f"{ci(c.nadeau_ci)} & {p_raw(c)} & {p_cell(c)} & {c.wins}/{c.k} \\\\"
            )
    body = "\n".join(rows)
    total = len(table_comps)
    survivors = significant(table_comps)
    if survivors:
        summary = (
            f"Of the {total} selector comparisons, "
            + (f"{len(survivors)} survive: " if len(survivors) > 1 else "1 survives: ")
            + join_english([comparison_name(c) for c in survivors])
            + ". On the other endpoints no selector separates from the framework selector once "
            "multiplicity is controlled, which is consistent with the mean-rank table rather "
            "than in tension with it."
        )
    else:
        summary = (
            f"None of the {total} selector comparisons separates from the framework selector once "
            "multiplicity is controlled, which is consistent with the mean-rank table rather "
            "than in tension with it."
        )
    return rf"""\begin{{table}}[H]
\caption{{Selector comparisons that survive Holm correction within a dataset (family $m=20$: the five other selectors against the framework selector at four budgets, on the same splits as Table~\ref{{tab:sel-full}}). {summary}}}\label{{tab:selector-stats}}
\centering\scriptsize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{@{{}}llccccr@{{}}}}
\toprule
Dataset & Comparison ($A$ vs $B$) & $\bar\Delta\pm\mathrm{{SD}}$ & Corrected 95\% interval & $p$ & $p_{{\mathrm{{H}}}}$ & Wins of $A$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render the supplementary tables from the benchmark results."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=paper_statistics.RESULTS_DIR,
        help="Directory holding the benchmark result JSON files (default: benchmarks/results)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write the .tex fragments (default: <results-dir>/paper_tables)",
    )
    args = parser.parse_args(argv)
    paper_statistics.RESULTS_DIR = args.results_dir
    out_dir = args.out_dir or (args.results_dir / "paper_tables")

    comps: list[Comparison] = []
    for builder in (
        build_internal,
        build_external,
        build_recommended_vs_reference,
        build_selectors,
        build_routing,
        build_sensitivity,
    ):
        comps.extend(builder())
    holm_adjust(comps)

    out_dir.mkdir(parents=True, exist_ok=True)
    fragments = {
        "tab_int.tex": table_internal(comps),
        "tab_ext.tex": table_external(comps),
        "tab_rec_random.tex": table_rec_random(comps),
        "tab_mde.tex": table_mde(comps),
        "tab_sensitivity_stats.tex": table_sensitivity_stats(comps),
        "tab_routing_stats.tex": table_routing_stats(comps),
        "tab_selector_stats.tex": table_selectors_stats(comps),
    }
    for name, body in fragments.items():
        (out_dir / name).write_text(body + "\n", encoding="utf-8")
        print(f"wrote {out_dir / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
