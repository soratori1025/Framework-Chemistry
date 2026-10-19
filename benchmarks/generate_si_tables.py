"""Render the Additional-file tables that changed with the corrected statistics.

The tables in Additional file 1 are generated here from the same objects that
``paper_statistics.py`` builds, so that the numbers in the manuscript and the
numbers in the repository cannot drift apart.  Every table environment written
by this script can be pasted into ``additional_file_1.tex`` unchanged.

Run::

    python benchmarks/paper_statistics.py        # refresh paper_statistics.json
    python benchmarks/generate_si_tables.py      # refresh the LaTeX fragments
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.paper_statistics import (  # noqa: E402
    OUT_DIR,
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


# ---------------------------------------------------------------------------
def table_internal(comps: list[Comparison]) -> str:
    rows: list[str] = []
    for dataset in PRIMARY_METRIC:
        metric = METRIC_LABEL[PRIMARY_METRIC[dataset]]
        members = [c for c in comps if c.table == "internal" and c.dataset == dataset]
        for index, c in enumerate(members):
            label = f"{DISPLAY[dataset]} ({metric})" if index == 0 else ""
            rows.append(
                f"{label} & {c.label_a} vs {c.label_b} & {delta_cell(c)} & {ci(c.naive_ci)} & "
                f"{ci(c.nadeau_ci)} & {p_raw(c)} & {p_cell(c)} & {c.wins}/{c.k} \\\\"
            )
        rows.append("\\midrule")
    if rows:
        rows.pop()

    body = "\n".join(rows)
    return rf"""\begin{{table}}[H]
\caption{{Internal paired comparisons on the main models. Hybrid: \cfg{{direct\_hybrid}}; graph only: \cfg{{residual\_only}}; prior/residual: \cfg{{prior\_residual}}. $\bar\Delta$ is the mean paired difference $m_A-m_B$ with the population SD of the differences; wins count the splits on which $A$ was better. Both intervals are two-sided $t$ intervals with four degrees of freedom: the nominal interval uses the paired standard error, and the corrected interval inflates the variance by $1/k+n_{{\mathrm{{test}}}}/n_{{\mathrm{{train}}}}$ to reflect the overlap of the training partitions (Nadeau--Bengio). $p$ is the corrected two-sided $p$-value and $p_{{\mathrm{{H}}}}$ is Holm-adjusted within this table ($m=20$); $^{{*}}p_{{\mathrm{{H}}}}<0.05$, $^{{**}}p_{{\mathrm{{H}}}}<0.01$. Only the hybrid-versus-Random-Forest comparison on Lipophilicity survives the correction; the ESOL and BACE comparisons that exclude zero in the nominal interval have corrected intervals that just include zero and are reported as inconclusive.}}\label{{tab:int}}
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
    for dataset in PRIMARY_METRIC:
        metric = METRIC_LABEL[PRIMARY_METRIC[dataset]]
        for c in [c for c in comps if c.table == "external" and c.dataset == dataset]:
            label = c.label_a.replace("Framework ", "")
            rows.append(
                f"{DISPLAY[dataset]} ({metric}) & {label} & {delta_cell(c)} & {ci(c.naive_ci)} & "
                f"{ci(c.nadeau_ci)} & {p_raw(c)} & {p_cell(c)} & {c.wins}/{c.k} \\\\"
            )
    body = "\n".join(rows)
    return rf"""\begin{{table}}[H]
\caption{{Framework minus Chemprop v2.3.1 on matched splits, for the graph only model and for the graph model with the 210 descriptors. Conventions follow Table~\ref{{tab:int}}; $p_{{\mathrm{{H}}}}$ is Holm-adjusted within this table ($m=10$). No comparison approaches the corrected threshold, so these data do not separate the two implementations, and the agreement in the direction of the descriptor effect is descriptive.}}\label{{tab:ext}}
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

    def header() -> str:
        return (
            "\\toprule\nDataset & $K$ & $\\bar\\Delta$ & Corrected 95\\% interval & "
            "$p_{\\mathrm{H}}$ & Wins of recommended \\\\\n\\midrule"
        )

    return rf"""\begin{{table}}[H]
\caption{{Framework-recommended top-$K$ subsets versus random $K$-subsets (ten draws per split, averaged) and versus the full 210-descriptor pool, with a Random Forest downstream. $\bar\Delta$ is the mean paired difference, recommended minus the comparator, so a negative value in panel (a) and a positive value in panel (b) mean that the recommended subset scores worse than the comparator (RMSE for ESOL and FreeSolv; AUROC for BBBP). $p_{{\mathrm{{H}}}}$ is Holm-adjusted within each dataset across the eight comparisons of the two panels ($m=8$); $^{{*}}p_{{\mathrm{{H}}}}<0.05$, $^{{**}}p_{{\mathrm{{H}}}}<0.01$. Recommended subsets beat random subsets of the same size at every budget on the two regression tasks and at $K=5$ on BBBP; against the full pool the only corrected differences are ESOL at $K=5$ and $K=10$, where the recommended subset is worse. Random subsets average ten draws per split, so those differences carry the draw noise of the comparator.}}\label{{tab:rec-random}}
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
    for dataset in PRIMARY_METRIC:
        c = next(
            c
            for c in comps
            if c.table == "internal"
            and c.dataset == dataset
            and c.label_a == "Hybrid"
            and c.label_b == "RDKit 2D + RF"
        )
        rows.append(
            f"{DISPLAY[dataset]} & {METRIC_LABEL[c.metric]} & {c.sd_pop:.3f} & {c.mde:.3f} \\\\"
        )
    body = "\n".join(rows)
    return rf"""\begin{{table}}[H]
\caption{{Minimum detectable difference of the hybrid-versus-Random-Forest comparison at 80\% power with five splits, two-sided $\alpha=0.05$, computed from the corrected (Nadeau--Bengio) standard error. The detectable difference exceeds several of the observed differences, which is the quantitative reason why most comparisons are reported as inconclusive rather than as evidence of equivalence. Values are in the units of the primary metric.}}\label{{tab:mde}}
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
    rows = []
    for c in [c for c in comps if c.table == "sensitivity"]:
        rows.append(
            f"{c.label_a} $-$ {c.label_b} & {delta_cell(c, 4, 4)} & {ci(c.nadeau_ci, 4)} & "
            f"{p_raw(c)} & {p_cell(c)} & {c.wins}/{c.k} \\\\"
        )
    body = "\n".join(rows)
    return rf"""\begin{{table}}[H]
\caption{{Paired statistics for the duplicate-controlled BBBP sensitivity analysis (five new scaffold manifests). $\bar\Delta$ is $A$ minus $B$ in AUROC; $p_{{\mathrm{{H}}}}$ is Holm-adjusted within this table ($m=5$). The hybrid-minus-graph difference is inconclusive for both implementations, and the difference between the framework hybrid and the RDKit 2D Random Forest does not survive multiplicity correction.}}\label{{tab:bbbp-sensitivity-stats}}
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
    rows = []
    for dataset in PRIMARY_METRIC:
        for c in [c for c in comps if c.table == "routing" and c.dataset == dataset]:
            rows.append(
                f"{DISPLAY[dataset]} & {c.label_b} & {delta_cell(c, 3, 3)} & {ci(c.nadeau_ci)} & "
                f"{p_raw(c)} & {p_cell(c)} & {c.wins}/{c.k} \\\\"
            )
        rows.append("\\midrule")
    if rows:
        rows.pop()
    body = "\n".join(rows)
    return rf"""\begin{{table}}[H]
\caption{{Paired comparisons for the harmonised descriptor routing: the recommender route minus each reference route, on the splits of Table~\ref{{tab:routing-si}}. Negative values on the regression tasks and positive values on the classification tasks mean that the recommender route is better. $p_{{\mathrm{{H}}}}$ is Holm-adjusted within each dataset ($m=5$). No comparison across the five endpoints and five reference routes survives the correction; the largest corrected effect is the recommender against random subsets on FreeSolv ($p=0.064$ before adjustment, $p=0.318$ after).}}\label{{tab:routing-stats}}
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
    rows = []
    for dataset in PRIMARY_METRIC:
        flagged = sorted(
            (c for c in comps if c.table == "selectors" and c.dataset == dataset
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
    return rf"""\begin{{table}}[H]
\caption{{Selector comparisons that survive Holm correction within a dataset (family $m=20$: the five other selectors against the framework selector at four budgets). Of the 100 selector comparisons, one survives: mutual information outperforms the framework selector on Lipophilicity at $K=5$. On the other four endpoints no selector separates from the framework selector once multiplicity is controlled, which is consistent with the mean-rank table rather than in tension with it.}}\label{{tab:selector-stats}}
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


def main() -> int:
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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
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
        (OUT_DIR / name).write_text(body + "\n", encoding="utf-8")
        print(f"wrote {OUT_DIR / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
