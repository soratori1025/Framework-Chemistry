"""Paired statistics and LaTeX table rows for the manuscript.

Every paired comparison reported in the manuscript and in Additional file 1 is
recomputed here directly from the archived per-split result files in
``benchmarks/results``.  Nothing in this module reads a manuscript value: the
tables are regenerated from raw per-split scores so that the reported numbers
cannot drift away from the artefacts.

Two intervals are produced for each comparison.

``naive``
    ``mean(d) +- t(0.975, k-1) * sd(d) / sqrt(k)``, the interval used in the
    first version of the manuscript.

``nadeau``
    The Nadeau-Bengio corrected resampled interval, used as the primary
    interval.  The ``k`` re-splits are independent partitions of the same
    finite pool of molecules, so their training sets overlap and the naive
    interval is anticonservative.  Inflating the variance by
    ``1/k + n_test/n_train`` compensates for the overlap
    (Nadeau & Bengio, 2003, *Machine Learning* 52:239-281; Dietterich, 1998,
    *Neural Computation* 10:1895-1923)::

        var_corrected = (1/k + n_test/n_train) * var(d)
        t = mean(d) / sqrt(var_corrected),  df = k - 1

    ``k`` equals the number of evaluated scaffold splits (five in the released
    results, up to twenty after re-running the runners with ``--seeds``).  For
    threshold metrics such as AUROC the correction is an approximation and is
    reported as such.

Multiplicity is controlled with Holm's step-down procedure within the family
of comparisons printed in a single table (or a single dataset block of a
table), and both the raw and the adjusted p-values are stored.

Run ``python benchmarks/paper_statistics.py`` to regenerate
``benchmarks/results/paper_statistics.json``.  The LaTeX table environments are
rendered from that JSON by ``benchmarks/generate_si_tables.py``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
SPLITS_DIR = REPO_ROOT / "benchmarks" / "splits"
OUT_DIR = RESULTS_DIR / "paper_tables"

#: Endpoints and the metric that the manuscript treats as primary for each.
PRIMARY_METRIC = {
    "esol": "rmse",
    "freesolv": "rmse",
    "lipophilicity": "rmse",
    "bbbp": "roc_auc",
    "bace": "roc_auc",
}

#: Metric orientation.  Any metric not listed is treated as "higher is better".
LOWER_IS_BETTER = {"rmse", "mae", "loss"}

#: Target for the power statement attached to each comparison.
POWER_TARGET = 0.80


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def load_json(filename: str):
    path = RESULTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"missing result file: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def split_sizes(dataset: str, seeds: list[int]) -> tuple[float, float]:
    """Mean (n_train, n_test) over the manifests actually used.

    Endpoints stored in a subdirectory (``bbbp_sensitivity/``) are found as
    well, so the Nadeau-Bengio correction also applies to the sensitivity
    analysis.
    """
    n_train, n_test = [], []
    for seed in seeds:
        candidates = (
            SPLITS_DIR / f"{dataset}_seed_{seed}.json",
            SPLITS_DIR / dataset / f"{dataset}_seed_{seed}.json",
        )
        path = next((c for c in candidates if c.exists()), None)
        if path is None:
            continue
        with open(path, "r", encoding="utf-8") as fh:
            split = json.load(fh)
        n_train.append(len(split["train"]))
        n_test.append(len(split["test"]))
    if not n_train:
        return float("nan"), float("nan")
    return float(np.mean(n_train)), float(np.mean(n_test))


def ordered(values: dict[int, float], seeds: list[int]) -> np.ndarray:
    missing = [s for s in seeds if s not in values]
    if missing:
        raise KeyError(f"no value for split seed(s) {missing}")
    return np.asarray([values[s] for s in seeds], dtype=float)


@dataclass
class Comparison:
    """A single paired comparison between two models on matched splits."""

    family: str
    table: str
    dataset: str
    metric: str
    label_a: str
    label_b: str
    scores_a: list[float]
    scores_b: list[float]
    n_train: float
    n_test: float
    delta: float
    sd: float
    sd_pop: float
    wins: int
    k: int
    naive_ci: tuple[float, float]
    naive_p: float
    nadeau_ci: tuple[float, float]
    nadeau_p: float
    mde: float
    holm_p: float | None = None

    def as_dict(self) -> dict:
        out = self.__dict__.copy()
        out["naive_ci"] = list(self.naive_ci)
        out["nadeau_ci"] = list(self.nadeau_ci)
        return out


def compare(
    family: str,
    table: str,
    dataset: str,
    metric: str,
    label_a: str,
    label_b: str,
    scores_a: list[float],
    scores_b: list[float],
    n_train: float,
    n_test: float,
) -> Comparison:
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(
            f"{dataset}/{label_a} vs {label_b}: {a.shape} scores versus {b.shape}"
        )
    d = a - b
    k = int(d.size)
    better = d > 0 if metric not in LOWER_IS_BETTER else d < 0

    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if k > 1 else 0.0
    sd_pop = float(d.std(ddof=0))
    t_crit = float(stats.t.ppf(0.975, k - 1)) if k > 1 else float("nan")

    se_naive = sd / math.sqrt(k) if k > 1 else float("nan")
    ci_naive = (mean - t_crit * se_naive, mean + t_crit * se_naive)
    t_naive = mean / se_naive if se_naive else float("nan")
    p_naive = float(2 * stats.t.sf(abs(t_naive), k - 1)) if k > 1 else float("nan")

    inflation = 1.0 / k + n_test / n_train if k > 1 else float("nan")
    se_nb = math.sqrt(inflation * sd ** 2) if k > 1 else float("nan")
    ci_nb = (mean - t_crit * se_nb, mean + t_crit * se_nb)
    t_nb = mean / se_nb if se_nb else float("nan")
    p_nb = float(2 * stats.t.sf(abs(t_nb), k - 1)) if k > 1 else float("nan")

    t_power = float(stats.t.ppf(POWER_TARGET, k - 1)) if k > 1 else float("nan")
    mde = (t_crit + t_power) * se_nb if k > 1 else float("nan")

    return Comparison(
        family=family,
        table=table,
        dataset=dataset,
        metric=metric,
        label_a=label_a,
        label_b=label_b,
        scores_a=[float(v) for v in a],
        scores_b=[float(v) for v in b],
        n_train=n_train,
        n_test=n_test,
        delta=mean,
        sd=sd,
        sd_pop=sd_pop,
        wins=int(better.sum()),
        k=k,
        naive_ci=ci_naive,
        naive_p=p_naive,
        nadeau_ci=ci_nb,
        nadeau_p=p_nb,
        mde=float(mde),
    )


def holm_adjust(comparisons: list[Comparison]) -> None:
    """Holm step-down adjustment, applied in place, family by family."""
    by_family: dict[str, list[Comparison]] = {}
    for c in comparisons:
        by_family.setdefault(c.family, []).append(c)

    for family, members in by_family.items():
        ordered = sorted(members, key=lambda c: c.nadeau_p)
        m = len(ordered)
        running = 0.0
        for rank, c in enumerate(ordered):
            raw = c.nadeau_p if math.isfinite(c.nadeau_p) else 1.0
            adjusted = min(1.0, (m - rank) * raw)
            running = max(running, adjusted)
            c.holm_p = float(running)


# ---------------------------------------------------------------------------
# table builders
# ---------------------------------------------------------------------------
def build_internal() -> list[Comparison]:
    """Hybrid / graph-only / prior-residual comparisons (Table S: internal)."""
    out: list[Comparison] = []
    for dataset in PRIMARY_METRIC:
        metric = PRIMARY_METRIC[dataset]
        ablation = load_json(f"modularity_ablation_{dataset}.json")
        baselines = load_json(f"baselines_{dataset}_aggregated.json")
        seeds = ablation.get("seeds") or list(range(1, len(ablation["raw_results"]["direct_hybrid"]) + 1))
        n_train, n_test = split_sizes(dataset, seeds)

        configs = {
            name: ordered(
                {seed: rec[metric] for seed, rec in zip(seeds, records)},
                seeds,
            ).tolist()
            for name, records in ablation["raw_results"].items()
        }
        rf_values: dict[int, float] = {}
        for run in baselines["raw_runs"]:
            if run["model"] == "rdkit_rf":
                rf_values[int(run["seed"])] = float(run["metrics"][metric])
        rf = ordered(rf_values, seeds).tolist()

        pairs = [
            ("Hybrid", configs["direct_hybrid"], "RDKit 2D + RF", rf),
            ("Hybrid", configs["direct_hybrid"], "Graph only", configs["residual_only"]),
            ("Prior/residual", configs["prior_residual"], "Graph only", configs["residual_only"]),
            ("Prior/residual", configs["prior_residual"], "Hybrid", configs["direct_hybrid"]),
        ]
        for label_a, a, label_b, b in pairs:
            out.append(
                compare(
                    family=f"internal::{dataset}",
                    table="internal",
                    dataset=dataset,
                    metric=metric,
                    label_a=label_a,
                    label_b=label_b,
                    scores_a=a,
                    scores_b=b,
                    n_train=n_train,
                    n_test=n_test,
                )
            )
    return out


def build_external() -> list[Comparison]:
    """Framework minus Chemprop v2.3.1 on matched splits."""
    out: list[Comparison] = []
    for dataset in PRIMARY_METRIC:
        metric = PRIMARY_METRIC[dataset]
        framework = load_json(f"modularity_ablation_{dataset}.json")
        chemprop = load_json(f"chemprop_v2_{dataset}.json")
        seeds = framework.get("seeds") or list(
            range(1, len(framework["raw_results"]["direct_hybrid"]) + 1)
        )
        n_train, n_test = split_sizes(dataset, seeds)

        ours = {
            name: ordered(
                {seed: rec[metric] for seed, rec in zip(seeds, records)}, seeds
            ).tolist()
            for name, records in framework["raw_results"].items()
        }
        theirs = {
            name: ordered(
                {seed: rec[metric] for seed, rec in zip(seeds, records)}, seeds
            ).tolist()
            for name, records in chemprop["raw_results"].items()
        }

        for label, ours_key, theirs_key in [
            ("Graph only", "residual_only", "chemprop_graph"),
            ("Graph + 210 descriptors", "direct_hybrid", "chemprop_hybrid"),
        ]:
            out.append(
                compare(
                    family=f"external::{dataset}",
                    table="external",
                    dataset=dataset,
                    metric=metric,
                    label_a=f"Framework {label.lower()}",
                    label_b=f"Chemprop v2.3.1 {label.lower()}",
                    scores_a=ours[ours_key],
                    scores_b=theirs[theirs_key],
                    n_train=n_train,
                    n_test=n_test,
                )
            )
    return out


def build_recommended_vs_reference() -> list[Comparison]:
    """Recommended top-K subsets versus random subsets and the full pool."""
    out: list[Comparison] = []
    for dataset in ["esol", "freesolv", "bbbp"]:
        study = load_json(f"feature_study_{dataset}.json")
        metric = study["primary_metric"]
        seeds = study.get("seeds") or list(
            range(1, len(study["raw_results"]["framework_recommendation"]["5"]) + 1)
        )
        n_train, n_test = split_sizes(dataset, seeds)
        for k in study["k_levels"]:
            rec = [rec_[metric] for rec_ in study["raw_results"]["framework_recommendation"][str(k)]]
            rnd = [rec_[metric] for rec_ in study["raw_results"]["random_subsets"][str(k)]]
            out.append(
                compare(
                    family=f"rec_random::{dataset}",
                    table="rec_random",
                    dataset=dataset,
                    metric=metric,
                    label_a=f"Recommended K={k}",
                    label_b=f"Random K={k}",
                    scores_a=rec,
                    scores_b=rnd,
                    n_train=n_train,
                    n_test=n_test,
                )
            )
        pool = [rec_[metric] for rec_ in study["raw_results"]["all_features"]["all"]]
        for k in study["k_levels"]:
            rec = [
                rec_[metric]
                for rec_ in study["raw_results"]["framework_recommendation"][str(k)]
            ]
            out.append(
                compare(
                    family=f"rec_random::{dataset}",
                    table="rec_random",
                    dataset=dataset,
                    metric=metric,
                    label_a=f"Recommended K={k}",
                    label_b="Full 210-descriptor pool",
                    scores_a=rec,
                    scores_b=pool,
                    n_train=n_train,
                    n_test=n_test,
                )
            )
    return out


def build_selectors() -> list[Comparison]:
    """Each selector against the framework selector, per dataset and budget."""
    out: list[Comparison] = []
    for dataset in PRIMARY_METRIC:
        study = load_json(f"multivariate_feature_study_{dataset}.json")
        metric = study["primary_metric"]
        per_seed = study["per_seed_results"]
        seeds = study.get("seeds") or list(range(1, len(per_seed["framework"]["5"]) + 1))
        n_train, n_test = split_sizes(dataset, seeds)
        for k in ["5", "10", "20", "50"]:
            reference = [rec[metric] for rec in per_seed["framework"][k]]
            for selector in ["spearman", "mutual_info", "mrmr", "rf_importance", "l1"]:
                scores = [rec[metric] for rec in per_seed[selector][k]]
                out.append(
                    compare(
                        family=f"selectors::{dataset}",
                        table="selectors",
                        dataset=dataset,
                        metric=metric,
                        label_a=f"{selector} K={k}",
                        label_b=f"framework K={k}",
                        scores_a=scores,
                        scores_b=reference,
                        n_train=n_train,
                        n_test=n_test,
                    )
                )
    return out


def build_routing() -> list[Comparison]:
    """Harmonised descriptor routing, recommender versus every reference route."""
    out: list[Comparison] = []
    for dataset in PRIMARY_METRIC:
        routing = load_json(f"evidence_routing_{dataset}.json")
        metric = routing["primary_metric"]
        summary = routing["summary"]
        seeds = routing.get("seeds") or list(
            range(1, len(summary["hybrid_rec"]["per_split"]) + 1)
        )
        n_train, n_test = split_sizes(dataset, seeds)
        rec = summary["hybrid_rec"]["per_split"]
        for key, label in [
            ("residual_only", "Graph only"),
            ("direct_hybrid", "All 210 descriptors"),
            ("hybrid_spearman", "Spearman control"),
            ("hybrid_mi", "Mutual-information control"),
            ("hybrid_random", "Random subsets"),
        ]:
            out.append(
                compare(
                    family=f"routing::{dataset}",
                    table="routing",
                    dataset=dataset,
                    metric=metric,
                    label_a="Recommender route",
                    label_b=label,
                    scores_a=rec,
                    scores_b=summary[key]["per_split"],
                    n_train=n_train,
                    n_test=n_test,
                )
            )
    return out


def build_sensitivity() -> list[Comparison]:
    """Duplicate-controlled BBBP sensitivity analysis."""
    out: list[Comparison] = []
    study = load_json("bbbp_sensitivity_study.json")
    chemprop = load_json("chemprop_bbbp_sensitivity.json")
    metric = "roc_auc"
    per_split = {k: v["per_split"] for k, v in study["summary"].items()}
    seeds = study.get("seeds") or list(range(1, len(per_split["framework_hybrid"]) + 1))
    n_train, n_test = split_sizes("bbbp_sensitivity", seeds)
    pairs = [
        ("Framework hybrid", per_split["framework_hybrid"], "Framework graph only", per_split["framework_graph_only"]),
        ("Chemprop hybrid", chemprop["summary"]["chemprop_hybrid"]["values"], "Chemprop graph only", chemprop["summary"]["chemprop_graph"]["values"]),
        ("Framework hybrid", per_split["framework_hybrid"], "RDKit 2D + RF", per_split["rdkit_rf"]),
        ("Framework graph only", per_split["framework_graph_only"], "RDKit 2D + RF", per_split["rdkit_rf"]),
        ("Chemprop hybrid", chemprop["summary"]["chemprop_hybrid"]["values"], "RDKit 2D + RF", per_split["rdkit_rf"]),
    ]
    for label_a, a, label_b, b in pairs:
        out.append(
            compare(
                family="sensitivity::bbbp",
                table="sensitivity",
                dataset="bbbp_sensitivity",
                metric=metric,
                label_a=label_a,
                label_b=label_b,
                scores_a=a,
                scores_b=b,
                n_train=n_train,
                n_test=n_test,
            )
        )
    return out


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    global RESULTS_DIR

    parser = argparse.ArgumentParser(
        description="Compute the paired statistics behind the manuscript tables."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory holding the benchmark result JSON files (default: benchmarks/results)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to write paper_statistics.json (default: <results-dir>/paper_statistics.json)",
    )
    args = parser.parse_args(argv)
    RESULTS_DIR = args.results_dir
    out_path = args.out or RESULTS_DIR / "paper_statistics.json"

    builders = [
        build_internal,
        build_external,
        build_recommended_vs_reference,
        build_selectors,
        build_routing,
        build_sensitivity,
    ]
    comparisons: list[Comparison] = []
    for builder in builders:
        comparisons.extend(builder())
    holm_adjust(comparisons)

    payload = {
        "n_comparisons": len(comparisons),
        "primary_interval": "nadeau_bengio_corrected_resampled_t",
        "multiplicity": "holm_step_down_within_family",
        "comparisons": [c.as_dict() for c in comparisons],
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"wrote {len(comparisons)} comparisons to {out_path}")
    print("\ninternal comparisons (naive -> Nadeau-Bengio, Holm-adjusted p):")
    for c in comparisons:
        if c.table != "internal":
            continue
        print(
            f"  {c.dataset:<15} {c.label_a:<15} vs {c.label_b:<15} "
            f"d={c.delta:+.4f} naive_p={c.naive_p:.4f} nb_p={c.nadeau_p:.4f} "
            f"holm_p={c.holm_p:.4f} wins={c.wins}/{c.k}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
