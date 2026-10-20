"""Distribution-free robustness checks for the manuscript statistics.

Three additions to the paired interval analysis in ``paper_statistics.py``:

1. **Exact sign-flip permutation tests.**  For every paired comparison over the
   ten splits the null distribution of the mean paired difference is enumerated
   (2^10 = 1024 sign vectors) rather than approximated by a t distribution.
   Holm adjustment is applied within the same families used by
   ``paper_statistics.py``.
2. **Friedman / Nemenyi omnibus test over selectors.**  The six selectors are
   compared as a randomised block design, with dataset-budget cells as blocks;
   the test gives mean ranks, an omnibus p-value and a Nemenyi critical
   difference, so the "ranked fourth of six" statement carries an uncertainty.
3. **Permutation tests for the routing comparisons** against the matched random
   control, again exact over the ten splits.

Reads only archived result files; writes ``statistical_robustness.json``.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"

DATASETS = ["esol", "freesolv", "lipophilicity", "bbbp", "bace"]
K_LEVELS = [5, 10, 20, 50]

# Nemenyi critical values (studentised range / sqrt(2)) for alpha = 0.05.
# Source: Demšar (2006), "Statistical comparisons of classifiers over multiple
# data sets", Table 5 (q_0.05).
NEMENYI_Q05 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
    8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268,
}


def sign_flip_permutation(a: np.ndarray, b: np.ndarray, n_perm_max: int = 4096) -> tuple[float, float]:
    """Exact (or Monte-Carlo) two-sided sign-flip permutation test."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(d)
    observed = float(np.mean(d))
    if n <= 12:
        counts = np.array([bin(m).count("1") for m in range(2 ** n)])
        sign_matrix = np.array(
            [[1.0 if (m >> bit) & 1 else -1.0 for bit in range(n)] for m in range(2 ** n)]
        )
        perm_means = sign_matrix @ d / n
        p = float(np.mean(np.abs(perm_means) >= abs(observed) - 1e-12))
        return observed, p
    rng = np.random.default_rng(20260101)
    perm_means = np.empty(n_perm_max)
    for i in range(n_perm_max):
        signs = rng.choice([-1.0, 1.0], size=n)
        perm_means[i] = float(np.mean(signs * d))
    p = float((np.sum(np.abs(perm_means) >= abs(observed) - 1e-12) + 1) / (n_perm_max + 1))
    return observed, p


def holm(pairs: list[tuple[str, float]]) -> dict[str, float]:
    """Holm step-down adjustment for a list of (label, raw p) within a family."""
    ordered = sorted(pairs, key=lambda t: t[1])
    m = len(ordered)
    running = 0.0
    out: dict[str, float] = {}
    for rank, (label, raw) in enumerate(ordered):
        value = min(1.0, (m - rank) * (raw if math.isfinite(raw) else 1.0))
        running = max(running, value)
        out[label] = float(running)
    return out


def permutation_tests() -> dict[str, Any]:
    with open(RESULTS_DIR / "paper_statistics.json", "r") as f:
        stats_payload = json.load(f)

    per_comparison: list[dict[str, Any]] = []
    for c in stats_payload["comparisons"]:
        observed, p = sign_flip_permutation(np.array(c["scores_a"]), np.array(c["scores_b"]))
        per_comparison.append({
            "family": c["family"],
            "table": c["table"],
            "dataset": c["dataset"],
            "label_a": c["label_a"],
            "label_b": c["label_b"],
            "delta": c["delta"],
            "wins": c["wins"],
            "k": c["k"],
            "nadeau_p": c["nadeau_p"],
            "holm_p_nadeau": c["holm_p"],
            "p_permutation": p,
            "k_permutations": 2 ** c["k"] if c["k"] <= 12 else None,
        })

    families: dict[str, list[tuple[str, float]]] = {}
    for row, c in zip(per_comparison, stats_payload["comparisons"]):
        key = f"{row['dataset']}|{row['label_a']}|{row['label_b']}|{row['table']}"
        families.setdefault(row["family"], []).append((key, row["p_permutation"]))
    adjusted: dict[str, float] = {}
    for family, members in families.items():
        adjusted.update({f"{family}::{k}": v for k, v in holm(members).items()})
    for row in per_comparison:
        key = f"{row['family']}::{row['dataset']}|{row['label_a']}|{row['label_b']}|{row['table']}"
        row["holm_p_permutation"] = adjusted.get(key)

    raw = np.array([r["p_permutation"] for r in per_comparison])
    nb = np.array([r["holm_p_nadeau"] for r in per_comparison])
    holm_perm = np.array([r["holm_p_permutation"] for r in per_comparison], dtype=float)

    agree = int(np.sum((nb < 0.05) == (holm_perm < 0.05)))
    return {
        "n_comparisons": len(per_comparison),
        "n_significant_holm_nadeau": int(np.sum(nb < 0.05)),
        "n_significant_holm_permutation": int(np.sum(holm_perm < 0.05)),
        "decision_agreement": f"{agree}/{len(per_comparison)}",
        "median_p_permutation": float(np.median(raw)),
        "comparisons": per_comparison,
    }


def selector_block_ranks() -> tuple[dict[str, float], dict[str, Any]]:
    """Mean ranks of the six selectors across the 20 dataset-budget cells."""
    metrics = {
        ds: json.load(open(RESULTS_DIR / f"multivariate_feature_study_{ds}.json", "r"))
        for ds in DATASETS
    }
    selectors = sorted(metrics["esol"]["per_seed_results"].keys())
    cell_ranks: dict[str, list[float]] = {s: [] for s in selectors}
    for ds, payload in metrics.items():
        primary = payload["primary_metric"]
        higher_better = primary == "roc_auc"
        for k in K_LEVELS:
            scores = {
                s: np.array([r[primary] for r in payload["per_seed_results"][s][str(k)]], dtype=float)
                for s in selectors
            }
            means = {s: float(np.mean(v)) for s, v in scores.items()}
            # rank on the mean over splits; 1 = best
            order = sorted(means.items(), key=lambda t: -t[1] if higher_better else t[1])
            for rank, (name, _) in enumerate(order, start=1):
                cell_ranks[name].append(float(rank))

    mean_ranks = {s: float(np.mean(v)) for s, v in cell_ranks.items()}
    matrix = np.array([cell_ranks[s] for s in selectors])  # selectors x cells
    n_blocks = matrix.shape[1]
    fried = stats.friedmanchisquare(*[matrix[i] for i in range(matrix.shape[0])])

    n_methods = len(selectors)
    q = NEMENYI_Q05.get(n_methods, 3.0)
    cd = q * math.sqrt(n_methods * (n_methods + 1) / (6.0 * n_blocks))

    se = math.sqrt(n_methods * (n_methods + 1) / (6.0 * n_blocks))
    pairwise: dict[str, float] = {}
    for a, b in itertools.combinations(selectors, 2):
        diff = abs(mean_ranks[a] - mean_ranks[b])
        q_observed = diff / se
        p = float(stats.studentized_range.sf(q_observed * math.sqrt(2), n_methods, np.inf))
        pairwise[f"{a} vs {b}"] = p

    payload = {
        "selectors": selectors,
        "mean_ranks": mean_ranks,
        "n_blocks": n_blocks,
        "friedman_chi2": float(fried.statistic),
        "friedman_p": float(fried.pvalue),
        "nemenyi_critical_difference_alpha_0.05": cd,
        "nemenyi_q05": q,
        "rank_matrix": {s: cell_ranks[s] for s in selectors},
        "pairwise_z_unadjusted": pairwise,
    }
    return mean_ranks, payload


def routing_permutation_tests() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for ds in DATASETS:
        payload = json.load(open(RESULTS_DIR / f"evidence_routing_{ds}.json", "r"))
        is_cls = payload["is_classification"]
        summary = payload["summary"]

        def oriented(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """Per-split difference oriented so that positive means a is better."""
            return (a - b) if is_cls else (b - a)

        rows: dict[str, Any] = {}
        for route in ["hybrid_rec", "hybrid_spearman", "hybrid_mi"]:
            if route not in summary:
                continue
            a = np.array(summary[route]["per_split"], dtype=float)
            b = np.array(summary["hybrid_random"]["per_split"], dtype=float)
            diff = oriented(a, b)
            _, p = sign_flip_permutation(diff, np.zeros_like(diff))
            rows[route] = {
                "mean_route": float(np.mean(a)),
                "mean_random": float(np.mean(b)),
                "delta_positive_is_better": float(np.mean(diff)),
                "p_permutation": p,
                "wins": int(np.sum(diff > 0)),
            }
        if "hybrid_rec" in summary:
            a = np.array(summary["hybrid_rec"]["per_split"], dtype=float)
            b = np.array(summary["direct_hybrid"]["per_split"], dtype=float)
            diff = oriented(a, b)
            _, p = sign_flip_permutation(diff, np.zeros_like(diff))
            rows["hybrid_rec_vs_all210"] = {
                "mean_route": float(np.mean(a)),
                "mean_all210": float(np.mean(b)),
                "delta_positive_is_better": float(np.mean(diff)),
                "p_permutation": p,
                "wins": int(np.sum(diff > 0)),
            }
        out[ds] = rows

    # Holm within the "route versus random control" family (all datasets, six routes)
    control_routes = {"hybrid_rec", "hybrid_spearman", "hybrid_mi"}
    raw_pairs = [
        (f"{ds}|{route}", row["p_permutation"])
        for ds, rows in out.items() for route, row in rows.items() if route in control_routes
    ]
    adj = holm(raw_pairs)
    for ds, rows in out.items():
        for route, row in rows.items():
            if route in control_routes:
                row["holm_p_permutation_within_family"] = adj.get(f"{ds}|{route}")
    return out


def random_control_summary() -> dict[str, Any]:
    """Aggregate the high-repetition rank tests, when they have been produced."""
    out: dict[str, Any] = {}
    cells_sig = 0
    cells_total = 0
    for ds in DATASETS:
        path = RESULTS_DIR / f"random_control_{ds}.json"
        if not path.exists():
            continue
        payload = json.load(open(path, "r"))
        better = 1 if payload["is_classification"] else -1
        rows: dict[str, Any] = {}
        for k, cell in payload["summary"].items():
            delta = better * (cell["recommended_mean"] - cell["random_ten_draw_mean"])
            rows[k] = {
                "recommended_mean": cell["recommended_mean"],
                "random_ten_draw_mean": cell["random_ten_draw_mean"],
                "delta_positive_is_better": delta,
                "p_signflip": cell["p_signflip_two_sided"],
                "mean_rank_percentile": cell["mean_rank_percentile"],
                "min_rank_percentile": cell["min_rank_percentile"],
                "p_rank_fisher": cell["p_rank_fisher"],
                "draws_per_cell": payload["draws_per_cell"],
            }
            cells_total += 1
            if cell["p_rank_fisher"] < 0.05:
                cells_sig += 1
        out[ds] = rows
    return {
        "cells_total": cells_total,
        "cells_with_fisher_p_below_0.05": cells_sig,
        "datasets": out,
    }


def main() -> None:
    perm = permutation_tests()
    _, friedman = selector_block_ranks()
    routing = routing_permutation_tests()
    random_ctl = random_control_summary()

    payload = {
        "description": "Distribution-free robustness checks (permutation, Friedman/Nemenyi).",
        "permutation_tests": {
            k: v for k, v in perm.items() if k != "comparisons"
        },
        "permutation_comparisons": perm["comparisons"],
        "selector_friedman": friedman,
        "routing_permutation": routing,
        "random_control_rank_tests": random_ctl,
    }
    out_path = RESULTS_DIR / "statistical_robustness.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print("permutation tests: "
          f"{perm['n_comparisons']} comparisons, "
          f"{perm['n_significant_holm_nadeau']} significant (Nadeau-Bengio Holm), "
          f"{perm['n_significant_holm_permutation']} significant (permutation Holm), "
          f"agreement {perm['decision_agreement']}")
    print("selector Friedman: chi2={:.2f} p={:.4g}, Nemenyi CD={:.3f} ranks".format(
        friedman["friedman_chi2"], friedman["friedman_p"],
        friedman["nemenyi_critical_difference_alpha_0.05"]))
    print("mean ranks: " + ", ".join(f"{k}={v:.2f}" for k, v in sorted(
        friedman["mean_ranks"].items(), key=lambda t: t[1])))
    print("routing vs random (permutation p):")
    for ds, rows in routing.items():
        for route, row in rows.items():
            if route in {"hybrid_rec", "hybrid_spearman", "hybrid_mi"}:
                print(f"  {ds:14s} {route:18s} delta={row['delta_positive_is_better']:+.4f} "
                      f"p={row['p_permutation']:.3f} holm={row.get('holm_p_permutation_within_family', float('nan')):.3f}")
    if random_ctl["cells_total"]:
        print(f"random control rank tests: {random_ctl['cells_with_fisher_p_below_0.05']}/"
              f"{random_ctl['cells_total']} cells with Fisher p < 0.05")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
