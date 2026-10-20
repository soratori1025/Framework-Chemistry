"""Validate ``paper_statistics.json``.

Two families of checks are available:

``published``
    The 30 values below are transcribed from the supplementary tables of an
    earlier five-split run of this project (internal paired comparisons and
    external comparisons). That run is **not** reported in the manuscript; the
    values are kept only as a frozen regression test for the statistics
    pipeline. Run this mode against the archived snapshot to confirm that the
    code still reproduces those numbers::

        python benchmarks/validate_paper_statistics.py \
            --statistics <archive>/paper_statistics.json --mode published

    The released ``benchmarks/results/paper_statistics.json`` holds the
    ten-split results, so point ``--statistics`` at the archived five-split
    snapshot when you want to check the old values themselves.

``consistency``
    Recomputation checks that hold for any release: every comparison is
    recomputed from its stored per-split scores (delta, SDs, wins), the naive
    and Nadeau--Bengio intervals and p-values are recomputed from the stored
    split sizes, the minimum detectable difference is recomputed from the
    corrected standard error, and the Holm step-down adjustment is recomputed
    from the raw corrected p-values within each comparison family.  This is the
    mode to run on the ten-split statistics file.

``auto`` (default)
    Runs ``published`` when the file holds the five-split reference comparisons
    (``k == 5`` throughout), otherwise runs ``consistency`` only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATISTICS = REPO_ROOT / "benchmarks" / "results" / "paper_statistics.json"
POWER_TARGET = 0.80
LOWER_IS_BETTER = {"rmse", "mae", "loss"}

# (dataset, label_a, label_b, delta, sd_pop, naive_ci_lo, naive_ci_hi, wins)
PUBLISHED = [
    ("esol", "Hybrid", "RDKit 2D + RF", -0.0524, 0.0335, -0.099, -0.006, 5),
    ("esol", "Hybrid", "Graph only", -0.0579, 0.0337, -0.105, -0.011, 4),
    ("esol", "Prior/residual", "Graph only", +0.0124, 0.0919, -0.115, +0.140, 2),
    ("esol", "Prior/residual", "Hybrid", +0.0703, 0.0969, -0.064, +0.205, 1),
    ("freesolv", "Hybrid", "RDKit 2D + RF", -0.0691, 0.1086, -0.220, +0.082, 3),
    ("freesolv", "Hybrid", "Graph only", -0.1346, 0.1619, -0.359, +0.090, 4),
    ("freesolv", "Prior/residual", "Graph only", -0.0479, 0.1680, -0.281, +0.185, 3),
    ("freesolv", "Prior/residual", "Hybrid", +0.0867, 0.1870, -0.173, +0.346, 2),
    ("lipophilicity", "Hybrid", "RDKit 2D + RF", -0.1468, 0.0210, -0.176, -0.118, 5),
    ("lipophilicity", "Hybrid", "Graph only", +0.0178, 0.0429, -0.042, +0.077, 2),
    ("lipophilicity", "Prior/residual", "Graph only", +0.0029, 0.0214, -0.027, +0.033, 2),
    ("lipophilicity", "Prior/residual", "Hybrid", -0.0149, 0.0427, -0.074, +0.044, 2),
    ("bbbp", "Hybrid", "RDKit 2D + RF", -0.0024, 0.0161, -0.025, +0.020, 3),
    ("bbbp", "Hybrid", "Graph only", -0.0084, 0.0332, -0.055, +0.038, 2),
    ("bbbp", "Prior/residual", "Graph only", +0.0079, 0.0239, -0.025, +0.041, 3),
    ("bbbp", "Prior/residual", "Hybrid", +0.0163, 0.0358, -0.033, +0.066, 4),
    ("bace", "Hybrid", "RDKit 2D + RF", -0.0293, 0.0151, -0.050, -0.008, 0),
    ("bace", "Hybrid", "Graph only", +0.0108, 0.0396, -0.044, +0.066, 3),
    ("bace", "Prior/residual", "Graph only", +0.0031, 0.0359, -0.047, +0.053, 3),
    ("bace", "Prior/residual", "Hybrid", -0.0077, 0.0153, -0.029, +0.014, 1),
    ("esol", "Framework graph only", "Chemprop v2.3.1 graph only", -0.1191, 0.0996, -0.257, +0.019, 4),
    ("esol", "Framework graph + 210 descriptors", "Chemprop v2.3.1 graph + 210 descriptors", -0.0314, 0.0994, -0.169, +0.107, 3),
    ("freesolv", "Framework graph only", "Chemprop v2.3.1 graph only", +0.0419, 0.1274, -0.135, +0.219, 2),
    ("freesolv", "Framework graph + 210 descriptors", "Chemprop v2.3.1 graph + 210 descriptors", -0.0408, 0.1338, -0.226, +0.145, 3),
    ("lipophilicity", "Framework graph only", "Chemprop v2.3.1 graph only", -0.0182, 0.0269, -0.056, +0.019, 3),
    ("lipophilicity", "Framework graph + 210 descriptors", "Chemprop v2.3.1 graph + 210 descriptors", -0.0351, 0.0557, -0.112, +0.042, 3),
    ("bbbp", "Framework graph only", "Chemprop v2.3.1 graph only", -0.0006, 0.0170, -0.024, +0.023, 3),
    ("bbbp", "Framework graph + 210 descriptors", "Chemprop v2.3.1 graph + 210 descriptors", -0.0029, 0.0208, -0.032, +0.026, 2),
    ("bace", "Framework graph only", "Chemprop v2.3.1 graph only", -0.0010, 0.0407, -0.058, +0.056, 1),
    ("bace", "Framework graph + 210 descriptors", "Chemprop v2.3.1 graph + 210 descriptors", -0.0038, 0.0153, -0.025, +0.017, 2),
]


def check_published(comps: list[dict], args) -> int:
    """Compare the transcribed supplementary values with the statistics file."""
    index = {(c["dataset"], c["label_a"], c["label_b"]): c for c in comps}
    problems = 0
    for ds, la, lb, delta, sd, lo, hi, wins in PUBLISHED:
        c = index.get((ds, la, lb))
        if c is None:
            print(f"MISSING {ds} {la} vs {lb}")
            problems += 1
            continue
        issues = []
        if abs(c["delta"] - delta) > args.delta_tol:
            issues.append(f"delta {c['delta']:+.4f} != {delta:+.4f}")
        if abs(c["sd_pop"] - sd) > args.delta_tol:
            issues.append(f"sd_pop {c['sd_pop']:.4f} != {sd:.4f}")
        if abs(c["naive_ci"][0] - lo) > args.ci_tol:
            issues.append(f"naive_lo {c['naive_ci'][0]:+.3f} != {lo:+.3f}")
        if abs(c["naive_ci"][1] - hi) > args.ci_tol:
            issues.append(f"naive_hi {c['naive_ci'][1]:+.3f} != {hi:+.3f}")
        if c["wins"] != wins:
            issues.append(f"wins {c['wins']} != {wins}")
        if issues:
            problems += 1
            print(f"FAIL {ds:<15} {la:<45} {'; '.join(issues)}")
        elif not args.quiet:
            print(f"OK   {ds:<15} {la:<45}")
    print(f"\npublished five-split checks: {len(PUBLISHED) - problems}/{len(PUBLISHED)} reproduced")
    return problems


def check_consistency(comps: list[dict], args) -> int:
    """Recompute every derived quantity from the stored scores and split sizes."""
    checks = 0
    failures = 0

    def fail(record: dict, message: str) -> None:
        nonlocal failures
        failures += 1
        print(f"FAIL {record['dataset']:<15} {record['label_a']} vs {record['label_b']}: {message}")

    def close(a: float, b: float, tol: float) -> bool:
        return abs(a - b) <= tol + 1e-9 * max(abs(a), abs(b))

    for record in comps:
        scores_a = np.asarray(record["scores_a"], dtype=float)
        scores_b = np.asarray(record["scores_b"], dtype=float)
        k = int(record["k"])
        if scores_a.shape != scores_b.shape or scores_a.size != k:
            fail(record, f"score arrays {scores_a.shape}/{scores_b.shape} inconsistent with k={k}")
            continue
        if args.expect_k is not None and k != args.expect_k:
            fail(record, f"k={k} != expected {args.expect_k}")

        d = scores_a - scores_b
        delta = float(d.mean())
        sd = float(d.std(ddof=1)) if k > 1 else 0.0
        sd_pop = float(d.std(ddof=0))
        lower = record["metric"] in LOWER_IS_BETTER
        wins = int((d < 0).sum() if lower else (d > 0).sum())

        if not close(delta, record["delta"], args.delta_tol):
            fail(record, f"delta {record['delta']:+.6f} != recomputed {delta:+.6f}")
        if not close(sd, record["sd"], args.delta_tol):
            fail(record, f"sd {record['sd']:.6f} != recomputed {sd:.6f}")
        if not close(sd_pop, record["sd_pop"], args.delta_tol):
            fail(record, f"sd_pop {record['sd_pop']:.6f} != recomputed {sd_pop:.6f}")
        if wins != record["wins"]:
            fail(record, f"wins {record['wins']} != recomputed {wins}")
        checks += 4

        if k <= 1:
            continue
        n_train, n_test = float(record["n_train"]), float(record["n_test"])
        t_crit = float(stats.t.ppf(0.975, k - 1))
        t_power = float(stats.t.ppf(POWER_TARGET, k - 1))

        se_naive = sd / math.sqrt(k)
        ci_naive = (delta - t_crit * se_naive, delta + t_crit * se_naive)
        p_naive = float(2 * stats.t.sf(abs(delta / se_naive), k - 1))

        inflation = 1.0 / k + n_test / n_train
        se_nb = math.sqrt(inflation * sd ** 2)
        ci_nb = (delta - t_crit * se_nb, delta + t_crit * se_nb)
        p_nb = float(2 * stats.t.sf(abs(delta / se_nb), k - 1))
        mde = (t_crit + t_power) * se_nb

        for name, model, stored in (
            ("naive_ci", ci_naive, record["naive_ci"]),
            ("nadeau_ci", ci_nb, record["nadeau_ci"]),
        ):
            if not (close(model[0], stored[0], args.delta_tol) and close(model[1], stored[1], args.delta_tol)):
                fail(record, f"{name} [{stored[0]:+.6f},{stored[1]:+.6f}] != recomputed "
                            f"[{model[0]:+.6f},{model[1]:+.6f}]")
            checks += 1
        for name, model, stored in (
            ("naive_p", p_naive, record["naive_p"]),
            ("nadeau_p", p_nb, record["nadeau_p"]),
        ):
            if not close(model, stored, args.p_tol):
                fail(record, f"{name} {stored:.3e} != recomputed {model:.3e}")
            checks += 1
        if not close(mde, record["mde"], args.delta_tol):
            fail(record, f"mde {record['mde']:.6f} != recomputed {mde:.6f}")
        checks += 1
        if not (ci_nb[1] - ci_nb[0] >= ci_naive[1] - ci_naive[0] - 1e-9):
            fail(record, "corrected interval narrower than the naive interval")
        checks += 1

    # Holm step-down recomputed within each family from the stored corrected p-values.
    families: dict[str, list[dict]] = {}
    for record in comps:
        families.setdefault(record["family"], []).append(record)
    for family, members in families.items():
        ordered = sorted(members, key=lambda c: c["nadeau_p"])
        m = len(ordered)
        running = 0.0
        for rank, record in enumerate(ordered):
            raw = record["nadeau_p"] if math.isfinite(record["nadeau_p"]) else 1.0
            running = max(running, min(1.0, (m - rank) * raw))
            stored = record.get("holm_p")
            if stored is None:
                fail(record, f"holm_p missing in family {family}")
            elif not close(running, stored, args.p_tol):
                fail(record, f"holm_p {stored:.6f} != recomputed {running:.6f} (family {family})")
            checks += 1

    k_values = sorted({int(c["k"]) for c in comps})
    k_report = k_values[0] if len(k_values) == 1 else k_values
    print(
        f"\nconsistency checks: {len(comps)} comparisons, k={k_report}, "
        f"{len(families)} families, {checks} recomputed quantities, {failures} failures"
    )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate paper_statistics.json (published reference values and/or internal consistency)."
    )
    parser.add_argument(
        "--statistics",
        type=Path,
        default=DEFAULT_STATISTICS,
        help="paper_statistics.json to validate (default: benchmarks/results/paper_statistics.json)",
    )
    parser.add_argument("--mode", choices=["auto", "published", "consistency"], default="auto")
    parser.add_argument("--expect-k", type=int, default=None, help="require this number of splits per comparison")
    parser.add_argument("--delta-tol", type=float, default=5e-5)
    parser.add_argument("--ci-tol", type=float, default=1.5e-3)
    parser.add_argument("--p-tol", type=float, default=1e-6)
    parser.add_argument("--quiet", action="store_true", help="Only print failures and the summary")
    args = parser.parse_args()

    if not args.statistics.is_file():
        print(f"missing statistics file: {args.statistics}")
        return 2
    data = json.loads(args.statistics.read_text(encoding="utf-8"))
    comps = data["comparisons"]

    k_values = {int(c["k"]) for c in comps}
    published_keys = {(ds, la, lb) for ds, la, lb, *_ in PUBLISHED}
    file_keys = {(c["dataset"], c["label_a"], c["label_b"]) for c in comps}
    has_published = published_keys <= file_keys
    run_published = args.mode == "published" or (args.mode == "auto" and has_published and k_values == {5})

    print(f"statistics file: {args.statistics} ({len(comps)} comparisons)")
    problems = 0
    if run_published:
        problems += check_published(comps, args)
    else:
        print("published five-split checks: skipped (file does not hold the five-split reference comparisons)")
    if args.mode != "published":
        problems += check_consistency(comps, args)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
