"""Compare a rerun against an archived snapshot of the benchmark results.

Every runner writes one JSON artefact per endpoint.  This script pairs the
archived artefact (the copy that the statistics were computed from)
with the rerun and reports, per model configuration, how far the rerun drifted
from the archived values for the seeds that both runs share.

The comparison is the reproduction check for the manuscript: it answers
"does a fresh environment reproduce the archived numbers, and by how much does a
retrained neural model move?" — which is what the minimum detectable difference
table in the paper is about.

Usage::

    python benchmarks/compare_split_runs.py --archive <dir> [--new benchmarks/results]
    python benchmarks/compare_split_runs.py --archive <dir> --seeds 1 2 3 4 5 --tol 1e-9

Read the shapes of the artefacts from the runners; a missing artefact in either
directory is reported as "skipped" rather than silently ignored.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NEW = REPO_ROOT / "benchmarks" / "results"

DATASETS = ("esol", "freesolv", "lipophilicity", "bbbp", "bace")


def load(path: Path) -> Any | None:
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def seed_list(payload: dict[str, Any], n: int) -> list[int]:
    seeds = payload.get("seeds")
    if isinstance(seeds, list) and seeds:
        return [int(s) for s in seeds]
    return list(range(1, n + 1))


# --------------------------------------------------------------------------
# readers: artefact -> {(configuration, seed): {metric: value}}
# --------------------------------------------------------------------------
def read_baselines(payload: dict[str, Any]) -> dict[tuple[str, int], dict[str, float]]:
    out: dict[tuple[str, int], dict[str, float]] = {}
    for run in payload.get("raw_runs", []):
        out[(run["model"], int(run["seed"]))] = {
            k: float(v) for k, v in run["metrics"].items()
        }
    return out


def _per_seed_list(
    payload: dict[str, Any], entries: dict[str, list[dict[str, float]]], label: str
) -> dict[tuple[str, int], dict[str, float]]:
    out: dict[tuple[str, int], dict[str, float]] = {}
    for name, records in entries.items():
        seeds = seed_list(payload, len(records))
        for seed, record in zip(seeds, records):
            out[(f"{label}{name}", seed)] = {k: float(v) for k, v in record.items()}
    return out


def read_modularity(payload: dict[str, Any]) -> dict[tuple[str, int], dict[str, float]]:
    return _per_seed_list(payload, payload["raw_results"], "")


def read_chemprop_v2(payload: dict[str, Any]) -> dict[tuple[str, int], dict[str, float]]:
    return _per_seed_list(payload, payload["raw_results"], "")


def read_feature_study(payload: dict[str, Any]) -> dict[tuple[str, int], dict[str, float]]:
    out: dict[tuple[str, int], dict[str, float]] = {}
    for method, by_k in payload["raw_results"].items():
        for k, records in by_k.items():
            seeds = seed_list(payload, len(records))
            for seed, record in zip(seeds, records):
                out[(f"{method}|K={k}", seed)] = {m: float(v) for m, v in record.items()}
    return out


def read_multivariate(payload: dict[str, Any]) -> dict[tuple[str, int], dict[str, float]]:
    out: dict[tuple[str, int], dict[str, float]] = {}
    for selector, by_k in payload["per_seed_results"].items():
        for k, records in by_k.items():
            seeds = seed_list(payload, len(records))
            for seed, record in zip(seeds, records):
                out[(f"{selector}|K={k}", seed)] = {m: float(v) for m, v in record.items()}
    return out


def read_routing(payload: dict[str, Any]) -> dict[tuple[str, int], dict[str, float]]:
    out: dict[tuple[str, int], dict[str, float]] = {}
    for route, block in payload["summary"].items():
        per_split = block.get("per_split")
        if not per_split:
            continue
        seeds = seed_list(payload, len(per_split))
        for seed, value in zip(seeds, per_split):
            out[(route, seed)] = {"primary": float(value)}
    return out


def _per_split_summary(
    payload: dict[str, Any], key: str
) -> dict[tuple[str, int], dict[str, float]]:
    out: dict[tuple[str, int], dict[str, float]] = {}
    for model, block in payload["summary"].items():
        values = block.get(key)
        if not values:
            continue
        seeds = seed_list(payload, len(values))
        for seed, value in zip(seeds, values):
            out[(model, seed)] = {"primary": float(value)}
    return out


READERS: dict[str, Callable[[dict[str, Any]], dict[tuple[str, int], dict[str, float]]]] = {
    "baselines": read_baselines,
    "modularity": read_modularity,
    "feature_study": read_feature_study,
    "multivariate": read_multivariate,
    "routing": read_routing,
    "chemprop_v2": read_chemprop_v2,
}


def artefacts(results_dir: Path) -> list[tuple[str, str, Path, Callable]]:
    """(family, endpoint, path, reader) for every comparable artefact."""
    found: list[tuple[str, str, Path, Callable]] = []
    for dataset in DATASETS:
        found.append(
            ("baselines", dataset, results_dir / f"baselines_{dataset}_aggregated.json", read_baselines)
        )
        found.append(
            ("modularity", dataset, results_dir / f"modularity_ablation_{dataset}.json", read_modularity)
        )
        found.append(
            ("chemprop_v2", dataset, results_dir / f"chemprop_v2_{dataset}.json", read_chemprop_v2)
        )
        found.append(
            ("multivariate", dataset, results_dir / f"multivariate_feature_study_{dataset}.json", read_multivariate)
        )
        found.append(
            ("routing", dataset, results_dir / f"evidence_routing_{dataset}.json", read_routing)
        )
    for dataset in ("esol", "freesolv", "bbbp", "lipophilicity"):
        path = results_dir / f"feature_study_{dataset}.json"
        if path.is_file():
            found.append(("feature_study", dataset, path, read_feature_study))
    found.append(
        (
            "bbbp_sensitivity",
            "bbbp_sensitivity",
            results_dir / "bbbp_sensitivity_study.json",
            lambda payload: _per_split_summary(payload, "per_split"),
        )
    )
    found.append(
        (
            "chemprop_bbbp_sensitivity",
            "bbbp_sensitivity",
            results_dir / "chemprop_bbbp_sensitivity.json",
            lambda payload: _per_split_summary(payload, "values"),
        )
    )
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path, required=True, help="Directory with the archived results")
    parser.add_argument("--new", type=Path, default=DEFAULT_NEW, help="Directory with the rerun results")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Seeds to compare (default: overlap)")
    parser.add_argument("--tol", type=float, default=1e-9, help="Absolute tolerance for 'identical'")
    parser.add_argument("--top", type=int, default=6, help="How many largest deltas to print per artefact")
    args = parser.parse_args()

    overall_values = 0
    overall_exact = 0
    worst: list[tuple[float, str]] = []

    for family, dataset, path, reader in artefacts(args.new):
        archived_path = args.archive / path.name
        if not archived_path.is_file():
            print(f"[skip] {path.name}: not in the archive")
            continue
        if not path.is_file():
            print(f"[skip] {path.name}: rerun artefact absent (not run yet?)")
            continue

        old = reader(load(archived_path))
        new = reader(load(path))
        overlap = sorted({seed for _, seed in old} & {seed for _, seed in new})
        if args.seeds is not None:
            overlap = [s for s in overlap if s in set(args.seeds)]
        if not overlap:
            print(f"[skip] {path.name}: no shared seeds")
            continue

        deltas: list[tuple[float, str, str, float, float]] = []
        n_values = 0
        n_exact = 0
        for key in sorted({k for k in old if k in new}):
            config, seed = key
            if seed not in overlap:
                continue
            for metric, value in old[key].items():
                if metric not in new[key]:
                    continue
                other = new[key][metric]
                diff = abs(other - value)
                n_values += 1
                if diff <= args.tol:
                    n_exact += 1
                deltas.append((diff, config, metric, value, other))

        overall_values += n_values
        overall_exact += n_exact
        deltas.sort(reverse=True)
        print(f"\n=== {family} / {dataset} ({path.name})")
        print(
            f"    shared seeds {overlap} | configs {len({c for c, _ in old if c in {c2 for c2, _ in new}})} "
            f"| values compared {n_values} | identical {n_exact}"
        )
        if deltas:
            print(f"    max |delta| = {deltas[0][0]:.5f}")
            for diff, config, metric, value, other in deltas[: args.top]:
                print(f"      {config:<28} {metric:<8} archived={value:.5f} new={other:.5f} delta={other - value:+.5f}")
            worst.append((deltas[0][0], f"{family}/{dataset}:{deltas[0][1]}/{deltas[0][2]}"))

    print("\n" + "=" * 78)
    print(f"TOTAL values compared: {overall_values} | identical within {args.tol:g}: {overall_exact} "
          f"({100.0 * overall_exact / max(overall_values, 1):.1f}%)")
    if worst:
        worst.sort(reverse=True)
        print("largest drift seen:")
        for diff, where in worst[:8]:
            print(f"  {diff:.5f}  {where}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
