"""Single-entry reproduction script for the manuscript benchmark suite.

Every number in the manuscript is produced by the stages below, in this order.
Each stage writes into ``benchmarks/results`` and is skipped if its outputs are
already present, so the script can be re-run after an interruption.

    python reproduce.py --stage all            # everything that runs in this environment
    python reproduce.py --stage data splits    # only prepare inputs
    python reproduce.py --stage stats figures  # recompute tables and figures
    python reproduce.py --list                 # show stage names

The Chemprop comparator, its duplicate-controlled sensitivity analysis and the
Chemprop arm of the divergence diagnosis need an environment in which
``import chemprop`` succeeds; run those with that interpreter (see
``REPRODUCING.md``):

    <chemprop-env-python> reproduce.py --stage chemprop chemprop-sensitivity
    <chemprop-env-python> reproduce.py --stage chemprop-divergence

Environments: see ``environment.yml`` and ``requirements-lock.txt``.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PY = sys.executable

DATASETS = ["esol", "freesolv", "lipophilicity", "bbbp", "bace"]

STAGES: dict[str, list[list[str]]] = {
    "data": [
        ["benchmarks/prepare_moleculenet_data.py"],
    ],
    "baselines": [
        ["benchmarks/run_canonical_baselines.py", "--dataset", "all"],
    ],
    "feature": [
        ["benchmarks/run_multivariate_feature_study.py", "--dataset", name] for name in DATASETS
    ] + [
        ["benchmarks/run_feature_evidence_study.py", "--dataset", name]
        for name in ["esol", "freesolv", "bbbp"]
    ],
    "random-control": [
        ["benchmarks/run_random_control_study.py", "--dataset", "all"],
    ],
    "ablation": [
        ["benchmarks/run_modularity_ablation.py", "--dataset", name] for name in DATASETS
    ],
    "routing": [
        ["benchmarks/run_evidence_routing_experiment.py", "--dataset", name] for name in DATASETS
    ],
    "chemprop": [
        ["benchmarks/run_chemprop_v2_benchmarks.py", "--dataset", "all"],
    ],
    "sensitivity": [
        ["benchmarks/run_bbbp_sensitivity_study.py"],
    ],
    "chemprop-sensitivity": [
        ["benchmarks/run_chemprop_bbbp_sensitivity.py"],
    ],
    "chemprop-divergence": [
        ["benchmarks/diagnose_esol_divergence.py", "--part", "chemprop",
         "--seeds", "3", "103", "203", "303", "403", "503"],
        ["benchmarks/diagnose_esol_divergence.py", "--part", "chemprop",
         "--seeds", "3", "103", "203", "--clip-z", "5"],
        ["benchmarks/diagnose_esol_divergence.py", "--part", "chemprop",
         "--seeds", "303", "403", "503", "--clip-z", "5", "--out-suffix", "_badseeds"],
    ],
    "audit": [
        ["benchmarks/run_structure_audit.py"],
    ],
    "seed-variance": [
        ["benchmarks/run_seed_variance_decomposition.py", "--dataset", "bbbp", "--split-seed", "1"],
    ],
    "stats": [
        ["benchmarks/paper_statistics.py"],
        ["benchmarks/statistical_robustness.py"],
        ["benchmarks/rebuild_routing_audit.py"],
        ["benchmarks/generate_manuscript_tables.py"],
        ["benchmarks/generate_si_tables.py"],
        ["benchmarks/generate_robustness_tables.py"],
        ["benchmarks/generate_sensitivity_tables.py"],
        ["benchmarks/validate_paper_statistics.py"],
    ],
    "figures": [
        ["benchmarks/make_manuscript_figures.py"],
    ],
    "divergence": [
        ["benchmarks/diagnose_esol_divergence.py", "--part", "extrapolation"],
        ["benchmarks/diagnose_esol_divergence.py", "--part", "framework"],
    ],
    "sensitivity-hp": [
        ["benchmarks/run_hyperparameter_sensitivity.py", "--dataset", "freesolv",
         "--splits", "1", "2", "3", "--hidden-dims", "64", "128", "256", "--threads", "3"],
    ],
    "redundancy": [
        ["benchmarks/run_redundancy_sensitivity.py", "--dataset", "esol", "freesolv", "bbbp",
         "--k", "20"],
    ],
}

ALL_LOCAL = [
    "data", "baselines", "feature", "random-control", "ablation", "routing",
    "sensitivity", "audit", "seed-variance", "sensitivity-hp", "redundancy",
    "stats", "figures", "divergence",
]


def run(command: list[str]) -> None:
    print(f"\n>>> {' '.join(command)}", flush=True)
    result = subprocess.run([PY, *command], cwd=REPO_ROOT)
    if result.returncode != 0:
        raise SystemExit(f"stage failed: {' '.join(command)} (exit {result.returncode})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", nargs="+", default=["all"],
                        help="stage names, or 'all' for every stage that runs locally")
    parser.add_argument("--list", action="store_true", help="list stage names and exit")
    args = parser.parse_args()

    if args.list:
        for name in STAGES:
            print(f"  {name:22s} {len(STAGES[name])} command(s)")
        return

    selected = ALL_LOCAL if args.stage == ["all"] else args.stage
    unknown = [s for s in selected if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s): {', '.join(unknown)} (see --list)")

    for stage in selected:
        for command in STAGES[stage]:
            run(command)
    print("\nall requested stages completed")


if __name__ == "__main__":
    main()
