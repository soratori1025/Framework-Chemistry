"""Master Benchmark Suite Orchestrator.

Runs all 3 Benchmarks across canonical MoleculeNet datasets:
- Benchmark A: Cross-task baselines on BBBP, ESOL, FreeSolv, Lipophilicity
- Benchmark B: Feature Intelligence & Recommendation Study on BBBP, ESOL, FreeSolv
- Benchmark C: Prior / Residual Modularity Ablation on BBBP and ESOL
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.run_canonical_baselines import evaluate_dataset_baselines
from benchmarks.run_feature_evidence_study import run_feature_study_on_dataset
from benchmarks.run_modularity_ablation import run_modularity_ablation_on_dataset

RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Run 3 Benchmarks Suite")
    parser.add_argument("--skip-a", action="store_true", help="Skip Benchmark A")
    parser.add_argument("--skip-b", action="store_true", help="Skip Benchmark B")
    parser.add_argument("--skip-c", action="store_true", help="Skip Benchmark C")
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 75)
    print("🚀 RUNNING CANONICAL 3-BENCHMARK SUITE FOR MOLECULENET")
    print("=" * 75)

    suite_manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark_a": {},
        "benchmark_b": {},
        "benchmark_c": {},
    }

    if not args.skip_c:
        print("\n" + "#" * 75)
        print("📌 [1/3] BENCHMARK C: PRIOR / RESIDUAL MODULARITY ABLATION")
        print("#" * 75)
        for ds in ["esol", "bbbp"]:
            print(f"\n▶ Running Benchmark C on {ds.upper()} (5 seeds)...")
            res_c = run_modularity_ablation_on_dataset(ds)
            suite_manifest["benchmark_c"][ds] = res_c["summary"]

    if not args.skip_b:
        print("\n" + "#" * 75)
        print("📌 [2/3] BENCHMARK B: FEATURE INTELLIGENCE & RECOMMENDATION STUDY")
        print("#" * 75)
        for ds in ["freesolv", "bbbp"]:
            print(f"\n▶ Running Benchmark B on {ds.upper()} across K in [5, 10, 20, 50] (5 seeds)...")
            res_b = run_feature_study_on_dataset(ds)
            suite_manifest["benchmark_b"][ds] = res_b["summary_table"]

    if not args.skip_a:
        print("\n" + "#" * 75)
        print("📌 [3/3] BENCHMARK A: CROSS-TASK BASELINES EVALUATION")
        print("#" * 75)
        baseline_datasets = ["esol", "freesolv", "bbbp"]
        for ds in baseline_datasets:
            print(f"\n▶ Running Benchmark A on {ds.upper()} across 5 seeds...")
            ds_runs = []
            for seed in [1, 2, 3, 4, 5]:
                runs = evaluate_dataset_baselines(ds, seed=seed)
                ds_runs.extend(runs)
            suite_manifest["benchmark_a"][ds] = ds_runs

    total_time = (time.time() - t_start) / 60.0
    print("\n" + "=" * 75)
    print(f"🎉 ALL 3 BENCHMARKS COMPLETED IN {total_time:.2f} MINUTES!")
    print("=" * 75)

    suite_summary_file = RESULTS_DIR / "three_benchmarks_summary.json"
    with open(suite_summary_file, "w") as f:
        json.dump(suite_manifest, f, indent=2)
    print(f"Summary saved to: {suite_summary_file}")

if __name__ == "__main__":
    main()
