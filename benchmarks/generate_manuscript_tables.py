from __future__ import annotations
import json
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"

def load_json(filename: str):
    p = RESULTS_DIR / filename
    if not p.exists():
        return None
    with open(p, "r") as f:
        return json.load(f)

def build_master_manifest():
    manifest = {
        "benchmark_a_baselines": {},
        "benchmark_b_feature_study": {},
        "benchmark_c_modularity_ablation": {},
    }

    for d in ["esol", "freesolv", "bbbp", "lipophilicity", "bace"]:
        data = load_json(f"baselines_{d}_aggregated.json")
        if data:
            manifest["benchmark_a_baselines"][d] = data["aggregated"]

    for d in ["esol", "freesolv", "bbbp"]:
        data = load_json(f"feature_study_{d}.json")
        if data:
            all_list = data["raw_results"]["all_features"]["all"]
            metric = data["primary_metric"]
            vals = [x[metric] for x in all_list]
            manifest["benchmark_b_feature_study"][d] = {
                "metric": metric,
                "all_features": {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "values": vals,
                },
                "summary_table": data["summary_table"],
            }

    for d in ["esol", "freesolv", "bbbp", "lipophilicity", "bace"]:
        data = load_json(f"modularity_ablation_{d}.json")
        if data:
            manifest["benchmark_c_modularity_ablation"][d] = data["summary"]

    manifest["chemprop_v2"] = {}
    for d in ["esol", "freesolv", "bbbp", "lipophilicity", "bace"]:
        data = load_json(f"chemprop_v2_{d}.json")
        if data:
            manifest["chemprop_v2"][d] = data["summary"]

    out_file = RESULTS_DIR / "manifest.json"
    with open(out_file, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"✓ Master manifest updated: {out_file}")
    return manifest

def print_summary_tables(manifest):
    print("\n" + "=" * 80)
    print("📊 TABLE 3: CANONICAL BASELINES ACROSS DATASETS (5 SEEDS, MEAN ± STD)")
    print("=" * 80)
    print(f"{'Dataset':<12} | {'Metric':<10} | {'Dummy':<15} | {'ECFP4+RF':<15} | {'RDKit 2D+RF':<15} | {'Descriptor MLP':<15}")
    print("-" * 88)
    for d, models in manifest["benchmark_a_baselines"].items():
        if "rmse" in models.get("dummy", {}):
            metric = "RMSE (↓)"
            dummy_str = f"{models['dummy']['rmse']['mean']:.3f} ± {models['dummy']['rmse']['std']:.3f}"
            ecfp_str = f"{models['ecfp4_rf']['rmse']['mean']:.3f} ± {models['ecfp4_rf']['rmse']['std']:.3f}"
            rdkit_str = f"{models['rdkit_rf']['rmse']['mean']:.3f} ± {models['rdkit_rf']['rmse']['std']:.3f}"
            mlp_str = f"{models['descriptor_mlp']['rmse']['mean']:.3f} ± {models['descriptor_mlp']['rmse']['std']:.3f}"
        else:
            metric = "ROC-AUC (↑)"
            dummy_str = f"{models['dummy']['roc_auc']['mean']:.3f} ± {models['dummy']['roc_auc']['std']:.3f}"
            ecfp_str = f"{models['ecfp4_rf']['roc_auc']['mean']:.3f} ± {models['ecfp4_rf']['roc_auc']['std']:.3f}"
            rdkit_str = f"{models['rdkit_rf']['roc_auc']['mean']:.3f} ± {models['rdkit_rf']['roc_auc']['std']:.3f}"
            mlp_str = f"{models['descriptor_mlp']['roc_auc']['mean']:.3f} ± {models['descriptor_mlp']['roc_auc']['std']:.3f}"
        print(f"{d.upper():<12} | {metric:<10} | {dummy_str:<15} | {ecfp_str:<15} | {rdkit_str:<15} | {mlp_str:<15}")

    print("\n" + "=" * 96)
    print("🔬 TABLE 4: PRIOR/RESIDUAL MODULARITY ABLATION (5 SEEDS, MEAN ± STD)")
    print("=" * 96)
    print(f"{'Configuration':<18} | {'Role Description':<30} | {'ESOL (RMSE ↓)':<14} | {'FreeSolv (RMSE ↓)':<16} | {'BBBP (AUROC ↑)':<14}")
    print("-" * 102)
    desc_map = {
        "prior_only": "Linear/Logistic on 6 PhysChem",
        "direct_only": "3-layer MLP on all Descriptors",
        "residual_only": "D-MPNN Graph-only",
        "direct_hybrid": "D-MPNN + Descriptors Concat",
        "prior_residual": "Explicit Prior + Residual",
    }
    configs = ["prior_only", "direct_only", "residual_only", "direct_hybrid", "prior_residual"]
    esol_data = manifest["benchmark_c_modularity_ablation"].get("esol", {})
    free_data = manifest["benchmark_c_modularity_ablation"].get("freesolv", {})
    bbbp_data = manifest["benchmark_c_modularity_ablation"].get("bbbp", {})

    for c in configs:
        e_val = esol_data.get(c, {})
        f_val = free_data.get(c, {})
        b_val = bbbp_data.get(c, {})
        e_str = f"{e_val.get('mean', 0.0):.4f} ± {e_val.get('std', 0.0):.4f}" if e_val else "N/A"
        f_str = f"{f_val.get('mean', 0.0):.4f} ± {f_val.get('std', 0.0):.4f}" if f_val else "N/A"
        b_str = f"{b_val.get('mean', 0.0):.4f} ± {b_val.get('std', 0.0):.4f}" if b_val else "N/A"
        print(f"{c:<18} | {desc_map[c]:<30} | {e_str:<14} | {f_str:<16} | {b_str:<14}")

    print("\n" + "=" * 80)
    print("📈 FEATURE INTELLIGENCE STUDY HIGHLIGHTS (SECTION 3.2)")
    print("=" * 80)
    for d, data in manifest["benchmark_b_feature_study"].items():
        all_feat = data["all_features"]
        metric_name = data["metric"].upper()
        print(f"\nDataset: {d.upper()} [{metric_name}] (All Candidate Descriptors: {all_feat['mean']:.4f} ± {all_feat['std']:.4f})")
        for row in data["summary_table"]:
            k = row["k"]
            fw_m = row.get("framework_recommendation_mean", 0.0)
            fw_s = row.get("framework_recommendation_std", 0.0)
            fw_stab = row.get("framework_recommendation_stability", 0.0)
            rnd_m = row.get("random_subsets_mean", 0.0)
            rnd_s = row.get("random_subsets_std", 0.0)
            print(f"  K={k:<2}: Framework = {fw_m:.4f} ± {fw_s:.4f} (Stability: {fw_stab:.2f}) vs Random = {rnd_m:.4f} ± {rnd_s:.4f}")

if __name__ == "__main__":
    m = build_master_manifest()
    print_summary_tables(m)
