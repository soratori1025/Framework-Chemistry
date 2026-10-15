import json
import math
import numpy as np
from pathlib import Path
from scipy import stats

results_dir = Path("benchmarks/results")
ablation_file = results_dir / "modularity_ablation_bbbp.json"

if not ablation_file.exists():
    print(f"File {ablation_file} does not exist yet.")
    exit(0)

with open(ablation_file) as f:
    data = json.load(f)

raw = data["raw_results"]
configs = ["prior_only", "direct_only", "residual_only", "direct_hybrid", "prior_residual"]

print("=== BBBP Modularity Ablation (5 seeds) ===")
summary = {}
for c in configs:
    rocs = [x["roc_auc"] for x in raw[c]]
    auprcs = [x["auprc"] for x in raw[c]]
    mean_roc = np.mean(rocs)
    std_sample = np.std(rocs, ddof=1)
    std_pop = np.std(rocs)
    summary[c] = {
        "rocs": rocs,
        "mean": mean_roc,
        "std_sample": std_sample,
        "std_pop": std_pop,
        "mean_auprc": np.mean(auprcs),
        "std_auprc": np.std(auprcs, ddof=1)
    }
    print(f"{c:<18}: AUROC = {mean_roc:.4f} ± {std_sample:.4f} (pop: {std_pop:.4f}), AUPRC = {np.mean(auprcs):.4f} ± {np.std(auprcs, ddof=1):.4f}")
    print(f"  per seed: {[round(r, 4) for r in rocs]}")

def paired_comp(name_a, name_b):
    va = np.array(summary[name_a]["rocs"])
    vb = np.array(summary[name_b]["rocs"])
    diff = va - vb
    mean_d = float(np.mean(diff))
    std_d = float(np.std(diff, ddof=1))
    n = len(diff)
    t_crit = stats.t.ppf(0.975, df=n-1)
    ci_low = mean_d - t_crit * std_d / math.sqrt(n)
    ci_high = mean_d + t_crit * std_d / math.sqrt(n)
    wins = int(np.sum(diff > 0))
    print(f"\nPaired comparison: {name_a} vs {name_b}")
    print(f"  Diffs per seed: {[round(d, 4) for d in diff]}")
    print(f"  Delta: {mean_d:+.4f} ± {std_d:.4f}")
    print(f"  95% CI: [{ci_low:+.4f}, {ci_high:+.4f}]")
    print(f"  Wins: {wins}/{n}")
    return mean_d, std_d, ci_low, ci_high, wins

paired_comp("prior_residual", "residual_only")
paired_comp("prior_residual", "direct_hybrid")
paired_comp("direct_hybrid", "residual_only")
paired_comp("direct_only", "residual_only")
