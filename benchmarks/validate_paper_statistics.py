"""Validate the recomputed naive statistics against the published SI values.

The published values below are transcribed from Additional file 1
(Tables: internal paired comparisons and external comparisons).  If a
recomputed value differs, the pairing is wrong and must be fixed before the
corrected intervals are trusted.
"""
import json
import sys
from pathlib import Path

R = Path(r"D:\ngan-work\chemistry-framework\Framework-Chemistry\benchmarks\results")
data = json.loads((R / "paper_statistics.json").read_text())
comps = data["comparisons"]

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

index = {(c["dataset"], c["label_a"], c["label_b"]): c for c in comps}

problems = 0
for ds, la, lb, delta, sd, lo, hi, wins in PUBLISHED:
    c = index.get((ds, la, lb))
    if c is None:
        print(f"MISSING {ds} {la} vs {lb}")
        problems += 1
        continue
    issues = []
    if abs(c["delta"] - delta) > 5e-5:
        issues.append(f"delta {c['delta']:+.4f} != {delta:+.4f}")
    if abs(c["sd_pop"] - sd) > 5e-5:
        issues.append(f"sd_pop {c['sd_pop']:.4f} != {sd:.4f}")
    if abs(c["naive_ci"][0] - lo) > 1.5e-3:
        issues.append(f"naive_lo {c['naive_ci'][0]:+.3f} != {lo:+.3f}")
    if abs(c["naive_ci"][1] - hi) > 1.5e-3:
        issues.append(f"naive_hi {c['naive_ci'][1]:+.3f} != {hi:+.3f}")
    if c["wins"] != wins:
        issues.append(f"wins {c['wins']} != {wins}")
    status = "OK " if not issues else "FAIL"
    if issues:
        problems += 1
    print(f"{status} {ds:<15} {la:<45} {'; '.join(issues)}")

print(f"\n{len(PUBLISHED) - problems}/{len(PUBLISHED)} published values reproduced")
sys.exit(1 if problems else 0)
