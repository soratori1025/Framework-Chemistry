"""Sensitivity of the description to the redundancy threshold (reviewer minor point).

The recommender prunes redundant descriptors with a fixed absolute-correlation
threshold of 0.85.  This study repeats the selection at 0.70, 0.75, 0.80, 0.85
and 0.90, holds every other element of the protocol fixed (scorecard on the
training partition, greedy pruning, Random Forest downstream, K), and reports
both how much the selected sets change and how much the downstream score moves.

    python benchmarks/run_redundancy_sensitivity.py
    python benchmarks/run_redundancy_sensitivity.py --dataset esol freesolv --k 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks._seeds import add_seed_argument, resolve_seeds  # noqa: E402
from benchmarks.run_feature_evidence_study import evaluate_subset  # noqa: E402
from chemistry_framework.feature_analysis import select_evidence_recommendation  # noqa: E402
from chemistry_framework.features.pharmacophore import extract_rdkit_2d_descriptors  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "moleculenet"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"

THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90]


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def run_dataset(dataset: str, seeds: list[int], thresholds: list[float], k: int) -> dict[str, Any]:
    meta = json.loads((DATA_DIR / "dataset_manifest.json").read_text())[dataset]
    df = pd.read_csv(REPO_ROOT / meta["clean_csv"])
    smiles = df[meta["smiles_col"]].astype(str).tolist()
    targets = df[meta["targets"][0]].values.astype(np.float64)
    is_cls = "classification" in meta["task_type"]
    metric = "roc_auc" if is_cls else "rmse"

    valid = ~np.isnan(targets)
    smiles = [s for s, v in zip(smiles, valid) if v]
    targets = targets[valid]
    X_raw, feature_names = extract_rdkit_2d_descriptors(smiles)
    X_clean = np.nan_to_num(X_raw, nan=0.0, posinf=1e4, neginf=-1e4)

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        split = json.loads((REPO_ROOT / meta["splits"][f"seed_{seed}"]["split_file"]).read_text())
        valid_indices = [i for i, v in enumerate(valid) if v]
        orig_to_clean = {orig: clean for clean, orig in enumerate(valid_indices)}
        tr = [orig_to_clean[i] for i in split["train"] if i in orig_to_clean]
        te = [orig_to_clean[i] for i in split["test"] if i in orig_to_clean]

        scaler = StandardScaler().fit(X_clean[tr])
        X_tr = np.nan_to_num(scaler.transform(X_clean[tr]), nan=0.0, posinf=3.0, neginf=-3.0)
        X_te = np.nan_to_num(scaler.transform(X_clean[te]), nan=0.0, posinf=3.0, neginf=-3.0)
        y_tr, y_te = targets[tr], targets[te]

        selections: dict[str, list[int]] = {}
        base: set[int] | None = None
        seed_rows: list[dict[str, Any]] = []
        for thr in thresholds:
            idx = select_evidence_recommendation(
                X_train=X_tr, y_train=y_tr, feature_names=feature_names, k=k,
                redundancy_threshold=thr,
            )
            selections[str(thr)] = idx
            if abs(thr - 0.85) < 1e-9:
                base = set(idx)
            metrics = evaluate_subset(X_tr, y_tr, X_te, y_te, idx, is_cls, seed)
            seed_rows.append({
                "dataset": dataset, "split": seed, "threshold": thr,
                "metric": metric, "score": float(metrics[metric]),
                "n_features_surviving_pruning": int(len(idx)),
                "features": [feature_names[i] for i in idx],
            })
        for row in seed_rows:
            own = set(selections[str(row["threshold"])])
            row["jaccard_vs_0.85"] = jaccard(own, base) if base is not None else float("nan")
        rows.extend(seed_rows)

    summary = {}
    for thr in thresholds:
        rows_t = [r for r in rows if r["threshold"] == thr]
        scores = np.array([r["score"] for r in rows_t])
        summary[str(thr)] = {
            "n_splits": len(rows_t),
            "mean_score": float(scores.mean()),
            "sd_score": float(scores.std()),
            "mean_jaccard_vs_0.85": float(np.mean([r["jaccard_vs_0.85"] for r in rows_t])),
            "worst_abs_deviation_from_085": float(np.max(np.abs(
                scores - np.array([r["score"] for r in rows if r["threshold"] == 0.85])
            ))),
        }
    return {"dataset": dataset, "metric": metric, "k": k,
            "thresholds": thresholds, "runs": rows, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", nargs="+", default=["esol", "freesolv", "bbbp"])
    parser.add_argument("--thresholds", type=float, nargs="+", default=THRESHOLDS)
    parser.add_argument("--k", type=int, default=20)
    add_seed_argument(parser)
    args = parser.parse_args()

    payload: dict[str, Any] = {
        "note": "one-factor-at-a-time sensitivity of the recommender to the redundancy threshold; "
                "scorecard, greedy pruning, Random Forest and K held fixed",
        "thresholds": args.thresholds,
        "k": args.k,
        "datasets": {},
    }
    for dataset in args.dataset:
        block = run_dataset(dataset, resolve_seeds(args.seeds), args.thresholds, args.k)
        payload["datasets"][dataset] = block
        print(f"\n{dataset} (K={args.k}, metric={block['metric']})")
        for thr in args.thresholds:
            s = block["summary"][str(thr)]
            print(f"  threshold {thr:.2f}: {s['mean_score']:.4f} +/- {s['sd_score']:.4f} "
                  f"(Jaccard vs 0.85: {s['mean_jaccard_vs_0.85']:.2f}, "
                  f"worst |dev|: {s['worst_abs_deviation_from_085']:.4f})")

    out = RESULTS_DIR / "redundancy_sensitivity.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
