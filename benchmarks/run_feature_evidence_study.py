"""Benchmark B: Systematic Feature Intelligence & Recommendation Study.

Compares feature selection strategies across K in {5, 10, 20, 50, 'all'}:
1. All Features (upper bound reference)
2. Framework Recommendation (Scorecard Spearman association + greedy redundancy filtering |r| < 0.85)
3. Spearman Top-K (conventional univariate ranking)
4. Mutual Information Top-K (conventional non-linear ranking)
5. Variance + Correlation Filter (univariate after redundancy pruning)
6. Random K-Feature Subsets (repeated random controls across 10 random seeds)

Strict train-only fit policy: all statistics, scalers, and selectors are fitted strictly
on the training partition of each split.
Also evaluates feature selection stability (pairwise Jaccard index across folds).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chemistry_framework.features.pharmacophore import extract_rdkit_2d_descriptors
from chemistry_framework.feature_analysis import select_evidence_recommendation
from benchmarks._seeds import DEFAULT_SEEDS, add_seed_argument, resolve_seeds

DATA_DIR = REPO_ROOT / "data" / "moleculenet"
SPLITS_DIR = REPO_ROOT / "benchmarks" / "splits"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"

K_LEVELS = [5, 10, 20, 50]
N_RANDOM_REPEATS = 10

def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    union = len(set_a | set_b)
    if union == 0:
        return 1.0
    return len(set_a & set_b) / union

def compute_stability(feature_sets: list[list[str]]) -> float:
    if len(feature_sets) < 2:
        return 1.0
    scores = [
        jaccard_similarity(set(a), set(b))
        for a, b in itertools.combinations(feature_sets, 2)
    ]
    return float(np.mean(scores))

def select_spearman_top_k(X_tr: np.ndarray, y_tr: np.ndarray, feature_names: list[str], k: int) -> list[int]:
    """Ranks by absolute Spearman correlation."""
    scores = []
    for j in range(X_tr.shape[1]):
        col = X_tr[:, j]
        if np.std(col) < 1e-6:
            scores.append(0.0)
            continue
        try:
            r, _ = stats.spearmanr(col, y_tr)
            scores.append(abs(r) if math.isfinite(r) else 0.0)
        except Exception:
            scores.append(0.0)
    top_indices = np.argsort(scores)[::-1][:k].tolist()
    return top_indices

def select_mutual_info_top_k(X_tr: np.ndarray, y_tr: np.ndarray, k: int, is_classification: bool, seed: int) -> list[int]:
    """Ranks by Mutual Information."""
    fn = mutual_info_classif if is_classification else mutual_info_regression
    try:
        mi = fn(X_tr, y_tr, random_state=seed)
    except Exception:
        mi = np.zeros(X_tr.shape[1])
    return np.argsort(mi)[::-1][:k].tolist()

def select_variance_correlation_top_k(X_tr: np.ndarray, y_tr: np.ndarray, k: int) -> list[int]:
    """Filters low variance (< 0.01) and redundant features (|r| > 0.85), then picks top-K by Spearman."""

    variances = np.var(X_tr, axis=0)
    keep_mask = variances >= 0.01
    valid_cols = np.where(keep_mask)[0]
    if len(valid_cols) == 0:
        return list(range(min(k, X_tr.shape[1])))

    scores = []
    for c in valid_cols:
        r, _ = stats.spearmanr(X_tr[:, c], y_tr)
        scores.append(abs(r) if math.isfinite(r) else 0.0)

    sorted_order = np.argsort(scores)[::-1]
    selected = []
    for idx_in_valid in sorted_order:
        cand_col = valid_cols[idx_in_valid]
        too_correlated = False
        for chosen in selected:
            r, _ = stats.pearsonr(X_tr[:, cand_col], X_tr[:, chosen])
            if abs(r) > 0.85:
                too_correlated = True
                break
        if not too_correlated:
            selected.append(cand_col)
        if len(selected) >= k:
            break

    if len(selected) < k:
        for idx_in_valid in sorted_order:
            cand = valid_cols[idx_in_valid]
            if cand not in selected:
                selected.append(cand)
            if len(selected) >= k:
                break
    return selected[:k]

def select_framework_recommendation(X_tr: np.ndarray, y_tr: np.ndarray, feature_names: list[str], k: int) -> list[int]:
    """Framework Recommendation: Evidence Scorecard (Spearman association + greedy redundancy pruning)."""
    return select_evidence_recommendation(
        X_train=X_tr,
        y_train=y_tr,
        feature_names=feature_names,
        k=k,
        redundancy_threshold=0.85,
    )

def evaluate_subset(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    indices: list[int],
    is_classification: bool,
    seed: int,
) -> dict[str, float]:
    sub_tr = X_tr[:, indices]
    sub_te = X_te[:, indices]

    if is_classification:
        clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
        clf.fit(sub_tr, y_tr)
        probs = clf.predict_proba(sub_te)[:, 1]
        try:
            roc = float(roc_auc_score(y_te, probs))
        except Exception:
            roc = 0.5
        try:
            pr = float(average_precision_score(y_te, probs))
        except Exception:
            pr = 0.0
        return {"roc_auc": roc, "auprc": pr}
    else:
        reg = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
        reg.fit(sub_tr, y_tr)
        preds = reg.predict(sub_te)
        rmse = float(math.sqrt(mean_squared_error(y_te, preds)))
        mae = float(mean_absolute_error(y_te, preds))
        r2 = float(r2_score(y_te, preds)) if len(y_te) > 1 else 0.0
        return {"rmse": rmse, "mae": mae, "r2": r2}

def run_feature_study_on_dataset(dataset_name: str, seeds: list[int] | None = None) -> dict[str, Any]:
    print(f"\n==================================================================")
    print(f"📊 Feature Intelligence & Recommendation Study: {dataset_name.upper()}")
    print(f"==================================================================")

    manifest_file = DATA_DIR / "dataset_manifest.json"
    with open(manifest_file, "r") as f:
        meta = json.load(f)[dataset_name]

    df_clean = pd.read_csv(REPO_ROOT / meta["clean_csv"])
    smiles = df_clean[meta["smiles_col"]].tolist()
    target_col = meta["targets"][0]
    targets = df_clean[target_col].values
    is_classification = "classification" in meta["task_type"]

    valid_target_mask = ~np.isnan(targets)
    if not np.all(valid_target_mask):
        print(f"Filtering {np.sum(~valid_target_mask)} missing targets...")
        smiles = [s for s, m in zip(smiles, valid_target_mask) if m]
        targets = targets[valid_target_mask]

    print("Extracting 200+ RDKit 2D descriptors...")
    X_raw, feature_names = extract_rdkit_2d_descriptors(smiles)
    n_features_total = X_raw.shape[1]
    print(f"Total candidate features: {n_features_total}")

    seed_list = resolve_seeds(seeds)
    method_names = [
        "framework_recommendation",
        "spearman_top_k",
        "mutual_info_top_k",
        "variance_correlation",
        "random_subsets",
    ]

    results: dict[str, dict[str, list[dict[str, float]]]] = {
        m: {str(k): [] for k in K_LEVELS} for m in method_names
    }
    results["all_features"] = {"all": []}
    selected_feature_sets: dict[str, dict[str, list[list[str]]]] = {
        m: {str(k): [] for k in K_LEVELS} for m in method_names
    }

    for seed in seed_list:
        print(f"\n--- Running Seed {seed} ---")
        split_file = REPO_ROOT / meta["splits"][f"seed_{seed}"]["split_file"]
        with open(split_file, "r") as f:
            split_dict = json.load(f)

        valid_indices = [i for i, v in enumerate(valid_target_mask) if v]
        orig_to_clean = {orig_i: clean_i for clean_i, orig_i in enumerate(valid_indices)}
        train_idx = [orig_to_clean[i] for i in split_dict["train"] if i in orig_to_clean]
        test_idx = [orig_to_clean[i] for i in split_dict["test"] if i in orig_to_clean]

        scaler = StandardScaler()
        X_clean = np.nan_to_num(X_raw, nan=0.0, posinf=1e4, neginf=-1e4)
        X_tr = scaler.fit_transform(X_clean[train_idx])
        X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=3.0, neginf=-3.0)

        X_te = scaler.transform(X_clean[test_idx])
        X_te = np.nan_to_num(X_te, nan=0.0, posinf=3.0, neginf=-3.0)

        y_tr = targets[train_idx]
        y_te = targets[test_idx]

        all_metrics = evaluate_subset(
            X_tr, y_tr, X_te, y_te,
            list(range(n_features_total)),
            is_classification, seed
        )
        results["all_features"]["all"].append(all_metrics)

        for k in K_LEVELS:

            idx_rec = select_framework_recommendation(X_tr, y_tr, feature_names, k)
            m_rec = evaluate_subset(X_tr, y_tr, X_te, y_te, idx_rec, is_classification, seed)
            results["framework_recommendation"][str(k)].append(m_rec)
            selected_feature_sets["framework_recommendation"][str(k)].append([feature_names[i] for i in idx_rec])

            idx_sp = select_spearman_top_k(X_tr, y_tr, feature_names, k)
            m_sp = evaluate_subset(X_tr, y_tr, X_te, y_te, idx_sp, is_classification, seed)
            results["spearman_top_k"][str(k)].append(m_sp)
            selected_feature_sets["spearman_top_k"][str(k)].append([feature_names[i] for i in idx_sp])

            idx_mi = select_mutual_info_top_k(X_tr, y_tr, k, is_classification, seed)
            m_mi = evaluate_subset(X_tr, y_tr, X_te, y_te, idx_mi, is_classification, seed)
            results["mutual_info_top_k"][str(k)].append(m_mi)
            selected_feature_sets["mutual_info_top_k"][str(k)].append([feature_names[i] for i in idx_mi])

            idx_vc = select_variance_correlation_top_k(X_tr, y_tr, k)
            m_vc = evaluate_subset(X_tr, y_tr, X_te, y_te, idx_vc, is_classification, seed)
            results["variance_correlation"][str(k)].append(m_vc)
            selected_feature_sets["variance_correlation"][str(k)].append([feature_names[i] for i in idx_vc])

            rand_runs = []
            rng = np.random.default_rng(seed * 1000 + k)
            for r in range(N_RANDOM_REPEATS):
                rand_idx = rng.choice(n_features_total, size=k, replace=False).tolist()
                rm = evaluate_subset(X_tr, y_tr, X_te, y_te, rand_idx, is_classification, seed + r)
                rand_runs.append(rm)

            avg_rand = {
                metric: float(np.mean([run[metric] for run in rand_runs]))
                for metric in rand_runs[0]
            }
            results["random_subsets"][str(k)].append(avg_rand)

    primary_metric = "roc_auc" if is_classification else "rmse"
    print(f"\n📈 Final Summary Across {len(seed_list)} Splits ({primary_metric.upper()}):")

    summary_table = []

    all_vals = [m[primary_metric] for m in results["all_features"]["all"]]
    print(f"  All Features (232): {np.mean(all_vals):.4f} ± {np.std(all_vals):.4f}")

    for k in K_LEVELS:
        row = {"k": k}
        print(f"\n  [K = {k}]")
        for m in method_names:
            vals = [run[primary_metric] for run in results[m][str(k)]]
            stability = compute_stability(selected_feature_sets[m][str(k)]) if m != "random_subsets" else 0.0
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals))
            row[f"{m}_mean"] = mean_val
            row[f"{m}_std"] = std_val
            row[f"{m}_stability"] = stability
            print(f"    - {m:<25}: {mean_val:.4f} ± {std_val:.4f} (Stability: {stability:.2f})")
        summary_table.append(row)

    output_payload = {
        "dataset": dataset_name,
        "is_classification": is_classification,
        "primary_metric": primary_metric,
        "seeds": seed_list,
        "k_levels": K_LEVELS,
        "raw_results": results,
        "summary_table": summary_table,
    }

    out_path = RESULTS_DIR / f"feature_study_{dataset_name}.json"
    with open(out_path, "w") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\n✅ Feature Intelligence Study for {dataset_name} saved to: {out_path}")
    return output_payload

def main():
    parser = argparse.ArgumentParser(description="Run Feature Intelligence Study")
    parser.add_argument("--dataset", type=str, default="esol", help="Dataset name: bbbp, esol, freesolv, etc.")
    add_seed_argument(parser)
    args = parser.parse_args()
    run_feature_study_on_dataset(args.dataset, seeds=args.seeds)

if __name__ == "__main__":
    main()
