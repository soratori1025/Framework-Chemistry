from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LogisticRegression, lars_path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chemistry_framework.features.pharmacophore import extract_rdkit_2d_descriptors
from chemistry_framework.feature_analysis import select_evidence_recommendation
from benchmarks._seeds import DEFAULT_SEEDS, add_seed_argument, resolve_seeds

DATA_DIR = REPO_ROOT / "data" / "moleculenet"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

K_LEVELS = [5, 10, 20, 50]

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

def select_spearman_top_k(X_tr: np.ndarray, y_tr: np.ndarray, k: int) -> list[int]:
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
    return np.argsort(scores)[::-1][:k].tolist()

def select_mutual_info_top_k(X_tr: np.ndarray, y_tr: np.ndarray, k: int, is_classification: bool, seed: int) -> list[int]:
    fn = mutual_info_classif if is_classification else mutual_info_regression
    try:
        mi = fn(X_tr, y_tr, random_state=seed)
    except Exception:
        mi = np.zeros(X_tr.shape[1])
    return np.argsort(mi)[::-1][:k].tolist()

def select_mrmr_top_k(X_tr: np.ndarray, y_tr: np.ndarray, k: int, is_classification: bool, seed: int) -> list[int]:
    """Minimum Redundancy Maximum Relevance (Peng et al. 2005)."""
    n_features = X_tr.shape[1]
    fn = mutual_info_classif if is_classification else mutual_info_regression
    try:
        mi = fn(X_tr, y_tr, random_state=seed)
    except Exception:
        mi = np.zeros(n_features)

    with np.errstate(divide="ignore", invalid="ignore"):
        corr_matrix = np.nan_to_num(np.abs(np.corrcoef(X_tr, rowvar=False)), nan=0.0)

    selected: list[int] = [int(np.argmax(mi))]
    candidates = set(range(n_features)) - {selected[0]}

    while len(selected) < min(k, n_features):
        best_cand = -1
        best_score = -float("inf")
        for cand in candidates:

            rel = mi[cand]

            red = np.mean([corr_matrix[cand, s] for s in selected])
            score = rel - red
            if score > best_score:
                best_score = score
                best_cand = cand
        if best_cand == -1:
            break
        selected.append(best_cand)
        candidates.remove(best_cand)

    return selected[:k]

def select_rf_importance_top_k(X_tr: np.ndarray, y_tr: np.ndarray, k: int, is_classification: bool, seed: int) -> list[int]:
    """Random Forest Gini / impurity-based feature importance on training fold."""
    if is_classification:
        rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=seed, n_jobs=-1)
    else:
        rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=seed, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    importances = rf.feature_importances_
    return np.argsort(importances)[::-1][:k].tolist()

def select_l1_top_k(X_tr: np.ndarray, y_tr: np.ndarray, k: int, is_classification: bool, seed: int) -> list[int]:
    """L1 Regularization selector:
    - Regression: LARS/Lasso regularization path (order of entry into active set).
    - Classification: L1-penalized Logistic Regression coefficient magnitude.
    """
    n_features = X_tr.shape[1]
    if not is_classification:
        try:
            _, active, _ = lars_path(X_tr, y_tr, method="lasso")
            selected = list(active)
            if len(selected) < k:
                remaining = [i for i in range(n_features) if i not in selected]
                selected.extend(remaining[:k - len(selected)])
            return selected[:k]
        except Exception:
            pass

    try:
        if is_classification:
            clf = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, random_state=seed, max_iter=2000)
            clf.fit(X_tr, y_tr)
            coefs = np.abs(clf.coef_[0])
        else:
            from sklearn.linear_model import Lasso
            lasso = Lasso(alpha=0.01, random_state=seed, max_iter=2000)
            lasso.fit(X_tr, y_tr)
            coefs = np.abs(lasso.coef_)
        return np.argsort(coefs)[::-1][:k].tolist()
    except Exception:
        return list(range(min(k, n_features)))

def select_framework_top_k(X_tr: np.ndarray, y_tr: np.ndarray, feature_names: list[str], k: int) -> list[int]:
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

def run_multivariate_study_on_dataset(dataset_name: str, seeds: list[int] | None = None) -> dict[str, Any]:
    print(f"\n==================================================================")
    print(f"📊 Multivariate Feature Study: {dataset_name.upper()}")
    print(f"==================================================================")

    manifest_file = DATA_DIR / "dataset_manifest.json"
    with open(manifest_file, "r") as f:
        meta = json.load(f)[dataset_name]

    df_clean = pd.read_csv(REPO_ROOT / meta["clean_csv"])
    smiles = df_clean[meta["smiles_col"]].tolist()
    target_col = meta["targets"][0]
    targets = df_clean[target_col].values
    is_classification = "classification" in meta["task_type"]
    primary_metric = "roc_auc" if is_classification else "rmse"

    valid_target_mask = ~np.isnan(targets)
    if not np.all(valid_target_mask):
        print(f"Filtering {np.sum(~valid_target_mask)} missing targets...")
        smiles = [s for s, m in zip(smiles, valid_target_mask) if m]
        targets = targets[valid_target_mask]

    print("Extracting 200+ RDKit 2D descriptors...")
    X_raw, feature_names = extract_rdkit_2d_descriptors(smiles)

    seed_list = resolve_seeds(seeds)
    selectors = [
        "framework",
        "spearman",
        "mutual_info",
        "mrmr",
        "rf_importance",
        "l1",
    ]

    results: dict[str, dict[str, list[dict[str, float]]]] = {
        m: {str(k): [] for k in K_LEVELS} for m in selectors
    }
    selected_feature_sets: dict[str, dict[str, list[list[str]]]] = {
        m: {str(k): [] for k in K_LEVELS} for m in selectors
    }

    for seed in seed_list:
        print(f"\n--- Running Split / Seed {seed} ---")
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

        for k in K_LEVELS:
            print(f"  Evaluating K={k}...")

            idx_fw = select_framework_top_k(X_tr, y_tr, feature_names, k)
            m_fw = evaluate_subset(X_tr, y_tr, X_te, y_te, idx_fw, is_classification, seed)
            results["framework"][str(k)].append(m_fw)
            selected_feature_sets["framework"][str(k)].append([feature_names[i] for i in idx_fw])

            idx_sp = select_spearman_top_k(X_tr, y_tr, k)
            m_sp = evaluate_subset(X_tr, y_tr, X_te, y_te, idx_sp, is_classification, seed)
            results["spearman"][str(k)].append(m_sp)
            selected_feature_sets["spearman"][str(k)].append([feature_names[i] for i in idx_sp])

            idx_mi = select_mutual_info_top_k(X_tr, y_tr, k, is_classification, seed)
            m_mi = evaluate_subset(X_tr, y_tr, X_te, y_te, idx_mi, is_classification, seed)
            results["mutual_info"][str(k)].append(m_mi)
            selected_feature_sets["mutual_info"][str(k)].append([feature_names[i] for i in idx_mi])

            idx_mrmr = select_mrmr_top_k(X_tr, y_tr, k, is_classification, seed)
            m_mrmr = evaluate_subset(X_tr, y_tr, X_te, y_te, idx_mrmr, is_classification, seed)
            results["mrmr"][str(k)].append(m_mrmr)
            selected_feature_sets["mrmr"][str(k)].append([feature_names[i] for i in idx_mrmr])

            idx_rf = select_rf_importance_top_k(X_tr, y_tr, k, is_classification, seed)
            m_rf = evaluate_subset(X_tr, y_tr, X_te, y_te, idx_rf, is_classification, seed)
            results["rf_importance"][str(k)].append(m_rf)
            selected_feature_sets["rf_importance"][str(k)].append([feature_names[i] for i in idx_rf])

            idx_l1 = select_l1_top_k(X_tr, y_tr, k, is_classification, seed)
            m_l1 = evaluate_subset(X_tr, y_tr, X_te, y_te, idx_l1, is_classification, seed)
            results["l1"][str(k)].append(m_l1)
            selected_feature_sets["l1"][str(k)].append([feature_names[i] for i in idx_l1])

            print(f"    FW: {m_fw[primary_metric]:.4f} | Sp: {m_sp[primary_metric]:.4f} | MI: {m_mi[primary_metric]:.4f} | mRMR: {m_mrmr[primary_metric]:.4f} | RF: {m_rf[primary_metric]:.4f} | L1: {m_l1[primary_metric]:.4f}")

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for m in selectors:
        summary[m] = {}
        for k in K_LEVELS:
            vals = [r[primary_metric] for r in results[m][str(k)]]
            summary[m][str(k)] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "stability_jaccard": compute_stability(selected_feature_sets[m][str(k)]),
            }

    print(f"\n==================================================================")
    print(f"🏆 SUMMARY FOR {dataset_name.upper()} ({primary_metric.upper()})")
    print(f"==================================================================")
    header = f"{'Method':<20}" + "".join([f"K={k:<12}" for k in K_LEVELS])
    print(header)
    print("-" * len(header))
    for m in selectors:
        row = f"{m:<20}"
        for k in K_LEVELS:
            mean = summary[m][str(k)]["mean"]
            std = summary[m][str(k)]["std"]
            row += f"{mean:.4f}±{std:.3f}  "
        print(row)

    output_path = RESULTS_DIR / f"multivariate_feature_study_{dataset_name}.json"
    with open(output_path, "w") as f:
        json.dump({
            "dataset": dataset_name,
            "primary_metric": primary_metric,
            "seeds": seed_list,
            "summary": summary,
            "per_seed_results": results,
        }, f, indent=2)
    print(f"Saved results to {output_path}")
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all", choices=["esol", "freesolv", "bbbp", "lipophilicity", "bace", "all"])
    add_seed_argument(parser)
    args = parser.parse_args()

    datasets = ["esol", "freesolv", "bbbp", "lipophilicity", "bace"] if args.dataset == "all" else [args.dataset]
    for d in datasets:
        run_multivariate_study_on_dataset(d, seeds=args.seeds)

if __name__ == "__main__":
    main()
