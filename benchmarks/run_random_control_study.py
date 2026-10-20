"""Benchmark C: high-repetition random-subset control for the recommender.

Reviewer-driven extension of the ten-draw control in
``run_feature_evidence_study.py``.  That study compared the framework
recommendation with the *mean of ten random subsets per split*, which leaves the
control with the draw noise of the comparator and restricts the comparison to
ESOL, FreeSolv and BBBP.  This study

1. extends the control to all five benchmark datasets (including
   Lipophilicity and BACE, which previously had no random control), and
2. keeps every random draw instead of only its mean, so that the recommender
   can be compared with the *distribution* of random subsets: the fraction of
   draws the recommendation beats (0.5 is chance), a per-split rank p-value with
   resolution 1/(N+1), and a Fisher-combined p-value across the ten splits.

The protocol is identical to ``run_multivariate_feature_study.py`` -- same
descriptor extraction, same train-only scaling and NaN handling, the same
Random Forest (100 trees, no depth limit), the same split manifests and the
same RNG convention (``default_rng(seed * 1000 + k)``), so the first ten draws
reproduce the published control exactly.  That reproduction is verified against
``feature_study_<dataset>.json`` at the end of each dataset.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chemistry_framework.feature_analysis import select_evidence_recommendation
from chemistry_framework.features.pharmacophore import extract_rdkit_2d_descriptors
from benchmarks._seeds import add_seed_argument, resolve_seeds

DATA_DIR = REPO_ROOT / "data" / "moleculenet"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"

K_LEVELS = [5, 10, 20, 50]
PUBLISHED_DRAWS = 10  # draws behind the published ten-draw control


def _fit_and_score(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    indices: list[int],
    is_classification: bool,
    rf_seed: int,
) -> float:
    """Train the downstream Random Forest on a subset and return the metric."""
    sub_tr = X_tr[:, indices]
    sub_te = X_te[:, indices]
    if is_classification:
        clf = RandomForestClassifier(n_estimators=100, random_state=rf_seed, n_jobs=1)
        clf.fit(sub_tr, y_tr)
        probs = clf.predict_proba(sub_te)[:, 1]
        return float(roc_auc_score(y_te, probs))
    reg = RandomForestRegressor(n_estimators=100, random_state=rf_seed, n_jobs=1)
    reg.fit(sub_tr, y_tr)
    preds = reg.predict(sub_te)
    return float(math.sqrt(np.mean((y_te - preds) ** 2)))


def _better(a: float, b: float, is_classification: bool) -> bool:
    return a > b if is_classification else a < b


def _prepare_dataset(dataset_name: str) -> dict[str, Any]:
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
        smiles = [s for s, m in zip(smiles, valid_target_mask) if m]
        targets = targets[valid_target_mask]

    X_raw, feature_names = extract_rdkit_2d_descriptors(smiles)
    return {
        "meta": meta,
        "targets": targets,
        "valid_target_mask": valid_target_mask,
        "X_raw": X_raw,
        "feature_names": feature_names,
        "is_classification": is_classification,
        "primary_metric": "roc_auc" if is_classification else "rmse",
    }


def run_random_control_dataset(
    dataset_name: str,
    draws: int = 50,
    seeds: list[int] | None = None,
    jobs: int = 12,
) -> dict[str, Any]:
    print(f"\n=== Random-subset control: {dataset_name.upper()} ({draws} draws per cell) ===", flush=True)
    prep = _prepare_dataset(dataset_name)
    meta = prep["meta"]
    targets = prep["targets"]
    valid_target_mask = prep["valid_target_mask"]
    X_raw = prep["X_raw"]
    feature_names = prep["feature_names"]
    is_classification = prep["is_classification"]
    primary_metric = prep["primary_metric"]

    n_features_total = X_raw.shape[1]
    seed_list = resolve_seeds(seeds)

    per_split: list[dict[str, Any]] = []
    for seed in seed_list:
        split_file = REPO_ROOT / meta["splits"][f"seed_{seed}"]["split_file"]
        with open(split_file, "r") as f:
            split_dict = json.load(f)

        valid_indices = [i for i, v in enumerate(valid_target_mask) if v]
        orig_to_clean = {orig_i: clean_i for clean_i, orig_i in enumerate(valid_indices)}
        train_idx = [orig_to_clean[i] for i in split_dict["train"] if i in orig_to_clean]
        test_idx = [orig_to_clean[i] for i in split_dict["test"] if i in orig_to_clean]

        scaler = StandardScaler()
        X_clean = np.nan_to_num(X_raw, nan=0.0, posinf=1e4, neginf=-1e4)
        X_tr = np.nan_to_num(scaler.fit_transform(X_clean[train_idx]), nan=0.0, posinf=3.0, neginf=-3.0)
        X_te = np.nan_to_num(scaler.transform(X_clean[test_idx]), nan=0.0, posinf=3.0, neginf=-3.0)
        y_tr = targets[train_idx]
        y_te = targets[test_idx]

        full_pool = _fit_and_score(
            X_tr, y_tr, X_te, y_te,
            list(range(n_features_total)), is_classification, seed,
        )

        cell_entries: dict[str, Any] = {}
        for k in K_LEVELS:
            rec_idx = select_evidence_recommendation(
                X_train=X_tr, y_train=y_tr, feature_names=feature_names,
                k=k, redundancy_threshold=0.85,
            )
            rec_score = _fit_and_score(
                X_tr, y_tr, X_te, y_te, list(rec_idx), is_classification, seed,
            )

            rng = np.random.default_rng(seed * 1000 + k)
            draw_index_sets = [
                rng.choice(n_features_total, size=k, replace=False).tolist()
                for _ in range(draws)
            ]
            draw_scores = Parallel(n_jobs=jobs, batch_size=4)(
                delayed(_fit_and_score)(
                    X_tr, y_tr, X_te, y_te, idx, is_classification, seed + r,
                )
                for r, idx in enumerate(draw_index_sets)
            )
            draw_scores_arr = np.asarray(draw_scores, dtype=float)

            if is_classification:
                n_better = int(np.sum(draw_scores_arr > rec_score))
                n_tied = int(np.sum(np.isclose(draw_scores_arr, rec_score, rtol=1e-9, atol=1e-12)))
            else:
                n_better = int(np.sum(draw_scores_arr < rec_score))
                n_tied = int(np.sum(np.isclose(draw_scores_arr, rec_score, rtol=1e-9, atol=1e-12)))
            # Ties count as half: without this a control whose draws all tie with
            # the recommendation would be reported as a near-certain win.
            n_effective = n_better + 0.5 * n_tied
            p_rank = (n_effective + 1.0) / (draws + 1.0)
            # Fraction of the control that the recommendation beat (0.5 = chance).
            percentile = 1.0 - n_effective / draws

            # Distribution of the published statistic (mean of 10 draws), obtained
            # by resampling ten draws from the same 50-draw pool.
            resample_rng = np.random.default_rng(seed * 7919 + k)
            resampled_means = np.array([
                float(np.mean(draw_scores_arr[resample_rng.integers(0, draws, PUBLISHED_DRAWS)]))
                for _ in range(2000)
            ])
            if is_classification:
                p_10draw = float((np.sum(resampled_means >= rec_score) + 1) / (len(resampled_means) + 1))
            else:
                p_10draw = float((np.sum(resampled_means <= rec_score) + 1) / (len(resampled_means) + 1))

            cell_entries[str(k)] = {
                "recommended": rec_score,
                "full_pool": full_pool,
                "random_draw_scores": [float(v) for v in draw_scores_arr],
                "random_mean": float(np.mean(draw_scores_arr)),
                "random_mean_of_first_10": float(np.mean(draw_scores_arr[:PUBLISHED_DRAWS])),
                "random_sd_within_split": float(np.std(draw_scores_arr)),
                "random_min": float(np.min(draw_scores_arr)),
                "random_max": float(np.max(draw_scores_arr)),
                "n_draws_better": n_better,
                "n_draws_tied": n_tied,
                "rank_percentile": float(percentile),
                "p_rank": float(p_rank),
                "p_ten_draw_mean": p_10draw,
                "ten_draw_mean_reference": float(np.mean(resampled_means)),
            }
            print(
                f"  split {seed} K={k}: rec={rec_score:.4f} "
                f"rand={np.mean(draw_scores_arr):.4f} (min {np.min(draw_scores_arr):.4f}) "
                f"pct={percentile:.3f} ties={n_tied}",
                flush=True,
            )

        per_split.append({
            "split": seed,
            "seed": seed,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "full_pool": full_pool,
            "cells": cell_entries,
        })

    # ---- aggregate -----------------------------------------------------------
    summary: dict[str, Any] = {}
    for k in K_LEVELS:
        rec = np.array([s["cells"][str(k)]["recommended"] for s in per_split])
        rnd = np.array([s["cells"][str(k)]["random_mean"] for s in per_split])
        rnd10 = np.array([s["cells"][str(k)]["random_mean_of_first_10"] for s in per_split])
        pool = np.array([s["cells"][str(k)]["full_pool"] for s in per_split])
        p_rank = np.array([s["cells"][str(k)]["p_rank"] for s in per_split])
        pct = np.array([s["cells"][str(k)]["rank_percentile"] for s in per_split])

        fisher_stat = float(-2.0 * np.sum(np.log(np.clip(p_rank, 1e-12, 1.0))))
        fisher_p = float(stats.chi2.sf(fisher_stat, df=2 * len(p_rank)))

        # Exact sign-flip permutation test over splits for the published
        # statistic (recommender minus the ten-draw mean).
        paired = rec - rnd10
        observed = float(np.mean(paired))
        signs = np.array([[1.0 if (m >> b) & 1 else -1.0 for b in range(len(paired))] for m in range(2 ** len(paired))])
        perm_means = signs @ paired / len(paired)
        # two-sided permutation p-value
        p_perm = float(np.mean(np.abs(perm_means) >= abs(observed) - 1e-12))

        summary[str(k)] = {
            "recommended_mean": float(np.mean(rec)),
            "recommended_sd": float(np.std(rec)),
            "random_mean": float(np.mean(rnd)),
            "random_mean_sd_across_splits": float(np.std(rnd)),
            "random_ten_draw_mean": float(np.mean(rnd10)),
            "random_ten_draw_sd_across_splits": float(np.std(rnd10)),
            "full_pool_mean": float(np.mean(pool)),
            "paired_delta_vs_ten_draw_control": float(observed),
            "paired_delta_sd": float(np.std(paired)),
            # Direction-aware: for RMSE a lower score is better, for AUROC a
            # higher one.  ``paired`` is recommended minus control, so the
            # comparison flips with the metric.
            "recommended_better_splits": int(
                np.sum(paired > 0) if is_classification else np.sum(paired < 0)
            ),
            "p_signflip_two_sided": p_perm,
            "mean_rank_percentile": float(np.mean(pct)),
            "min_rank_percentile": float(np.min(pct)),
            "p_rank_fisher": fisher_p,
            "p_rank_median": float(np.median(p_rank)),
        }
        print(f"  K={k}: delta(rec-ten-draw)={observed:+.4f} p_signflip={p_perm:.4f} "
              f"mean pct={np.mean(pct):.3f} fisher p={fisher_p:.3g}", flush=True)

    payload = {
        "dataset": dataset_name,
        "is_classification": is_classification,
        "primary_metric": primary_metric,
        "draws_per_cell": draws,
        "seeds": seed_list,
        "k_levels": K_LEVELS,
        "protocol": "run_multivariate_feature_study.py (100-tree Random Forest, train-only scaling)",
        "delta_sign": "recommended minus control; negative is better for RMSE, positive for AUROC",
        "summary": summary,
        "per_split": per_split,
    }

    # ---- verification against the published ten-draw control -----------------
    feature_study = RESULTS_DIR / f"feature_study_{dataset_name}.json"
    if feature_study.exists():
        with open(feature_study, "r") as f:
            published = json.load(f)
        checks = []
        for k in K_LEVELS:
            published_mean = float(np.mean([
                run[primary_metric] for run in published["raw_results"]["random_subsets"][str(k)]
            ]))
            reproduced = float(np.mean([
                s["cells"][str(k)]["random_mean_of_first_10"] for s in per_split
            ]))
            checks.append({
                "k": k,
                "published_ten_draw_mean": published_mean,
                "reproduced_ten_draw_mean": reproduced,
                "abs_diff": abs(published_mean - reproduced),
            })
        payload["reproduction_check"] = checks
        worst = max(c["abs_diff"] for c in checks)
        print(f"  reproduction of published ten-draw control: max |diff| = {worst:.2e}", flush=True)

    output_path = RESULTS_DIR / f"random_control_{dataset_name}.json"
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  saved {output_path}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="High-repetition random-subset control")
    parser.add_argument(
        "--dataset", type=str, default="all",
        choices=["esol", "freesolv", "bbbp", "lipophilicity", "bace", "all"],
    )
    parser.add_argument("--draws", type=int, default=50, help="random draws per dataset-budget cell")
    parser.add_argument("--jobs", type=int, default=12, help="parallel workers for the Random Forest fits")
    add_seed_argument(parser)
    args = parser.parse_args()

    datasets = (
        ["esol", "freesolv", "lipophilicity", "bbbp", "bace"]
        if args.dataset == "all" else [args.dataset]
    )
    for name in datasets:
        run_random_control_dataset(name, draws=args.draws, seeds=args.seeds, jobs=args.jobs)


if __name__ == "__main__":
    main()
