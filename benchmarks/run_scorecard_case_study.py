"""Experiment E2: Scorecard Case Study & Dataset Diagnostics (Journal of Cheminformatics).

1. BBBP Hard Split Analysis:
   - For all 5 splits of BBBP, compute the number of descriptors with test values outside
     the training range (out_of_range_count > 0) and the percentage of test molecules affected.
   - Correlate with model performance degradation on Split 5 (where direct_hybrid dropped to 0.873).

2. Scaffold vs. Random Split Distribution Shift:
   - Compare scaffold partitioning against random partitioning on ESOL and Lipophilicity.
   - Quantify distribution shift via:
     * Number and fraction of out-of-range descriptors
     * Test molecules affected by out-of-range features
     * Evidence class distribution shifts (ood_unsupported, redundant, weak_uncertain, supported)
     * Mean normalized feature mean shift: |mu_test - mu_train| / sigma_train.

3. Dataset Audit & Structural Curation Analysis (connecting with Schuh et al. 2023):
   - For all 5 datasets (ESOL, FreeSolv, BBBP, Lipophilicity, BACE):
     * Count duplicate records by Canonical SMILES and InChIKey
     * Identify conflicting labels (contradictory annotations for identical chemical species)
     * Report duplicate frequency and noise floor implications.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chemistry_framework.features.pharmacophore import extract_rdkit_2d_descriptors

DATA_DIR = REPO_ROOT / "data" / "moleculenet"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# The benchmark design uses ten split manifests (seeds 1-10). The dataset
# directories also hold extra manifests used only by the duplicate-controlled
# BBBP sensitivity analysis, so every loop has to filter on these seeds.
STUDY_SEEDS = tuple(range(1, 11))

def analyze_bbbp_splits() -> dict[str, Any]:
    print("\n" + "=" * 70)
    print("🔬 E2.1: BBBP Split Breakdown & OOD Descriptor Analysis")
    print("=" * 70)

    manifest_file = DATA_DIR / "dataset_manifest.json"
    with open(manifest_file, "r") as f:
        meta = json.load(f)["bbbp"]

    df_clean = pd.read_csv(REPO_ROOT / meta["clean_csv"])
    smiles = df_clean[meta["smiles_col"]].tolist()
    targets = df_clean[meta["targets"][0]].values

    X_raw, feature_names = extract_rdkit_2d_descriptors(smiles)
    X_clean = np.nan_to_num(X_raw, nan=0.0, posinf=1e4, neginf=-1e4)

    ablation_file = RESULTS_DIR / "modularity_ablation_bbbp.json"
    ablation_data = {}
    if ablation_file.exists():
        with open(ablation_file, "r") as f:
            ablation_data = json.load(f)["raw_results"]

    split_reports = []
    seeds = list(STUDY_SEEDS)

    for seed in seeds:
        split_file = REPO_ROOT / meta["splits"][f"seed_{seed}"]["split_file"]
        with open(split_file, "r") as f:
            split_dict = json.load(f)

        train_idx = split_dict["train"]
        test_idx = split_dict["test"]

        X_tr = X_clean[train_idx]
        X_te = X_clean[test_idx]

        train_mins = X_tr.min(axis=0)
        train_maxs = X_tr.max(axis=0)

        below = X_te < (train_mins - 1e-6)
        above = X_te > (train_maxs + 1e-6)
        is_ood_matrix = below | above

        ood_desc_counts = is_ood_matrix.sum(axis=0)
        n_ood_descriptors = int(np.sum(ood_desc_counts > 0))
        pct_ood_descriptors = float(n_ood_descriptors / len(feature_names) * 100)

        mols_affected = np.any(is_ood_matrix, axis=1)
        n_mols_affected = int(np.sum(mols_affected))
        pct_mols_affected = float(n_mols_affected / len(test_idx) * 100)

        avg_ood_per_mol = float(np.mean(is_ood_matrix.sum(axis=1)))

        dir_hyb_auc = ablation_data.get("direct_hybrid", [{}])[seed - 1].get("roc_auc", float("nan"))
        dir_only_auc = ablation_data.get("direct_only", [{}])[seed - 1].get("roc_auc", float("nan"))
        res_only_auc = ablation_data.get("residual_only", [{}])[seed - 1].get("roc_auc", float("nan"))
        prior_res_auc = ablation_data.get("prior_residual", [{}])[seed - 1].get("roc_auc", float("nan"))

        report = {
            "seed": seed,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_ood_descriptors": n_ood_descriptors,
            "pct_ood_descriptors": pct_ood_descriptors,
            "n_mols_affected": n_mols_affected,
            "pct_mols_affected": pct_mols_affected,
            "avg_ood_per_mol": avg_ood_per_mol,
            "direct_hybrid_auc": dir_hyb_auc,
            "direct_only_auc": dir_only_auc,
            "residual_only_auc": res_only_auc,
            "prior_residual_auc": prior_res_auc,
        }
        split_reports.append(report)
        print(f"Seed {seed}: {n_ood_descriptors} OOD descriptors ({pct_ood_descriptors:.1f}%), "
              f"{n_mols_affected}/{len(test_idx)} mols ({pct_mols_affected:.1f}%) affected | "
              f"Direct Hybrid AUC={dir_hyb_auc:.4f}, Graph AUC={res_only_auc:.4f}")

    # The design-hardest split of the original five-split study was split 5; with
    # the ten-split rerun it stays split 5 so that the contrast remains comparable
    # and is not silently replaced by the newly added split 10.
    s5 = split_reports[4]
    others = [r for r in split_reports if r["seed"] != 5]
    ood_desc, mols, hyb = ([r[k] for r in others] for k in ("n_ood_descriptors", "pct_mols_affected", "direct_hybrid_auc"))

    print("\n🔍 Split 5 Contrast (against the other nine splits):")
    print(f"  - Split 5 OOD descriptors: {s5['n_ood_descriptors']} vs Mean(others): {np.mean(ood_desc):.1f}")
    print(f"  - Split 5 Mols affected: {s5['pct_mols_affected']:.1f}% vs Mean(others): {np.mean(mols):.1f}%")
    print(f"  - Split 5 Direct Hybrid AUC: {s5['direct_hybrid_auc']:.4f} vs Mean(others): {np.mean(hyb):.4f}")

    return {
        "n_splits": len(split_reports),
        "split_reports": split_reports,
        "split_5_contrast": {
            "split_5": s5,
            "mean_other_splits": {
                "n_ood_descriptors": float(np.mean(ood_desc)),
                "pct_mols_affected": float(np.mean(mols)),
                "direct_hybrid_auc": float(np.mean(hyb)),
            },
        },
    }

def analyze_scaffold_ood_means(datasets: tuple[str, ...] = ("esol", "lipophilicity", "bbbp")) -> dict[str, Any]:
    """Cross-split mean out-of-range descriptor statistics on the scaffold manifests.

    This is the quantity quoted in Additional file 1 ("Across all ten scaffold
    splits ..."): for every saved scaffold manifest, count the descriptors whose
    test values fall outside the training range and the fraction of test
    molecules affected by at least one such descriptor.
    """
    print("\n" + "=" * 70)
    print("🔬 E2.4: Out-of-range descriptor means across all saved scaffold splits")
    print("=" * 70)

    manifest_file = DATA_DIR / "dataset_manifest.json"
    with open(manifest_file, "r") as f:
        manifest = json.load(f)

    out: dict[str, Any] = {}
    for dataset_name in datasets:
        meta = manifest[dataset_name]
        df_clean = pd.read_csv(REPO_ROOT / meta["clean_csv"])
        smiles_col = meta.get("smiles_col", "smiles")
        if smiles_col not in df_clean.columns:
            smiles_col = "mol" if "mol" in df_clean.columns else "smiles"
        X_raw, feature_names = extract_rdkit_2d_descriptors(df_clean[smiles_col].tolist())
        X_clean = np.nan_to_num(X_raw, nan=0.0, posinf=1e4, neginf=-1e4)

        per_split = []
        for key, entry in sorted(meta["splits"].items(), key=lambda kv: int(kv[0].split("_")[1])):
            seed = int(key.split("_")[1])
            if seed not in STUDY_SEEDS:
                continue
            sfile = REPO_ROOT / entry["split_file"]
            if not sfile.exists():
                continue
            with open(sfile, "r") as f:
                split = json.load(f)
            X_tr, X_te = X_clean[split["train"]], X_clean[split["test"]]
            is_ood = (X_te < X_tr.min(axis=0) - 1e-6) | (X_te > X_tr.max(axis=0) + 1e-6)
            n_desc = int(np.sum(is_ood.sum(axis=0) > 0))
            per_split.append({
                "seed": seed,
                "n_test": len(X_te),
                "n_ood_descriptors": n_desc,
                "pct_ood_descriptors": 100.0 * n_desc / len(feature_names),
                "pct_mols_affected": 100.0 * float(np.mean(np.any(is_ood, axis=1))),
            })

        def mean_sd(field: str) -> dict[str, float]:
            vals = [r[field] for r in per_split]
            return {"mean": float(np.mean(vals)), "sd": float(np.std(vals))}

        out[dataset_name] = {
            "n_splits": len(per_split),
            "n_descriptors": len(feature_names),
            "n_ood_descriptors": mean_sd("n_ood_descriptors"),
            "pct_ood_descriptors": mean_sd("pct_ood_descriptors"),
            "pct_mols_affected": mean_sd("pct_mols_affected"),
            "per_split": per_split,
        }
        print(f"{dataset_name}: {len(per_split)} splits, "
              f"{out[dataset_name]['n_ood_descriptors']['mean']:.1f} "
              f"+/- {out[dataset_name]['n_ood_descriptors']['sd']:.1f} out-of-range descriptors, "
              f"{out[dataset_name]['pct_mols_affected']['mean']:.1f} "
              f"+/- {out[dataset_name]['pct_mols_affected']['sd']:.1f}% test molecules affected")
    return out


def analyze_scaffold_vs_random(dataset_name: str = "esol") -> dict[str, Any]:
    print("\n" + "=" * 70)
    print(f"🔬 E2.2: Scaffold vs. Random Split Distribution Shift on {dataset_name.upper()}")
    print("=" * 70)

    manifest_file = DATA_DIR / "dataset_manifest.json"
    with open(manifest_file, "r") as f:
        meta = json.load(f)[dataset_name]

    df_clean = pd.read_csv(REPO_ROOT / meta["clean_csv"])
    smiles_col = meta.get("smiles_col", "smiles")
    if smiles_col not in df_clean.columns:
        smiles_col = "mol" if "mol" in df_clean.columns else "smiles"
    smiles = df_clean[smiles_col].tolist()
    targets = df_clean[meta["targets"][0]].values.astype(np.float64)

    X_raw, feature_names = extract_rdkit_2d_descriptors(smiles)
    X_clean = np.nan_to_num(X_raw, nan=0.0, posinf=1e4, neginf=-1e4)

    split_file = REPO_ROOT / meta["splits"]["seed_1"]["split_file"]
    with open(split_file, "r") as f:
        scaffold_split = json.load(f)

    np.random.seed(42)
    n = len(smiles)
    indices = np.random.permutation(n)
    n_tr = int(n * 0.8)
    n_va = int(n * 0.1)
    random_split = {
        "train": indices[:n_tr].tolist(),
        "val": indices[n_tr:n_tr + n_va].tolist(),
        "test": indices[n_tr + n_va:].tolist(),
    }

    def evaluate_split_shift(split_dict: dict[str, list[int]], split_type: str) -> dict[str, Any]:
        tr_idx = split_dict["train"]
        te_idx = split_dict["test"]
        X_tr = X_clean[tr_idx]
        X_te = X_clean[te_idx]
        y_tr = targets[tr_idx]

        tr_min = X_tr.min(axis=0)
        tr_max = X_tr.max(axis=0)
        tr_std = np.std(X_tr, axis=0) + 1e-8
        tr_mean = np.mean(X_tr, axis=0)
        te_mean = np.mean(X_te, axis=0)

        std_shift = np.abs(te_mean - tr_mean) / tr_std
        mean_shift = float(np.mean(std_shift))
        max_shift = float(np.max(std_shift))

        below = X_te < (tr_min - 1e-6)
        above = X_te > (tr_max + 1e-6)
        is_ood = below | above
        ood_desc_count = int(np.sum(is_ood.sum(axis=0) > 0))
        pct_ood_desc = float(ood_desc_count / len(feature_names) * 100)
        pct_mols_affected = float(np.mean(np.any(is_ood, axis=1)) * 100)

        n_features = len(feature_names)
        ood_features = set()
        for j in range(n_features):
            if is_ood[:, j].sum() > 0 or np.count_nonzero(np.abs(X_tr[:, j]) > 1e-12) == 0:
                ood_features.add(j)

        redundant_features = set()
        corr_mat = np.corrcoef(X_tr.T)
        corr_mat = np.nan_to_num(corr_mat, nan=0.0)
        for i in range(n_features):
            if i in ood_features:
                continue
            for j in range(i):
                if j in ood_features or j in redundant_features:
                    continue
                if abs(corr_mat[i, j]) > 0.85:
                    redundant_features.add(i)
                    break

        spearman_scores = []
        for j in range(n_features):
            if np.std(X_tr[:, j]) < 1e-6:
                spearman_scores.append(0.0)
            else:
                r, _ = stats.spearmanr(X_tr[:, j], y_tr)
                spearman_scores.append(abs(r) if math.isfinite(r) else 0.0)

        supported_features = set()
        weak_features = set()
        for j in range(n_features):
            if j in ood_features or j in redundant_features:
                continue
            if spearman_scores[j] >= 0.15:
                supported_features.add(j)
            else:
                weak_features.add(j)

        res = {
            "split_type": split_type,
            "n_ood_descriptors": ood_desc_count,
            "pct_ood_descriptors": pct_ood_desc,
            "pct_mols_affected": pct_mols_affected,
            "mean_covariate_shift": mean_shift,
            "max_covariate_shift": max_shift,
            "evidence_class_breakdown": {
                "ood_unsupported": len(ood_features),
                "redundant": len(redundant_features),
                "supported": len(supported_features),
                "weak_uncertain": len(weak_features),
            }
        }
        print(f"[{split_type.upper()}] OOD descriptors: {ood_desc_count} ({pct_ood_desc:.1f}%), "
              f"Mols affected: {pct_mols_affected:.1f}%, Mean shift: {mean_shift:.3f}")
        print(f"  Evidence classes: {res['evidence_class_breakdown']}")
        return res

    scaffold_res = evaluate_split_shift(scaffold_split, "scaffold")
    random_res = evaluate_split_shift(random_split, "random")

    return {
        "dataset": dataset_name,
        "scaffold": scaffold_res,
        "random": random_res,
    }

def audit_dataset_hygiene() -> dict[str, Any]:
    print("\n" + "=" * 70)
    print("🔬 E2.3: Dataset Hygiene & Structural Collision Audit (Schuh et al. 2023)")
    print("=" * 70)

    datasets = ["esol", "freesolv", "bbbp", "lipophilicity", "bace"]
    manifest_file = DATA_DIR / "dataset_manifest.json"
    with open(manifest_file, "r") as f:
        all_meta = json.load(f)

    audit_results = {}

    for d in datasets:
        meta = all_meta[d]
        df_clean = pd.read_csv(REPO_ROOT / meta["clean_csv"])
        smiles_col = meta.get("smiles_col", "smiles")
        if smiles_col not in df_clean.columns:
            smiles_col = "mol" if "mol" in df_clean.columns else "smiles"
        target_col = meta["targets"][0]

        raw_smiles = df_clean[smiles_col].tolist()
        raw_targets = df_clean[target_col].values
        is_classification = "classification" in meta["task_type"]

        canonical_smiles = []
        inchikeys = []
        valid_indices = []

        for i, s in enumerate(raw_smiles):
            try:
                m = Chem.MolFromSmiles(s)
                if m is not None:
                    cs = Chem.MolToSmiles(m)
                    ik = Chem.MolToInchiKey(m)
                    canonical_smiles.append(cs)
                    inchikeys.append(ik)
                    valid_indices.append(i)
            except Exception:
                pass

        targets_valid = raw_targets[valid_indices]
        n_total = len(canonical_smiles)

        df_audit = pd.DataFrame({
            "smiles": canonical_smiles,
            "inchikey": inchikeys,
            "target": targets_valid,
        })

        unique_inchikeys = df_audit["inchikey"].nunique()
        duplicate_groups = df_audit.groupby("inchikey").filter(lambda g: len(g) > 1)
        n_duplicate_mols = len(duplicate_groups)
        n_duplicate_structures = duplicate_groups["inchikey"].nunique()

        conflicting_groups = 0
        max_discrepancy = 0.0

        for ik, group in duplicate_groups.groupby("inchikey"):
            vals = group["target"].values
            if is_classification:
                if len(np.unique(vals)) > 1:
                    conflicting_groups += 1
            else:
                span = float(np.max(vals) - np.min(vals))
                if span > max_discrepancy:
                    max_discrepancy = span

                if span > 0.5 * float(np.std(targets_valid)):
                    conflicting_groups += 1

        d_report = {
            "dataset": d,
            "task_type": meta["task_type"],
            "total_molecules": n_total,
            "unique_structures_inchikey": unique_inchikeys,
            "duplicate_molecules": n_duplicate_mols,
            "duplicate_structures": n_duplicate_structures,
            "pct_duplicate_molecules": float(n_duplicate_mols / n_total * 100),
            "conflicting_duplicate_groups": conflicting_groups,
            "max_discrepancy": max_discrepancy if not is_classification else ("N/A (binary)"),
        }
        audit_results[d] = d_report
        print(f"[{d.upper()}] Total: {n_total} | Unique InChIKeys: {unique_inchikeys} | "
              f"Duplicates: {n_duplicate_mols} ({d_report['pct_duplicate_molecules']:.1f}%) | "
              f"Conflicting: {conflicting_groups} groups (Max discrepancy: {max_discrepancy:.3f} if reg)")

    return audit_results

def main():
    bbbp_analysis = analyze_bbbp_splits()
    ood_means = analyze_scaffold_ood_means()
    scaffold_shift_esol = analyze_scaffold_vs_random("esol")
    scaffold_shift_lipo = analyze_scaffold_vs_random("lipophilicity")
    hygiene_audit = audit_dataset_hygiene()

    full_case_study = {
        "bbbp_hard_split_analysis": bbbp_analysis,
        "scaffold_ood_means": ood_means,
        "scaffold_vs_random_shift": {
            "esol": scaffold_shift_esol,
            "lipophilicity": scaffold_shift_lipo,
        },
        "dataset_hygiene_audit": hygiene_audit,
    }

    out_file = RESULTS_DIR / "scorecard_case_study.json"
    with open(out_file, "w") as f:
        json.dump(full_case_study, f, indent=2)

    print("\n" + "=" * 70)
    print(f"✅ Experiment E2 complete. Results saved to: {out_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()
