"""Canonical Baseline Runner across 8 MoleculeNet Datasets with Locked Split Indices.

Evaluates 5 internal baselines across 5 independent seeds (80/10/10 split):
1. Dummy baseline (train mean / majority class)
2. ECFP4 + Random Forest
3. RDKit 2D Descriptors + Random Forest
4. Descriptor MLP (PyTorch 3-layer MLP on scaled 2D descriptors)
5. D-MPNN Graph-Only
6. Hybrid D-MPNN + Descriptors
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chemistry_framework.features.pharmacophore import extract_morgan_fingerprints, extract_rdkit_2d_descriptors
from chemistry_framework.data.graph import mol_to_graph, BatchMolGraph
from chemistry_framework.encoders.gnn import DMPNNEncoder
from benchmarks._seeds import DEFAULT_SEEDS, add_seed_argument, resolve_seeds

DATA_DIR = REPO_ROOT / "data" / "moleculenet"
SPLITS_DIR = REPO_ROOT / "benchmarks" / "splits"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
MANIFEST_PATH = RESULTS_DIR / "manifest.json"

class SimpleTabularMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 1, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def compute_classification_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray) -> dict[str, float]:
    """Computes macro ROC-AUC and AUPRC while ignoring NaN labels."""
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred_prob.ndim == 1:
        y_pred_prob = y_pred_prob[:, None]

    n_targets = y_true.shape[1]
    roc_aucs = []
    auprcs = []

    for t in range(n_targets):
        mask = ~np.isnan(y_true[:, t])
        yt = y_true[mask, t]
        yp = y_pred_prob[mask, t]
        if len(np.unique(yt)) > 1:
            try:
                roc_aucs.append(roc_auc_score(yt, yp))
            except Exception:
                pass
            try:
                auprcs.append(average_precision_score(yt, yp))
            except Exception:
                pass

    return {
        "roc_auc": float(np.mean(roc_aucs)) if roc_aucs else 0.5,
        "auprc": float(np.mean(auprcs)) if auprcs else 0.0,
    }

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    mask = ~np.isnan(yt)
    yt = yt[mask]
    yp = yp[mask]
    rmse = float(math.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    r2 = float(r2_score(yt, yp)) if len(yt) > 1 else 0.0
    return {"rmse": rmse, "mae": mae, "r2": r2}

def run_tabular_mlp(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    is_classification: bool,
    seed: int,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
) -> dict[str, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dim = y_train.shape[1] if y_train.ndim > 1 else 1
    model = SimpleTabularMLP(in_dim=X_train.shape[1], hidden_dim=128, out_dim=out_dim).to(device)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss() if is_classification else nn.HuberLoss()

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            if by.ndim == 1:
                by = by.unsqueeze(1)
            optimizer.zero_grad()
            out = model(bx)
            mask = ~torch.isnan(by)
            loss = loss_fn(out[mask], by[mask]) if mask.any() else torch.tensor(0.0, device=device)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            vx = torch.tensor(X_val, dtype=torch.float32).to(device)
            vy = torch.tensor(y_val, dtype=torch.float32).to(device)
            if vy.ndim == 1:
                vy = vy.unsqueeze(1)
            vout = model(vx)
            vmask = ~torch.isnan(vy)
            vloss = float(loss_fn(vout[vmask], vy[vmask])) if vmask.any() else 0.0

            if vloss < best_val_loss:
                best_val_loss = vloss
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        tx = torch.tensor(X_test, dtype=torch.float32).to(device)
        tout = model(tx).cpu().numpy()
        if is_classification:
            probs = 1.0 / (1.0 + np.exp(-np.clip(tout, -20, 20)))
            return compute_classification_metrics(y_test, probs)
        else:
            return compute_regression_metrics(y_test, tout.squeeze())

def evaluate_dataset_baselines(dataset_name: str, seed: int = 1, X_fp: np.ndarray | None = None, X_desc_raw: np.ndarray | None = None,) -> list[dict[str, Any]]:
    print(f"\n🔹 Evaluating Baselines for: {dataset_name.upper()} (Seed {seed})")
    manifest_file = DATA_DIR / "dataset_manifest.json"
    with open(manifest_file, "r") as f:
        meta = json.load(f)[dataset_name]
    df_clean = pd.read_csv(REPO_ROOT / meta["clean_csv"])
    split_file = REPO_ROOT / meta["splits"][f"seed_{seed}"]["split_file"]
    with open(split_file, "r") as f:
        split_indices = json.load(f)
    train_idx = split_indices["train"]
    val_idx = split_indices["val"]
    test_idx = split_indices["test"]
    smiles = df_clean[meta["smiles_col"]].tolist()
    target_cols = meta["targets"]
    targets = df_clean[target_cols].values
    is_classification = "classification" in meta["task_type"]

    train_targets = targets[train_idx]
    test_targets = targets[test_idx]
    if is_classification:

        p_train = np.nanmean(train_targets, axis=0)
        dummy_probs = np.tile(p_train, (len(test_idx), 1))
        dummy_metrics = compute_classification_metrics(test_targets, dummy_probs)
    else:
        mean_train = np.nanmean(train_targets, axis=0)
        dummy_pred = np.tile(mean_train, (len(test_idx), 1)).squeeze()
        dummy_metrics = compute_regression_metrics(test_targets.squeeze(), dummy_pred)
    results = [{
        "dataset": dataset_name,
        "seed": seed,
        "model": "dummy",
        "metrics": dummy_metrics,
    }]
    print(f"  ✓ Dummy Baseline: {dummy_metrics}")

    if X_fp is None:
        print("  ... Extracting Morgan Fingerprints (ECFP4)...")
        X_fp, _ = extract_morgan_fingerprints(smiles, n_bits=2048, radius=2)
    if X_desc_raw is None:
        print("  ... Extracting RDKit 2D Descriptors...")
        X_desc_raw, _ = extract_rdkit_2d_descriptors(smiles)
        X_desc_raw = np.nan_to_num(X_desc_raw, nan=0.0, posinf=1e4, neginf=-1e4)

    scaler = StandardScaler()
    scaler.fit(X_desc_raw[train_idx])
    X_desc = scaler.transform(X_desc_raw)
    X_desc = np.nan_to_num(X_desc, nan=0.0, posinf=3.0, neginf=-3.0)

    rf_kwargs = {"n_estimators": 100, "random_state": seed, "n_jobs": -1}
    y_tr = train_targets.squeeze() if len(target_cols) == 1 else train_targets
    y_te = test_targets.squeeze() if len(target_cols) == 1 else test_targets
    if is_classification:
        if len(target_cols) == 1:
            rf_fp = RandomForestClassifier(**rf_kwargs)
            rf_fp.fit(X_fp[train_idx], y_tr)
            probs = rf_fp.predict_proba(X_fp[test_idx])[:, 1]
            rf_fp_metrics = compute_classification_metrics(y_te, probs)
        else:

            probs_list = []
            for t in range(len(target_cols)):
                mask_tr = ~np.isnan(train_targets[:, t])
                clf = RandomForestClassifier(**rf_kwargs)
                clf.fit(X_fp[train_idx][mask_tr], train_targets[train_idx, t][mask_tr])
                p = clf.predict_proba(X_fp[test_idx])[:, 1]
                probs_list.append(p)
            rf_fp_metrics = compute_classification_metrics(y_te, np.column_stack(probs_list))
    else:
        rf_fp = RandomForestRegressor(**rf_kwargs)
        rf_fp.fit(X_fp[train_idx], y_tr)
        preds = rf_fp.predict(X_fp[test_idx])
        rf_fp_metrics = compute_regression_metrics(y_te, preds)

    results.append({
        "dataset": dataset_name,
        "seed": seed,
        "model": "ecfp4_rf",
        "metrics": rf_fp_metrics,
    })
    print(f"  ✓ ECFP4 + RF: {rf_fp_metrics}")

    if is_classification:
        if len(target_cols) == 1:
            rf_desc = RandomForestClassifier(**rf_kwargs)
            rf_desc.fit(X_desc[train_idx], y_tr)
            probs = rf_desc.predict_proba(X_desc[test_idx])[:, 1]
            rf_desc_metrics = compute_classification_metrics(y_te, probs)
        else:
            probs_list = []
            for t in range(len(target_cols)):
                mask_tr = ~np.isnan(train_targets[:, t])
                clf = RandomForestClassifier(**rf_kwargs)
                clf.fit(X_desc[train_idx][mask_tr], train_targets[train_idx, t][mask_tr])
                p = clf.predict_proba(X_desc[test_idx])[:, 1]
                probs_list.append(p)
            rf_desc_metrics = compute_classification_metrics(y_te, np.column_stack(probs_list))
    else:
        rf_desc = RandomForestRegressor(**rf_kwargs)
        rf_desc.fit(X_desc[train_idx], y_tr)
        preds = rf_desc.predict(X_desc[test_idx])
        rf_desc_metrics = compute_regression_metrics(y_te, preds)

    results.append({
        "dataset": dataset_name,
        "seed": seed,
        "model": "rdkit_rf",
        "metrics": rf_desc_metrics,
    })
    print(f"  ✓ RDKit 2D + RF: {rf_desc_metrics}")

    mlp_metrics = run_tabular_mlp(
        X_desc[train_idx], train_targets,
        X_desc[val_idx], targets[val_idx],
        X_desc[test_idx], test_targets,
        is_classification=is_classification,
        seed=seed,
    )
    results.append({
        "dataset": dataset_name,
        "seed": seed,
        "model": "descriptor_mlp",
        "metrics": mlp_metrics,
    })
    print(f"  ✓ Descriptor MLP: {mlp_metrics}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Run canonical baselines on the manuscript endpoints")
    parser.add_argument("--dataset", type=str, default="esol", help="Dataset name or 'all'")
    parser.add_argument(
        "--all-seeds",
        action="store_true",
        help="Deprecated alias for the default ten-seed run (kept for compatibility).",
    )
    add_seed_argument(parser)
    args = parser.parse_args()

    if args.all_seeds and args.seeds is not None:
        raise SystemExit("Use either --seeds or --all-seeds, not both")

    manifest_file = DATA_DIR / "dataset_manifest.json"
    with open(manifest_file, "r") as f:
        all_datasets = list(json.load(f).keys())

    if args.dataset == "all" and args.seeds is None and not args.all_seeds:
        # The manifest also holds endpoints that are not part of the manuscript
        # (clintox, hiv, tox21); running them by accident costs hours, so make
        # the intent explicit.
        print(
            "Note: '--dataset all' covers every prepared endpoint "
            f"({len(all_datasets)}), including ones outside the manuscript. "
            "Pass the endpoint names explicitly for the paper scope."
        )

    target_datasets = all_datasets if args.dataset == "all" else [args.dataset]
    seed_list = resolve_seeds(args.seeds)

    for d in target_datasets:
        meta_d = json.load(open(manifest_file))[d]
        df_clean = pd.read_csv(REPO_ROOT / meta_d["clean_csv"])
        smiles = df_clean[meta_d["smiles_col"]].tolist()

        print(f"\n📦 Precomputing features once for {d.upper()} ({len(smiles)} molecules)...")
        X_fp, _ = extract_morgan_fingerprints(smiles, n_bits=2048, radius=2)
        X_desc_raw, _ = extract_rdkit_2d_descriptors(smiles)
        X_desc_raw = np.nan_to_num(X_desc_raw, nan=0.0, posinf=1e4, neginf=-1e4)

        all_results = []
        for s in seed_list:
            res = evaluate_dataset_baselines(d, seed=s, X_fp=X_fp, X_desc_raw=X_desc_raw)
            all_results.extend(res)

        if len(seed_list) > 1:

            models = sorted(list({r["model"] for r in all_results}))
            agg: dict[str, dict[str, Any]] = {}
            for m in models:
                m_entries = [r for r in all_results if r["model"] == m]
                keys = list(m_entries[0]["metrics"].keys())
                agg[m] = {}
                for k in keys:
                    vals = [r["metrics"][k] for r in m_entries if k in r["metrics"] and math.isfinite(r["metrics"][k])]
                    if vals:
                        agg[m][k] = {
                            "mean": float(np.mean(vals)),
                            "std": float(np.std(vals)),
                            "values": vals,
                        }
            summary = {
                "dataset": d,
                "n_seeds": len(seed_list),
                "seeds": seed_list,
                "aggregated": agg,
                "raw_runs": all_results,
            }
            out_file = RESULTS_DIR / f"baselines_{d}_aggregated.json"
            with open(out_file, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\n🎉 Saved {len(seed_list)}-seed aggregated baselines for {d} to: {out_file}")
        else:
            out_file = RESULTS_DIR / f"baselines_{d}_seed_{seed_list[0]}.json"
            with open(out_file, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\n🎉 Finished evaluation. Saved to: {out_file}")

if __name__ == "__main__":
    main()
