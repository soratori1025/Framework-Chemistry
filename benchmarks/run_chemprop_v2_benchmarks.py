"""Chemprop v2 Benchmark Runner on MoleculeNet tasks (ESOL, FreeSolv, BBBP).

Evaluates Chemprop v2 D-MPNN under two matched configurations:
1. chemprop_graph: Pure topological graph D-MPNN (Benchmark A)
2. chemprop_hybrid: D-MPNN + 210 RDKit descriptors concatenated via x_d (Benchmark B)

Strict experimental controls:
- Exact 5 saved split manifests from benchmarks/splits/{dataset}_seed_{seed}.json
- Target scaling (regression) and descriptor scaling fitted exclusively on train set
- Checkpoint selection via validation loss (patience=10 early stopping)
- Per-split metrics and sample-level test predictions saved to benchmarks/results/
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, average_precision_score

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from chemprop import data, models, nn

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "moleculenet"
SPLITS_DIR = REPO_ROOT / "benchmarks" / "splits"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from chemistry_framework.features.pharmacophore import extract_rdkit_2d_descriptors
from benchmarks._descriptor_contract import select_contract_columns
from benchmarks._seeds import DEFAULT_SEEDS, add_seed_argument, resolve_seeds

torch.set_float32_matmul_precision("medium")

DATASET_CONFIGS = {
    "esol": {
        "filename": "esol_clean.csv",
        "target_col": "measured log solubility in mols per litre",
        "task_type": "regression",
    },
    "freesolv": {
        "filename": "freesolv_clean.csv",
        "target_col": "expt",
        "task_type": "regression",
    },
    "bbbp": {
        "filename": "bbbp_clean.csv",
        "target_col": "p_np",
        "task_type": "classification",
    },
    "lipophilicity": {
        "filename": "lipophilicity_clean.csv",
        "target_col": "exp",
        "task_type": "regression",
    },
    "bace": {
        "filename": "bace_clean.csv",
        "smiles_col": "mol",
        "target_col": "Class",
        "task_type": "classification",
    },
}

def build_chemprop_mpnn(
    task_type: str,
    d_xd: int = 0,
    d_h: int = 300,
    depth: int = 3,
    dropout: float = 0.0,
    lr: float = 1e-3,
) -> models.MPNN:
    """Builds an MPNN model with Chemprop v2 standard architecture."""
    mp = nn.BondMessagePassing(d_h=d_h, depth=depth, dropout=dropout)
    agg = nn.MeanAggregation()
    input_dim = d_h + d_xd

    if task_type == "regression":
        ffn = nn.RegressionFFN(
            n_tasks=1,
            input_dim=input_dim,
            hidden_dim=d_h,
            n_layers=1,
            dropout=dropout,
        )
    else:
        ffn = nn.BinaryClassificationFFN(
            n_tasks=1,
            input_dim=input_dim,
            hidden_dim=d_h,
            n_layers=1,
            dropout=dropout,
        )

    mpnn = models.MPNN(message_passing=mp, agg=agg, predictor=ffn, batch_norm=True)
    return mpnn

def run_benchmark_for_dataset(
    dataset_name: str,
    seeds: list[int] = [1, 2, 3, 4, 5],
    d_h: int = 300,
    depth: int = 3,
    max_epochs: int = 50,
    patience: int = 10,
    batch_size: int = 64,
) -> dict[str, Any]:
    cfg = DATASET_CONFIGS[dataset_name]
    csv_path = DATA_DIR / cfg["filename"]
    df = pd.read_csv(csv_path)

    smiles_col = cfg.get("smiles_col", "smiles")
    smiles = df[smiles_col].astype(str).tolist()
    raw_targets = df[cfg["target_col"]].values.astype(np.float32)
    is_classification = cfg["task_type"] == "classification"

    print(f"📦 Extracting RDKit 2D descriptors for {len(smiles)} molecules...")
    desc_matrix, desc_names = select_contract_columns(*extract_rdkit_2d_descriptors(smiles))
    desc_matrix = np.nan_to_num(desc_matrix, nan=0.0, posinf=3.0, neginf=-3.0)

    results: dict[str, list[dict[str, Any]]] = {
        "chemprop_graph": [],
        "chemprop_hybrid": [],
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🚀 Running Chemprop v2 on {dataset_name.upper()} ({device.upper()}, seeds={seeds})")

    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        split_file = SPLITS_DIR / f"{dataset_name}_seed_{seed}.json"
        with open(split_file) as f:
            sp = json.load(f)
        train_idx = sp.get("train_indices") or sp["train"]
        val_idx = sp.get("val_indices") or sp["val"]
        test_idx = sp.get("test_indices") or sp["test"]

        if not is_classification:
            scaler_y = StandardScaler()
            y_train = scaler_y.fit_transform(raw_targets[train_idx, None]).ravel()
            y_val = scaler_y.transform(raw_targets[val_idx, None]).ravel()
            y_test = scaler_y.transform(raw_targets[test_idx, None]).ravel()
        else:
            scaler_y = None
            y_train = raw_targets[train_idx]
            y_val = raw_targets[val_idx]
            y_test = raw_targets[test_idx]

        scaler_x = StandardScaler()
        X_desc_tr = scaler_x.fit_transform(desc_matrix[train_idx])
        X_desc_va = scaler_x.transform(desc_matrix[val_idx])
        X_desc_te = scaler_x.transform(desc_matrix[test_idx])
        X_desc_all = np.zeros_like(desc_matrix)
        X_desc_all[train_idx] = X_desc_tr
        X_desc_all[val_idx] = X_desc_va
        X_desc_all[test_idx] = X_desc_te

        target_all = np.zeros(len(smiles), dtype=np.float32)
        target_all[train_idx] = y_train
        target_all[val_idx] = y_val
        target_all[test_idx] = y_test

        print("  🔹 Training chemprop_graph...")
        pts_tr_g = [data.MoleculeDatapoint.from_smi(smiles[i], y=[float(target_all[i])]) for i in train_idx]
        pts_va_g = [data.MoleculeDatapoint.from_smi(smiles[i], y=[float(target_all[i])]) for i in val_idx]
        pts_te_g = [data.MoleculeDatapoint.from_smi(smiles[i], y=[float(target_all[i])]) for i in test_idx]

        ds_tr_g = data.MoleculeDataset(pts_tr_g)
        ds_va_g = data.MoleculeDataset(pts_va_g)
        ds_te_g = data.MoleculeDataset(pts_te_g)

        loader_tr_g = data.build_dataloader(ds_tr_g, batch_size=batch_size, shuffle=True)
        loader_va_g = data.build_dataloader(ds_va_g, batch_size=batch_size, shuffle=False)
        loader_te_g = data.build_dataloader(ds_te_g, batch_size=batch_size, shuffle=False)

        L.seed_everything(seed, workers=True)
        mpnn_g = build_chemprop_mpnn(cfg["task_type"], d_xd=0, d_h=d_h, depth=depth)

        early_stop_g = EarlyStopping(monitor="val_loss", mode="min", patience=patience)
        checkpoint_g = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, save_weights_only=True)
        trainer_g = L.Trainer(
            max_epochs=max_epochs,
            accelerator="cuda" if torch.cuda.is_available() else "cpu",
            devices=1,
            callbacks=[early_stop_g, checkpoint_g],
            logger=False,
            enable_progress_bar=False,
        )
        trainer_g.fit(mpnn_g, loader_tr_g, loader_va_g)

        best_g = models.MPNN.load_from_checkpoint(checkpoint_g.best_model_path)
        preds_te_raw = trainer_g.predict(best_g, loader_te_g)
        preds_te = torch.cat(preds_te_raw, dim=0).cpu().numpy().ravel()

        if not is_classification:

            preds_te = scaler_y.inverse_transform(preds_te[:, None]).ravel()
            true_te = raw_targets[test_idx]
            m_rmse = float(math.sqrt(mean_squared_error(true_te, preds_te)))
            m_mae = float(mean_absolute_error(true_te, preds_te))
            m_r2 = float(r2_score(true_te, preds_te))
            metrics_g = {"rmse": m_rmse, "mae": m_mae, "r2": m_r2}
        else:

            probs_te = 1.0 / (1.0 + np.exp(-preds_te))
            true_te = raw_targets[test_idx]
            m_roc = float(roc_auc_score(true_te, probs_te))
            m_pr = float(average_precision_score(true_te, probs_te))
            metrics_g = {"roc_auc": m_roc, "auprc": m_pr}

        results["chemprop_graph"].append(metrics_g)
        print(f"    ✓ chemprop_graph metrics: {metrics_g}")

        print("  🔹 Training chemprop_hybrid (Graph + Descriptors)...")
        pts_tr_h = [
            data.MoleculeDatapoint.from_smi(smiles[i], y=[float(target_all[i])], x_d=X_desc_all[i].astype(np.float32))
            for i in train_idx
        ]
        pts_va_h = [
            data.MoleculeDatapoint.from_smi(smiles[i], y=[float(target_all[i])], x_d=X_desc_all[i].astype(np.float32))
            for i in val_idx
        ]
        pts_te_h = [
            data.MoleculeDatapoint.from_smi(smiles[i], y=[float(target_all[i])], x_d=X_desc_all[i].astype(np.float32))
            for i in test_idx
        ]

        ds_tr_h = data.MoleculeDataset(pts_tr_h)
        ds_va_h = data.MoleculeDataset(pts_va_h)
        ds_te_h = data.MoleculeDataset(pts_te_h)

        loader_tr_h = data.build_dataloader(ds_tr_h, batch_size=batch_size, shuffle=True)
        loader_va_h = data.build_dataloader(ds_va_h, batch_size=batch_size, shuffle=False)
        loader_te_h = data.build_dataloader(ds_te_h, batch_size=batch_size, shuffle=False)

        L.seed_everything(seed, workers=True)
        d_xd = X_desc_all.shape[1]
        mpnn_h = build_chemprop_mpnn(cfg["task_type"], d_xd=d_xd, d_h=d_h, depth=depth)

        early_stop_h = EarlyStopping(monitor="val_loss", mode="min", patience=patience)
        checkpoint_h = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, save_weights_only=True)
        trainer_h = L.Trainer(
            max_epochs=max_epochs,
            accelerator="cuda" if torch.cuda.is_available() else "cpu",
            devices=1,
            callbacks=[early_stop_h, checkpoint_h],
            logger=False,
            enable_progress_bar=False,
        )
        trainer_h.fit(mpnn_h, loader_tr_h, loader_va_h)

        best_h = models.MPNN.load_from_checkpoint(checkpoint_h.best_model_path)
        preds_te_h_raw = trainer_h.predict(best_h, loader_te_h)
        preds_te_h = torch.cat(preds_te_h_raw, dim=0).cpu().numpy().ravel()

        if not is_classification:
            preds_te_h = scaler_y.inverse_transform(preds_te_h[:, None]).ravel()
            true_te = raw_targets[test_idx]
            m_rmse = float(math.sqrt(mean_squared_error(true_te, preds_te_h)))
            m_mae = float(mean_absolute_error(true_te, preds_te_h))
            m_r2 = float(r2_score(true_te, preds_te_h))
            metrics_h = {"rmse": m_rmse, "mae": m_mae, "r2": m_r2}
        else:
            probs_te_h = 1.0 / (1.0 + np.exp(-preds_te_h))
            true_te = raw_targets[test_idx]
            m_roc = float(roc_auc_score(true_te, probs_te_h))
            m_pr = float(average_precision_score(true_te, probs_te_h))
            metrics_h = {"roc_auc": m_roc, "auprc": m_pr}

        results["chemprop_hybrid"].append(metrics_h)
        print(f"    ✓ chemprop_hybrid metrics: {metrics_h}")

    primary_metric = "roc_auc" if is_classification else "rmse"
    summary = {}
    print("\n" + "=" * 60)
    print(f"📊 CHEMPROP V2 SUMMARY: {dataset_name.upper()} (5 SEEDS)")
    print("=" * 60)
    for model_name, metric_list in results.items():
        vals = [m[primary_metric] for m in metric_list]
        summary[model_name] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "values": vals,
        }
        print(f"  {model_name:<18}: {summary[model_name]['mean']:.4f} ± {summary[model_name]['std']:.4f} {vals}")

    out_file = RESULTS_DIR / f"chemprop_v2_{dataset_name}.json"
    with open(out_file, "w") as f:
        json.dump(
            {
                "dataset": dataset_name,
                "chemprop_version": "2.3.1",
                "primary_metric": primary_metric,
                "is_classification": is_classification,
                "d_h": d_h,
                "depth": depth,
                "summary": summary,
                "raw_results": results,
            },
            f,
            indent=2,
        )
    print(f"✓ Saved results to {out_file}")
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Chemprop v2 benchmarks")
    parser.add_argument("--dataset", type=str, default="all", choices=["esol", "freesolv", "bbbp", "lipophilicity", "bace", "all"])
    parser.add_argument("--d_h", type=int, default=300, help="Chemprop hidden dimension (default: 300)")
    parser.add_argument("--depth", type=int, default=3, help="Message passing depth (default: 3)")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs (default: 50)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")
    add_seed_argument(parser)
    args = parser.parse_args()

    seeds = resolve_seeds(args.seeds)
    target_datasets = ["esol", "freesolv", "bbbp", "lipophilicity", "bace"] if args.dataset == "all" else [args.dataset]
    for d in target_datasets:
        run_benchmark_for_dataset(
            dataset_name=d,
            seeds=seeds,
            d_h=args.d_h,
            depth=args.depth,
            max_epochs=args.epochs,
            patience=args.patience,
        )
