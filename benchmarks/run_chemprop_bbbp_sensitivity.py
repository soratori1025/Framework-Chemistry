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
from sklearn.metrics import roc_auc_score, average_precision_score

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from chemprop import data, models, nn

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "moleculenet"
SPLITS_DIR = REPO_ROOT / "benchmarks" / "splits" / "bbbp_sensitivity"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from chemistry_framework.features.pharmacophore import extract_rdkit_2d_descriptors
from benchmarks._descriptor_contract import select_contract_columns
from benchmarks._seeds import add_seed_argument, resolve_seeds

torch.set_float32_matmul_precision("medium")

def build_chemprop_mpnn(
    d_xd: int = 0,
    d_h: int = 300,
    depth: int = 3,
    dropout: float = 0.0,
) -> models.MPNN:
    mp = nn.BondMessagePassing(d_h=d_h, depth=depth, dropout=dropout)
    agg = nn.MeanAggregation()
    input_dim = d_h + d_xd
    ffn = nn.BinaryClassificationFFN(
        n_tasks=1,
        input_dim=input_dim,
        hidden_dim=d_h,
        n_layers=1,
        dropout=dropout,
    )
    return models.MPNN(message_passing=mp, agg=agg, predictor=ffn, batch_norm=True)

def run_bbbp_sensitivity_chemprop(
    seeds: list[int] | None = None,
    d_h: int = 300,
    depth: int = 3,
    max_epochs: int = 50,
    patience: int = 10,
    batch_size: int = 64,
) -> dict[str, Any]:
    csv_path = DATA_DIR / "bbbp_sensitivity_clean.csv"
    df = pd.read_csv(csv_path)

    seed_list = resolve_seeds(seeds)

    smiles = df["smiles"].astype(str).tolist()
    raw_targets = df["p_np"].values.astype(np.float32)

    desc_matrix, desc_names = select_contract_columns(*extract_rdkit_2d_descriptors(smiles))
    desc_matrix = np.nan_to_num(desc_matrix, nan=0.0, posinf=3.0, neginf=-3.0)

    results: dict[str, list[dict[str, Any]]] = {
        "chemprop_graph": [],
        "chemprop_hybrid": [],
    }

    for seed in seed_list:
        split_file = SPLITS_DIR / f"bbbp_sensitivity_seed_{seed}.json"
        with open(split_file) as f:
            sp = json.load(f)
        train_idx = sp["train"]
        val_idx = sp["val"]
        test_idx = sp["test"]

        scaler_x = StandardScaler()
        X_desc_tr = scaler_x.fit_transform(desc_matrix[train_idx])
        X_desc_va = scaler_x.transform(desc_matrix[val_idx])
        X_desc_te = scaler_x.transform(desc_matrix[test_idx])
        X_desc_all = np.zeros_like(desc_matrix)
        X_desc_all[train_idx] = X_desc_tr
        X_desc_all[val_idx] = X_desc_va
        X_desc_all[test_idx] = X_desc_te

        target_all = np.zeros(len(smiles), dtype=np.float32)
        target_all[train_idx] = raw_targets[train_idx]
        target_all[val_idx] = raw_targets[val_idx]
        target_all[test_idx] = raw_targets[test_idx]

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
        mpnn_g = build_chemprop_mpnn(d_xd=0, d_h=d_h, depth=depth)

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

        probs_te = 1.0 / (1.0 + np.exp(-preds_te))
        true_te = raw_targets[test_idx]
        m_roc = float(roc_auc_score(true_te, probs_te))
        m_pr = float(average_precision_score(true_te, probs_te))
        results["chemprop_graph"].append({"roc_auc": m_roc, "auprc": m_pr})

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
        mpnn_h = build_chemprop_mpnn(d_xd=d_xd, d_h=d_h, depth=depth)

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

        probs_te_h = 1.0 / (1.0 + np.exp(-preds_te_h))
        m_roc_h = float(roc_auc_score(true_te, probs_te_h))
        m_pr_h = float(average_precision_score(true_te, probs_te_h))
        results["chemprop_hybrid"].append({"roc_auc": m_roc_h, "auprc": m_pr_h})

    summary = {}
    for model_name, metric_list in results.items():
        vals = [m["roc_auc"] for m in metric_list]
        summary[model_name] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "values": vals,
        }

    out_file = RESULTS_DIR / "chemprop_bbbp_sensitivity.json"
    with open(out_file, "w") as f:
        json.dump(
            {
                "dataset": "bbbp_sensitivity",
                "chemprop_version": "2.3.1",
                "primary_metric": "roc_auc",
                "seeds": seed_list,
                "summary": summary,
                "raw_results": results,
            },
            f,
            indent=2,
        )
    print(f"Results saved to {out_file}")
    for m, s in summary.items():
        print(f"  {m}: {s['mean']:.4f} +- {s['std']:.4f} {s['values']}")
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chemprop v2.3.1 BBBP duplicate-controlled sensitivity study")
    add_seed_argument(parser)
    args = parser.parse_args()
    run_bbbp_sensitivity_chemprop(seeds=args.seeds)
