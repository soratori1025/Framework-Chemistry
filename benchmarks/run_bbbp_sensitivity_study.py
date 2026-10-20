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
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chemistry_framework.data.graph import BatchMolGraph, mol_to_graph
from chemistry_framework.encoders.gnn import DMPNNEncoder
from chemistry_framework.features.pharmacophore import extract_rdkit_2d_descriptors
from benchmarks._seeds import add_seed_argument, resolve_seeds

DATA_DIR = REPO_ROOT / "data" / "moleculenet"
SPLITS_DIR = REPO_ROOT / "benchmarks" / "splits" / "bbbp_sensitivity"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

class MoleculeGraphDataset(Dataset):
    def __init__(self, graphs: list[Any], descriptors: np.ndarray, targets: np.ndarray):
        self.graphs = graphs
        self.descriptors = torch.tensor(descriptors, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> tuple[Any, torch.Tensor, torch.Tensor]:
        return self.graphs[idx], self.descriptors[idx], self.targets[idx]

def collate_graph_batch(batch: list[tuple[Any, torch.Tensor, torch.Tensor]]):
    graphs, descs, targets = zip(*batch)
    batch_graph = BatchMolGraph(graphs)
    return batch_graph, torch.stack(descs), torch.stack(targets)

class DMPNNGraphOnly(nn.Module):
    def __init__(self, hidden_dim: int = 128, depth: int = 3, out_dim: int = 1):
        super().__init__()
        self.graph_encoder = DMPNNEncoder(hidden_dim=hidden_dim, depth=depth, use_virtual_node=False, aggregation="sum")
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, batch_graph: BatchMolGraph, desc: torch.Tensor | None = None) -> torch.Tensor:
        h = self.graph_encoder(batch_graph)
        return self.head(h)

class DMPNNHybrid(nn.Module):
    def __init__(self, desc_dim: int, hidden_dim: int = 128, depth: int = 3, out_dim: int = 1):
        super().__init__()
        self.graph_encoder = DMPNNEncoder(hidden_dim=hidden_dim, depth=depth, use_virtual_node=False, aggregation="sum")
        self.desc_encoder = nn.Sequential(
            nn.Linear(desc_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, batch_graph: BatchMolGraph, desc: torch.Tensor) -> torch.Tensor:
        hg = self.graph_encoder(batch_graph)
        hd = self.desc_encoder(desc)
        fused = torch.cat([hg, hd], dim=-1)
        return self.head(fused)

def train_and_eval(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    seed: int,
    epochs: int = 40,
    lr: float = 1e-3,
) -> dict[str, float]:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        for batch_graph, bdesc, by in train_loader:
            batch_graph = batch_graph.to(device)
            bdesc = bdesc.to(device)
            by = by.to(device)
            if by.ndim == 1:
                by = by.unsqueeze(1)
            optimizer.zero_grad()
            out = model(batch_graph, bdesc)
            mask = ~torch.isnan(by)
            loss = loss_fn(out[mask], by[mask]) if mask.any() else torch.tensor(0.0, device=device)
            loss.backward()
            optimizer.step()

        model.eval()
        total_val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for vgraph, vdesc, vy in val_loader:
                vgraph = vgraph.to(device)
                vdesc = vdesc.to(device)
                vy_dev = vy.to(device)
                if vy_dev.ndim == 1:
                    vy_dev = vy_dev.unsqueeze(1)
                vout = model(vgraph, vdesc)
                vmask = ~torch.isnan(vy_dev)
                if vmask.any():
                    vloss = loss_fn(vout[vmask], vy_dev[vmask]).item()
                    total_val_loss += vloss * int(vmask.sum().item())
                    n_val += int(vmask.sum().item())

        cur_val_loss = total_val_loss / max(n_val, 1)
        if cur_val_loss < best_val_loss:
            best_val_loss = cur_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    test_preds, test_trues = [], []
    with torch.no_grad():
        for tgraph, tdesc, ty in test_loader:
            tgraph = tgraph.to(device)
            tdesc = tdesc.to(device)
            tout = model(tgraph, tdesc)
            test_preds.append(tout.cpu().numpy())
            test_trues.append(ty.numpy())

    yp = np.concatenate(test_preds, axis=0).ravel()
    yt = np.concatenate(test_trues, axis=0).ravel()
    tmask = ~np.isnan(yt)

    probs = 1.0 / (1.0 + np.exp(-np.clip(yp, -20, 20)))
    try:
        roc = float(roc_auc_score(yt[tmask], probs[tmask]))
    except Exception:
        roc = 0.5
    try:
        pr = float(average_precision_score(yt[tmask], probs[tmask]))
    except Exception:
        pr = 0.0
    return {"roc_auc": roc, "auprc": pr}

def create_sensitivity_dataset_and_splits(seeds: list[int]) -> tuple[pd.DataFrame, list[dict[str, list[int]]]]:
    df_raw = pd.read_csv(DATA_DIR / "bbbp_clean.csv")
    inchikeys = []
    for s in df_raw["smiles"]:
        m = Chem.MolFromSmiles(str(s))
        inchikeys.append(Chem.MolToInchiKey(m) if m else None)
    df_raw["inchikey"] = inchikeys

    conflicting_iks = set()
    for ik, g in df_raw.groupby("inchikey"):
        if len(g) > 1 and len(g["p_np"].unique()) > 1:
            conflicting_iks.add(ik)

    df_clean = df_raw[~df_raw["inchikey"].isin(conflicting_iks)].copy()
    df_dedup = df_clean.drop_duplicates(subset=["inchikey"], keep="first").reset_index(drop=True)
    csv_out = DATA_DIR / "bbbp_sensitivity_clean.csv"
    df_dedup.to_csv(csv_out, index=False)

    smiles = df_dedup["smiles"].tolist()
    groups: dict[str, list[int]] = {}
    for idx, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            key = scaffold or f"acyclic_{Chem.MolToSmiles(mol)}"
        except Exception:
            key = f"raw_{smi}"
        groups.setdefault(key, []).append(idx)

    group_values = list(groups.values())
    n_total = len(df_dedup)
    fractions = (0.8, 0.1, 0.1)
    splits = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        shuffled = [g for g in group_values]
        rng.shuffle(shuffled)
        shuffled.sort(key=len, reverse=True)

        targets = np.asarray(fractions) * n_total
        bins: list[list[int]] = [[], [], []]
        for group in shuffled:
            ratios = [len(bins[i]) / max(targets[i], 1.0) for i in range(3)]
            sel = int(np.argmin(ratios))
            bins[sel].extend(group)

        sp = {"train": bins[0], "val": bins[1], "test": bins[2]}
        sp_path = SPLITS_DIR / f"bbbp_sensitivity_seed_{seed}.json"
        with open(sp_path, "w") as f:
            json.dump(sp, f, indent=2)
        splits.append(sp)

    return df_dedup, splits

def main():
    print("=" * 70)
    print("🔬 BBBP Duplicate-Controlled Sensitivity Study (P0 Benchmark)")
    print("=" * 70)

    parser = argparse.ArgumentParser(description="BBBP duplicate-controlled sensitivity study")
    add_seed_argument(parser)
    args = parser.parse_args()
    seed_list = resolve_seeds(args.seeds)

    df_dedup, splits = create_sensitivity_dataset_and_splits(seed_list)
    print(f"Sensitivity dataset size: {len(df_dedup)} molecules")
    print(f"Class distribution: {df_dedup['p_np'].value_counts().to_dict()}")

    smiles = df_dedup["smiles"].tolist()
    targets = df_dedup["p_np"].values.astype(float)

    print("Pre-parsing molecular graphs...")
    graphs = [mol_to_graph(s) for s in smiles]
    valid_flags = [g is not None for g in graphs]

    print("Extracting 210 RDKit 2D descriptors...")
    X_raw, feat_names = extract_rdkit_2d_descriptors(smiles)
    X_clean = np.nan_to_num(X_raw, nan=0.0, posinf=1e4, neginf=-1e4)

    results = {
        "framework_graph_only": [],
        "framework_hybrid": [],
        "rdkit_rf": [],
    }

    batch_size = 64

    for split_idx, sp in enumerate(splits):
        seed = seed_list[split_idx]
        print(f"\n--- Split {seed}/{len(seed_list)} ---")
        train_idx = sp["train"]
        val_idx = sp["val"]
        test_idx = sp["test"]

        scaler = StandardScaler()
        scaler.fit(X_clean[train_idx])
        X_all = scaler.transform(X_clean)
        X_all = np.nan_to_num(X_all, nan=0.0, posinf=3.0, neginf=-3.0)

        X_tr, y_tr = X_all[train_idx], targets[train_idx]
        X_val, y_val = X_all[val_idx], targets[val_idx]
        X_te, y_te = X_all[test_idx], targets[test_idx]

        rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        rf_probs = rf.predict_proba(X_te)[:, 1]
        rf_auc = float(roc_auc_score(y_te, rf_probs))
        rf_ap = float(average_precision_score(y_te, rf_probs))
        results["rdkit_rf"].append({"roc_auc": rf_auc, "auprc": rf_ap})
        print(f"  ✓ RDKit 2D + RF: AUROC={rf_auc:.4f}")

        graphs_tr = [graphs[i] for i in train_idx]
        graphs_val = [graphs[i] for i in val_idx]
        graphs_te = [graphs[i] for i in test_idx]

        tr_ds = MoleculeGraphDataset(graphs_tr, X_tr, y_tr)
        val_ds = MoleculeGraphDataset(graphs_val, X_val, y_val)
        te_ds = MoleculeGraphDataset(graphs_te, X_te, y_te)

        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_graph_batch)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_graph_batch)
        te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_graph_batch)

        m_graph = DMPNNGraphOnly(hidden_dim=128, depth=3)
        res_graph = train_and_eval(m_graph, tr_loader, val_loader, te_loader, seed=seed)
        results["framework_graph_only"].append(res_graph)
        print(f"  ✓ Framework Graph Only: AUROC={res_graph['roc_auc']:.4f}")

        m_hyb = DMPNNHybrid(desc_dim=X_tr.shape[1], hidden_dim=128, depth=3)
        res_hyb = train_and_eval(m_hyb, tr_loader, val_loader, te_loader, seed=seed)
        results["framework_hybrid"].append(res_hyb)
        print(f"  ✓ Framework Hybrid (Graph+210): AUROC={res_hyb['roc_auc']:.4f}")

    summary = {}
    for m, recs in results.items():
        vals = [r["roc_auc"] for r in recs]
        summary[m] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "per_split": vals,
        }

    out_file = RESULTS_DIR / "bbbp_sensitivity_study.json"
    with open(out_file, "w") as f:
        json.dump(
            {
                "dataset": "bbbp_sensitivity",
                "primary_metric": "roc_auc",
                "seeds": seed_list,
                "summary": summary,
                "raw_results": results,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("📊 BBBP Sensitivity Benchmark Summary (AUROC):")
    for m, s in summary.items():
        print(f"  - {m:<22}: {s['mean']:.4f} ± {s['std']:.4f} (per-split: {[round(v, 4) for v in s['per_split']]})")
    diff = np.array(summary["framework_hybrid"]["per_split"]) - np.array(summary["framework_graph_only"]["per_split"])
    print(f"  ★ Hybrid minus Graph-only Δ: {np.mean(diff):+.4f} ± {np.std(diff):.4f} (wins: {np.sum(diff > 0)}/{len(seed_list)})")
    print(f"Saved results to: {out_file}")

if __name__ == "__main__":
    main()
