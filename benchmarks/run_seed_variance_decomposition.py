"""Seed Variance vs Split Variance Decomposition Study (E3).

Evaluates the variance introduced by neural network weight initialization / mini-batch shuffling
under a fixed split (Split 1) across 3 seeds (e.g. 42, 123, 999) under deterministic flags:
- torch.use_deterministic_algorithms(True)
- CUBLAS_WORKSPACE_CONFIG=:4096:8

Compares SD_seed against SD_split from the 5-fold cross-validation.
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
import torch.nn as nn
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rdkit import Chem
from chemistry_framework.data.graph import BatchMolGraph, mol_to_graph
from chemistry_framework.encoders.gnn import DMPNNEncoder
from chemistry_framework.features.pharmacophore import extract_rdkit_2d_descriptors
from chemistry_framework.feature_analysis import select_evidence_recommendation

DATA_DIR = REPO_ROOT / "data" / "moleculenet"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

class MoleculeGraphDataset(Dataset):
    def __init__(self, graphs: list[Any], descriptors: np.ndarray, targets: np.ndarray, prior_preds: np.ndarray | None = None):
        self.graphs = graphs
        self.descriptors = torch.tensor(descriptors, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.prior_preds = torch.tensor(prior_preds, dtype=torch.float32) if prior_preds is not None else None

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]:
        p = self.prior_preds[idx] if self.prior_preds is not None else torch.zeros(1)
        return self.graphs[idx], self.descriptors[idx], self.targets[idx], p

def collate_graph_batch(batch: list[tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]]):
    graphs, descs, targets, priors = zip(*batch)
    batch_graph = BatchMolGraph(graphs)
    return batch_graph, torch.stack(descs), torch.stack(targets), torch.stack(priors)

class DirectHybridGNN(nn.Module):
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

    def forward(self, batch_graph: BatchMolGraph, desc: torch.Tensor, prior_pred: torch.Tensor | None = None) -> torch.Tensor:
        hg = self.graph_encoder(batch_graph)
        hd = self.desc_encoder(desc)
        fused = torch.cat([hg, hd], dim=-1)
        return self.head(fused)

class ResidualOnlyGNN(nn.Module):
    def __init__(self, hidden_dim: int = 128, depth: int = 3, out_dim: int = 1):
        super().__init__()
        self.graph_encoder = DMPNNEncoder(hidden_dim=hidden_dim, depth=depth, use_virtual_node=False, aggregation="sum")
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, batch_graph: BatchMolGraph, desc: torch.Tensor, prior_pred: torch.Tensor | None = None) -> torch.Tensor:
        h = self.graph_encoder(batch_graph)
        return self.head(h)

class PriorResidualGNN(nn.Module):
    def __init__(self, hidden_dim: int = 128, depth: int = 3, out_dim: int = 1):
        super().__init__()
        self.graph_encoder = DMPNNEncoder(hidden_dim=hidden_dim, depth=depth, use_virtual_node=False, aggregation="sum")
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, batch_graph: BatchMolGraph, desc: torch.Tensor, prior_pred: torch.Tensor) -> torch.Tensor:
        h = self.graph_encoder(batch_graph)
        res = self.residual_head(h)
        return prior_pred + res

def train_and_eval(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    is_classification: bool,
    seed: int,
    epochs: int = 40,
    lr: float = 1e-3,
) -> float:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss() if is_classification else nn.HuberLoss()

    best_val_score = -float("inf") if is_classification else float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        for batch_graph, bdesc, by, bprior in train_loader:
            batch_graph = batch_graph.to(device)
            bdesc, by, bprior = bdesc.to(device), by.to(device), bprior.to(device)
            if by.ndim == 1:
                by = by.unsqueeze(1)
            optimizer.zero_grad()
            out = model(batch_graph, bdesc, bprior)
            mask = ~torch.isnan(by)
            loss = loss_fn(out[mask], by[mask]) if mask.any() else torch.tensor(0.0, device=device)
            loss.backward()
            optimizer.step()

        model.eval()
        vpreds, vtrues = [], []
        with torch.no_grad():
            for vgraph, vdesc, vy, vprior in val_loader:
                vgraph = vgraph.to(device)
                vdesc, vprior = vdesc.to(device), vprior.to(device)
                vout = model(vgraph, vdesc, vprior)
                vpreds.append(vout.cpu().numpy())
                vtrues.append(vy.numpy())

        vp = np.concatenate(vpreds, axis=0).ravel()
        vt = np.concatenate(vtrues, axis=0).ravel()
        vmask = ~np.isnan(vt)

        if is_classification:
            probs = 1.0 / (1.0 + np.exp(-np.clip(vp, -20, 20)))
            try:
                cur_val = float(roc_auc_score(vt[vmask], probs[vmask]))
            except Exception:
                cur_val = 0.5
            is_better = cur_val > best_val_score
        else:
            cur_val = float(math.sqrt(mean_squared_error(vt[vmask], vp[vmask])))
            is_better = cur_val < best_val_score

        if is_better:
            best_val_score = cur_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    test_preds, test_trues = [], []
    with torch.no_grad():
        for tgraph, tdesc, ty, tprior in test_loader:
            tgraph = tgraph.to(device)
            tdesc, tprior = tdesc.to(device), tprior.to(device)
            tout = model(tgraph, tdesc, tprior)
            test_preds.append(tout.cpu().numpy())
            test_trues.append(ty.numpy())

    yp = np.concatenate(test_preds, axis=0).ravel()
    yt = np.concatenate(test_trues, axis=0).ravel()
    tmask = ~np.isnan(yt)

    if is_classification:
        probs = 1.0 / (1.0 + np.exp(-np.clip(yp, -20, 20)))
        try:
            return float(roc_auc_score(yt[tmask], probs[tmask]))
        except Exception:
            return 0.5
    else:
        return float(math.sqrt(mean_squared_error(yt[tmask], yp[tmask])))

def run_seed_variance_study(dataset_name: str = "bbbp", split_seed: int = 1, test_seeds: list[int] | None = None) -> dict[str, Any]:
    if test_seeds is None:
        test_seeds = [42, 123, 999]

    print(f"\n==================================================================")
    print(f"🔬 Seed Variance Decomposition Study: {dataset_name.upper()} (Split Seed {split_seed})")
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

    valid_mask = ~np.isnan(targets)
    if not np.all(valid_mask):
        smiles = [s for s, m in zip(smiles, valid_mask) if m]
        targets = targets[valid_mask]

    print(f"Featurizing {len(smiles)} molecules...")
    X_raw, feature_names = extract_rdkit_2d_descriptors(smiles)
    graphs = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        graphs.append(mol_to_graph(mol) if mol is not None else mol_to_graph(Chem.MolFromSmiles("C")))

    split_file = REPO_ROOT / meta["splits"][f"seed_{split_seed}"]["split_file"]
    with open(split_file, "r") as f:
        split_dict = json.load(f)

    valid_indices = [i for i, v in enumerate(valid_mask) if v]
    orig_to_clean = {orig_i: clean_i for clean_i, orig_i in enumerate(valid_indices)}
    train_idx = [orig_to_clean[i] for i in split_dict["train"] if i in orig_to_clean]
    val_idx = [orig_to_clean[i] for i in split_dict["val"] if i in orig_to_clean]
    test_idx = [orig_to_clean[i] for i in split_dict["test"] if i in orig_to_clean]

    scaler = StandardScaler()
    X_clean = np.nan_to_num(X_raw, nan=0.0, posinf=1e4, neginf=-1e4)
    X_tr = scaler.fit_transform(X_clean[train_idx])
    X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=3.0, neginf=-3.0)
    X_val = np.nan_to_num(scaler.transform(X_clean[val_idx]), nan=0.0, posinf=3.0, neginf=-3.0)
    X_te = np.nan_to_num(scaler.transform(X_clean[test_idx]), nan=0.0, posinf=3.0, neginf=-3.0)

    y_tr = targets[train_idx]
    y_val = targets[val_idx]
    y_te = targets[test_idx]

    prior_names = ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds"]
    prior_features = [f"rdkit_2d.{name}" for name in prior_names]
    prior_indices = [i for i, name in enumerate(feature_names) if name in prior_features]
    if len(prior_indices) == 0:
        prior_indices = [i for i, name in enumerate(feature_names) if any(n in name for n in prior_names)]
    X_p_tr = X_tr[:, prior_indices]
    X_p_val = X_val[:, prior_indices]
    X_p_te = X_te[:, prior_indices]

    if is_classification:
        prior_clf = LogisticRegression(penalty="l2", C=1.0, random_state=split_seed)
        prior_clf.fit(X_p_tr, y_tr)
        p_train = prior_clf.decision_function(X_p_tr)[:, None]
        p_val = prior_clf.decision_function(X_p_val)[:, None]
        p_test = prior_clf.decision_function(X_p_te)[:, None]
    else:
        prior_reg = Ridge(alpha=1.0, random_state=split_seed)
        prior_reg.fit(X_p_tr, y_tr)
        p_train = prior_reg.predict(X_p_tr)[:, None]
        p_val = prior_reg.predict(X_p_val)[:, None]
        p_test = prior_reg.predict(X_p_te)[:, None]

    rec_indices = select_evidence_recommendation(X_tr, y_tr, feature_names, k=20, redundancy_threshold=0.85)
    X_rec_tr = X_tr[:, rec_indices]
    X_rec_val = X_val[:, rec_indices]
    X_rec_te = X_te[:, rec_indices]

    graphs_tr = [graphs[i] for i in train_idx]
    graphs_val = [graphs[i] for i in val_idx]
    graphs_te = [graphs[i] for i in test_idx]

    models_to_test = ["residual_only", "direct_hybrid_210", "hybrid_rec_20", "prior_residual"]
    seed_results: dict[str, list[float]] = {m: [] for m in models_to_test}

    batch_size = 64

    for t_seed in test_seeds:
        print(f"\n--- Testing Initialization Seed {t_seed} ---")
        g = torch.Generator()
        g.manual_seed(t_seed)

        ds_tr = MoleculeGraphDataset(graphs_tr, X_tr, y_tr)
        ds_val = MoleculeGraphDataset(graphs_val, X_val, y_val)
        ds_te = MoleculeGraphDataset(graphs_te, X_te, y_te)
        tr_loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, collate_fn=collate_graph_batch, generator=g)
        val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, collate_fn=collate_graph_batch)
        te_loader = DataLoader(ds_te, batch_size=batch_size, shuffle=False, collate_fn=collate_graph_batch)

        m_res = ResidualOnlyGNN(hidden_dim=128, depth=3)
        res_score = train_and_eval(m_res, tr_loader, val_loader, te_loader, is_classification, seed=t_seed)
        seed_results["residual_only"].append(res_score)
        print(f"  residual_only: {res_score:.4f}")

        m_dir = DirectHybridGNN(desc_dim=X_tr.shape[1], hidden_dim=128, depth=3)
        dir_score = train_and_eval(m_dir, tr_loader, val_loader, te_loader, is_classification, seed=t_seed)
        seed_results["direct_hybrid_210"].append(dir_score)
        print(f"  direct_hybrid_210: {dir_score:.4f}")

        ds_rec_tr = MoleculeGraphDataset(graphs_tr, X_rec_tr, y_tr)
        ds_rec_val = MoleculeGraphDataset(graphs_val, X_rec_val, y_val)
        ds_rec_te = MoleculeGraphDataset(graphs_te, X_rec_te, y_te)
        tr_rec_loader = DataLoader(ds_rec_tr, batch_size=batch_size, shuffle=True, collate_fn=collate_graph_batch, generator=g)
        val_rec_loader = DataLoader(ds_rec_val, batch_size=batch_size, shuffle=False, collate_fn=collate_graph_batch)
        te_rec_loader = DataLoader(ds_rec_te, batch_size=batch_size, shuffle=False, collate_fn=collate_graph_batch)

        m_rec = DirectHybridGNN(desc_dim=len(rec_indices), hidden_dim=128, depth=3)
        rec_score = train_and_eval(m_rec, tr_rec_loader, val_rec_loader, te_rec_loader, is_classification, seed=t_seed)
        seed_results["hybrid_rec_20"].append(rec_score)
        print(f"  hybrid_rec_20: {rec_score:.4f}")

        ds_pr_tr = MoleculeGraphDataset(graphs_tr, X_tr, y_tr, prior_preds=p_train)
        ds_pr_val = MoleculeGraphDataset(graphs_val, X_val, y_val, prior_preds=p_val)
        ds_pr_te = MoleculeGraphDataset(graphs_te, X_te, y_te, prior_preds=p_test)
        tr_pr_loader = DataLoader(ds_pr_tr, batch_size=batch_size, shuffle=True, collate_fn=collate_graph_batch, generator=g)
        val_pr_loader = DataLoader(ds_pr_val, batch_size=batch_size, shuffle=False, collate_fn=collate_graph_batch)
        te_pr_loader = DataLoader(ds_pr_te, batch_size=batch_size, shuffle=False, collate_fn=collate_graph_batch)

        m_pr = PriorResidualGNN(hidden_dim=128, depth=3)
        pr_score = train_and_eval(m_pr, tr_pr_loader, val_pr_loader, te_pr_loader, is_classification, seed=t_seed)
        seed_results["prior_residual"].append(pr_score)
        print(f"  prior_residual: {pr_score:.4f}")

    summary = {}
    print(f"\n==================================================================")
    print(f"🏆 SEED VARIANCE DECOMPOSITION ({dataset_name.upper()}, Split Seed {split_seed})")
    print(f"==================================================================")
    for m in models_to_test:
        vals = seed_results[m]
        mean = float(np.mean(vals))
        sd = float(np.std(vals))
        summary[m] = {
            "mean": mean,
            "std_seed": sd,
            "seeds": test_seeds,
            "scores": vals,
        }
        print(f"{m:<20}: {mean:.4f} ± {sd:.4f}  (scores: {[round(v, 4) for v in vals]})")

    out_file = RESULTS_DIR / f"seed_variance_{dataset_name}_split_{split_seed}.json"
    with open(out_file, "w") as f:
        json.dump({"dataset": dataset_name, "split_seed": split_seed, "summary": summary}, f, indent=2)
    print(f"Saved to {out_file}")
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="bbbp")
    parser.add_argument("--split-seed", type=int, default=1)
    args = parser.parse_args()

    run_seed_variance_study(args.dataset, split_seed=args.split_seed)

if __name__ == "__main__":
    main()
