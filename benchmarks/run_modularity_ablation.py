"""Benchmark C: Role-Specific Prior/Residual Modularity Ablation.

Addresses Section 3.3 Question:
Can users change feature definitions, predictive roles, and learning components
without rewriting the experiment pipeline?

Evaluates 5 distinct architectural configurations across 5 fixed seeds (80/10/10 split):
1. direct_only: All tabular descriptors passed directly into a 3-layer MLP.
2. prior_only: Ridge/Linear prior fitted strictly on basic physicochemical features (MW, LogP, TPSA, HBD, HBA).
3. residual_only: Neural graph encoder (D-MPNN) trained directly on targets without any prior.
4. direct_hybrid: Graph encoder (D-MPNN) + Tabular descriptors concatenated in standard fusion.
5. prior_residual: Explicit two-branch routing: y_hat = f_prior(X_prior) + f_residual(Graph).

Tested on 1 classification endpoint (BBBP) and 1 regression endpoint (ESOL) using
identical saved 5-seed splits.
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
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chemistry_framework.features.pharmacophore import extract_rdkit_2d_descriptors
from chemistry_framework.data.graph import mol_to_graph, BatchMolGraph
from chemistry_framework.encoders.gnn import DMPNNEncoder
from benchmarks._seeds import DEFAULT_SEEDS, add_seed_argument, resolve_seeds

DATA_DIR = REPO_ROOT / "data" / "moleculenet"
SPLITS_DIR = REPO_ROOT / "benchmarks" / "splits"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"

PHYS_FEATURE_SUBSET = ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds"]

class MoleculeGraphDataset(Dataset):
    def __init__(self, smiles: list[str], descriptors: np.ndarray, targets: np.ndarray):
        self.smiles = smiles
        self.descriptors = torch.tensor(descriptors, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.graphs = [mol_to_graph(s) for s in smiles]

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> tuple[Any, torch.Tensor, torch.Tensor]:
        return self.graphs[idx], self.descriptors[idx], self.targets[idx]

def collate_graph_batch(batch: list[tuple[Any, torch.Tensor, torch.Tensor]]) -> tuple[BatchMolGraph, torch.Tensor, torch.Tensor]:
    graphs, descs, targets = zip(*batch)
    batch_graph = BatchMolGraph(graphs)
    return batch_graph, torch.stack(descs), torch.stack(targets)

class DirectMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, batch_graph: Any, desc: torch.Tensor) -> torch.Tensor:
        return self.net(desc)

class ResidualOnlyGNN(nn.Module):
    def __init__(self, hidden_dim: int = 128, depth: int = 3, out_dim: int = 1):
        super().__init__()
        self.encoder = DMPNNEncoder(hidden_dim=hidden_dim, depth=depth, use_virtual_node=False, aggregation="sum")
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, batch_graph: BatchMolGraph, desc: torch.Tensor) -> torch.Tensor:
        h = self.encoder(batch_graph)
        return self.head(h)

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

    def forward(self, batch_graph: BatchMolGraph, desc: torch.Tensor) -> torch.Tensor:
        hg = self.graph_encoder(batch_graph)
        hd = self.desc_encoder(desc)
        fused = torch.cat([hg, hd], dim=-1)
        return self.head(fused)

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

def train_and_eval_torch_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    is_classification: bool,
    seed: int,
    epochs: int = 40,
    lr: float = 1e-3,
    prior_train: torch.Tensor | None = None,
    prior_val: torch.Tensor | None = None,
    prior_test: torch.Tensor | None = None,
) -> dict[str, float]:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss() if is_classification else nn.HuberLoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        for batch_graph, bdesc, by in train_loader:
            batch_graph = batch_graph.to(device)
            bdesc, by = bdesc.to(device), by.to(device)
            if by.ndim == 1:
                by = by.unsqueeze(1)
            optimizer.zero_grad()

            if prior_train is not None:

                p_b = bdesc[:, :1]
                out = model(batch_graph, bdesc, p_b)
            else:
                out = model(batch_graph, bdesc)

            mask = ~torch.isnan(by)
            loss = loss_fn(out[mask], by[mask]) if mask.any() else torch.tensor(0.0, device=device)
            loss.backward()
            optimizer.step()

        model.eval()
        vlosses = []
        with torch.no_grad():
            for vgraph, vdesc, vy in val_loader:
                vgraph = vgraph.to(device)
                vdesc, vy = vdesc.to(device), vy.to(device)
                if vy.ndim == 1:
                    vy = vy.unsqueeze(1)
                if prior_val is not None:
                    p_v = vdesc[:, :1]
                    vout = model(vgraph, vdesc, p_v)
                else:
                    vout = model(vgraph, vdesc)
                vmask = ~torch.isnan(vy)
                if vmask.any():
                    vl = float(loss_fn(vout[vmask], vy[vmask]))
                    vlosses.append(vl)

            cur_vloss = float(np.mean(vlosses)) if vlosses else float("inf")
            if cur_vloss < best_val_loss:
                best_val_loss = cur_vloss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    test_preds, test_trues = [], []
    with torch.no_grad():
        for tgraph, tdesc, ty in test_loader:
            tgraph = tgraph.to(device)
            tdesc = tdesc.to(device)
            if prior_test is not None:
                p_t = tdesc[:, :1]
                tout = model(tgraph, tdesc, p_t)
            else:
                tout = model(tgraph, tdesc)

            test_preds.append(tout.cpu().numpy())
            test_trues.append(ty.numpy())

    y_p = np.concatenate(test_preds, axis=0)
    y_t = np.concatenate(test_trues, axis=0)

    if is_classification:
        probs = 1.0 / (1.0 + np.exp(-np.clip(y_p, -20, 20)))
        mask = ~np.isnan(y_t.ravel())
        try:
            roc = float(roc_auc_score(y_t.ravel()[mask], probs.ravel()[mask]))
        except Exception:
            roc = 0.5
        try:
            pr = float(average_precision_score(y_t.ravel()[mask], probs.ravel()[mask]))
        except Exception:
            pr = 0.0
        return {"roc_auc": roc, "auprc": pr}
    else:
        yt = y_t.ravel()
        yp = y_p.ravel()
        mask = ~np.isnan(yt)
        rmse = float(math.sqrt(mean_squared_error(yt[mask], yp[mask])))
        mae = float(mean_absolute_error(yt[mask], yp[mask]))
        r2 = float(r2_score(yt[mask], yp[mask])) if len(yt[mask]) > 1 else 0.0
        return {"rmse": rmse, "mae": mae, "r2": r2}

def run_modularity_ablation_on_dataset(dataset_name: str, seeds: list[int] | None = None) -> dict[str, Any]:
    print(f"\n==================================================================")
    print(f"🔬 Prior/Residual Modularity Ablation: {dataset_name.upper()}")
    print(f"==================================================================")

    manifest_file = DATA_DIR / "dataset_manifest.json"
    with open(manifest_file, "r") as f:
        meta = json.load(f)[dataset_name]

    df_clean = pd.read_csv(REPO_ROOT / meta["clean_csv"])
    smiles = df_clean[meta["smiles_col"]].tolist()
    target_col = meta["targets"][0]
    targets = df_clean[target_col].values.astype(np.float32)
    is_classification = "classification" in meta["task_type"]

    valid_mask = [mol_to_graph(s) is not None and not np.isnan(t) for s, t in zip(smiles, targets)]
    smiles = [s for s, v in zip(smiles, valid_mask) if v]
    targets = targets[valid_mask]

    print("Extracting full 2D descriptors and physicochemical prior features...")
    X_raw, feature_names = extract_rdkit_2d_descriptors(smiles)

    phys_indices = [
        i for i, name in enumerate(feature_names)
        if any(name.endswith(f".{p}") or name == p for p in PHYS_FEATURE_SUBSET)
    ]
    print(f"Physicochemical prior features ({len(phys_indices)}): {[feature_names[i] for i in phys_indices]}")

    seed_list = resolve_seeds(seeds)
    configs = ["direct_only", "prior_only", "residual_only", "direct_hybrid", "prior_residual"]
    results: dict[str, list[dict[str, float]]] = {c: [] for c in configs}

    for seed in seed_list:
        print(f"\n--- Running Seed {seed} ---")
        split_file = REPO_ROOT / meta["splits"][f"seed_{seed}"]["split_file"]
        with open(split_file, "r") as f:
            split_dict = json.load(f)

        valid_indices = [i for i, v in enumerate(valid_mask) if v]
        orig_to_clean = {orig_i: clean_i for clean_i, orig_i in enumerate(valid_indices)}
        train_idx = [orig_to_clean[i] for i in split_dict["train"] if i in orig_to_clean]
        val_idx = [orig_to_clean[i] for i in split_dict["val"] if i in orig_to_clean]
        test_idx = [orig_to_clean[i] for i in split_dict["test"] if i in orig_to_clean]

        scaler = StandardScaler()
        X_clean = np.nan_to_num(X_raw, nan=0.0, posinf=1e4, neginf=-1e4)
        scaler.fit(X_clean[train_idx])
        X_all = scaler.transform(X_clean)
        X_all = np.nan_to_num(X_all, nan=0.0, posinf=3.0, neginf=-3.0)
        X_phys = X_all[:, phys_indices]

        if is_classification:
            prior_clf = LogisticRegression(C=1.0, random_state=seed, max_iter=500)
            prior_clf.fit(X_phys[train_idx], targets[train_idx])
            prior_probs_te = prior_clf.predict_proba(X_phys[test_idx])[:, 1]
            try:
                p_roc = float(roc_auc_score(targets[test_idx], prior_probs_te))
            except Exception:
                p_roc = 0.5
            try:
                p_pr = float(average_precision_score(targets[test_idx], prior_probs_te))
            except Exception:
                p_pr = 0.0
            prior_metrics = {"roc_auc": p_roc, "auprc": p_pr}
            prior_pred_train = prior_clf.decision_function(X_phys[train_idx])[:, None]
            prior_pred_val = prior_clf.decision_function(X_phys[val_idx])[:, None]
            prior_pred_test = prior_clf.decision_function(X_phys[test_idx])[:, None]
        else:
            prior_reg = Ridge(alpha=1.0, random_state=seed)
            prior_reg.fit(X_phys[train_idx], targets[train_idx])
            prior_pred_te = prior_reg.predict(X_phys[test_idx])
            rmse = float(math.sqrt(mean_squared_error(targets[test_idx], prior_pred_te)))
            mae = float(mean_absolute_error(targets[test_idx], prior_pred_te))
            r2 = float(r2_score(targets[test_idx], prior_pred_te)) if len(targets[test_idx]) > 1 else 0.0
            prior_metrics = {"rmse": rmse, "mae": mae, "r2": r2}
            prior_pred_train = prior_reg.predict(X_phys[train_idx])[:, None]
            prior_pred_val = prior_reg.predict(X_phys[val_idx])[:, None]
            prior_pred_test = prior_reg.predict(X_phys[test_idx])[:, None]

        results["prior_only"].append(prior_metrics)
        print(f"  ✓ Prior-Only: {prior_metrics}")

        batch_size = 64
        train_ds = MoleculeGraphDataset([smiles[i] for i in train_idx], X_all[train_idx], targets[train_idx])
        val_ds = MoleculeGraphDataset([smiles[i] for i in val_idx], X_all[val_idx], targets[val_idx])
        test_ds = MoleculeGraphDataset([smiles[i] for i in test_idx], X_all[test_idx], targets[test_idx])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_graph_batch)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_graph_batch)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_graph_batch)

        m_direct = DirectMLP(in_dim=X_all.shape[1], hidden_dim=128)
        dir_metrics = train_and_eval_torch_model(m_direct, train_loader, val_loader, test_loader, is_classification, seed)
        results["direct_only"].append(dir_metrics)
        print(f"  ✓ Direct-Only (MLP): {dir_metrics}")

        m_res_only = ResidualOnlyGNN(hidden_dim=128, depth=3)
        res_metrics = train_and_eval_torch_model(m_res_only, train_loader, val_loader, test_loader, is_classification, seed)
        results["residual_only"].append(res_metrics)
        print(f"  ✓ Residual-Only (D-MPNN): {res_metrics}")

        m_hybrid = DirectHybridGNN(desc_dim=X_all.shape[1], hidden_dim=128, depth=3)
        hyb_metrics = train_and_eval_torch_model(m_hybrid, train_loader, val_loader, test_loader, is_classification, seed)
        results["direct_hybrid"].append(hyb_metrics)
        print(f"  ✓ Direct Hybrid: {hyb_metrics}")

        train_ds_pr = MoleculeGraphDataset([smiles[i] for i in train_idx], prior_pred_train, targets[train_idx])
        val_ds_pr = MoleculeGraphDataset([smiles[i] for i in val_idx], prior_pred_val, targets[val_idx])
        test_ds_pr = MoleculeGraphDataset([smiles[i] for i in test_idx], prior_pred_test, targets[test_idx])

        tr_loader_pr = DataLoader(train_ds_pr, batch_size=batch_size, shuffle=True, collate_fn=collate_graph_batch)
        val_loader_pr = DataLoader(val_ds_pr, batch_size=batch_size, shuffle=False, collate_fn=collate_graph_batch)
        te_loader_pr = DataLoader(test_ds_pr, batch_size=batch_size, shuffle=False, collate_fn=collate_graph_batch)

        m_pr = PriorResidualGNN(hidden_dim=128, depth=3)
        pr_metrics = train_and_eval_torch_model(
            m_pr, tr_loader_pr, val_loader_pr, te_loader_pr, is_classification, seed,
            prior_train=torch.tensor(prior_pred_train, dtype=torch.float32),
            prior_val=torch.tensor(prior_pred_val, dtype=torch.float32),
            prior_test=torch.tensor(prior_pred_test, dtype=torch.float32),
        )
        results["prior_residual"].append(pr_metrics)
        print(f"  ✓ Prior + Residual: {pr_metrics}")

    primary_metric = "roc_auc" if is_classification else "rmse"
    print(f"\n📈 Final Modularity Summary Across {len(seed_list)} Splits ({primary_metric.upper()}):")
    summary = {}
    for c in configs:
        vals = [r[primary_metric] for r in results[c]]
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals))
        summary[c] = {"mean": mean_v, "std": std_v}
        print(f"  - {c:<20}: {mean_v:.4f} ± {std_v:.4f}")

    out_payload = {
        "dataset": dataset_name,
        "is_classification": is_classification,
        "primary_metric": primary_metric,
        "seeds": seed_list,
        "raw_results": results,
        "summary": summary,
    }
    out_file = RESULTS_DIR / f"modularity_ablation_{dataset_name}.json"
    with open(out_file, "w") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\n✅ Modularity ablation saved to: {out_file}")
    return out_payload

def main():
    parser = argparse.ArgumentParser(description="Run Prior/Residual Modularity Ablation")
    parser.add_argument("--dataset", type=str, default="esol", help="Dataset name: bbbp or esol")
    add_seed_argument(parser)
    args = parser.parse_args()
    run_modularity_ablation_on_dataset(args.dataset, seeds=args.seeds)

if __name__ == "__main__":
    main()
