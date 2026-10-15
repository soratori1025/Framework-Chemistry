"""Experiment E1: Evidence-Guided Descriptor Routing (Journal of Cheminformatics).

Evaluates whether replacing all 210 descriptors in DirectHybridGNN with evidence-recommended
top-K features (K in {10, 20, 50}, K* chosen strictly by validation metric on each split):
1. Fixes degradation on representation-sensitive tasks (Lipophilicity, BBBP Split 5)
2. Retains performance on tasks where descriptors aid learning (ESOL, FreeSolv, BACE)

Configurations evaluated across 5 datasets x 5 splits:
- direct_hybrid: All 210 descriptors (from modularity ablation)
- hybrid_rec: Top-K recommended by Evidence Scorecard (Spearman + greedy redundancy |r| < 0.85), K* chosen on val
- hybrid_spearman: Univariate Spearman Top-K* (using the same K* selected for hybrid_rec on that split)
- hybrid_mi: Mutual Information Top-K* (using the same K*)
- hybrid_random: Random K* descriptors (3 random draws per split, averaged)
- prior_rec: Explicit Prior + Residual, with prior fitted on Top-K* recommended features
- residual_only: Graph-only D-MPNN reference (from modularity ablation)

Strict validation policy:
All selectors, scalers, and hyperparameter selections (K*) are fitted strictly on train
and selected on validation. Test folds are evaluated only once for final reporting.
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
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
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
from chemistry_framework.feature_analysis import select_evidence_recommendation
from benchmarks.run_feature_evidence_study import select_spearman_top_k, select_mutual_info_top_k

DATA_DIR = REPO_ROOT / "data" / "moleculenet"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

K_CANDIDATES = [10, 20, 50]
RANDOM_DRAWS = 3


# -------------------------------------------------------------
# PyTorch Datasets and Models
# -------------------------------------------------------------

class MoleculeGraphDataset(Dataset):
    def __init__(self, graphs: list[Any], descriptors: np.ndarray, targets: np.ndarray, prior_preds: np.ndarray | None = None):
        self.graphs = graphs
        self.descriptors = torch.tensor(descriptors, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.prior_preds = torch.tensor(prior_preds, dtype=torch.float32) if prior_preds is not None else None

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        p = self.prior_preds[idx] if self.prior_preds is not None else torch.zeros(1)
        return self.graphs[idx], self.descriptors[idx], self.targets[idx], p


def collate_graph_batch(batch: list[tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor | None]]):
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


# -------------------------------------------------------------
# Training with Validation-Metric Selection
# -------------------------------------------------------------

def train_and_eval_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    is_classification: bool,
    seed: int,
    epochs: int = 40,
    lr: float = 1e-3,
) -> tuple[float, dict[str, float]]:
    """Trains model with early checkpoint selection on validation metric.
    
    Returns:
        (best_val_metric, test_metrics_dict)
    """
    torch.manual_seed(seed)
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

        # Validation evaluation
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
                cur_val_score = float(roc_auc_score(vt[vmask], probs[vmask]))
            except Exception:
                cur_val_score = 0.5
            is_better = cur_val_score > best_val_score
        else:
            cur_val_score = float(math.sqrt(mean_squared_error(vt[vmask], vp[vmask])))
            is_better = cur_val_score < best_val_score

        if is_better:
            best_val_score = cur_val_score
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
            roc = float(roc_auc_score(yt[tmask], probs[tmask]))
        except Exception:
            roc = 0.5
        try:
            pr = float(average_precision_score(yt[tmask], probs[tmask]))
        except Exception:
            pr = 0.0
        test_metrics = {"roc_auc": roc, "auprc": pr}
    else:
        rmse = float(math.sqrt(mean_squared_error(yt[tmask], yp[tmask])))
        mae = float(mean_absolute_error(yt[tmask], yp[tmask]))
        r2 = float(r2_score(yt[tmask], yp[tmask])) if len(yt[tmask]) > 1 else 0.0
        test_metrics = {"rmse": rmse, "mae": mae, "r2": r2}

    return best_val_score, test_metrics


# -------------------------------------------------------------
# Core Experiment Execution
# -------------------------------------------------------------

def run_evidence_routing_dataset(dataset_name: str) -> dict[str, Any]:
    print(f"\n==================================================================")
    print(f"🚀 Experiment E1: Evidence-Guided Routing on {dataset_name.upper()}")
    print(f"==================================================================")

    manifest_file = DATA_DIR / "dataset_manifest.json"
    with open(manifest_file, "r") as f:
        meta = json.load(f)[dataset_name]

    df_clean = pd.read_csv(REPO_ROOT / meta["clean_csv"])
    smiles_col = meta.get("smiles_col", "smiles")
    if smiles_col not in df_clean.columns:
        smiles_col = "mol" if "mol" in df_clean.columns else "smiles"
    smiles = df_clean[smiles_col].tolist()
    target_col = meta["targets"][0]
    targets = df_clean[target_col].values.astype(np.float32)
    is_classification = "classification" in meta["task_type"]
    primary_metric = "roc_auc" if is_classification else "rmse"

    # Pre-parse graphs and filter valid
    print("Pre-parsing molecular graphs...")
    valid_flags = []
    parsed_graphs = []
    for s, t in zip(smiles, targets):
        if np.isnan(t):
            valid_flags.append(False)
            parsed_graphs.append(None)
            continue
        g = mol_to_graph(s)
        if g is None:
            valid_flags.append(False)
            parsed_graphs.append(None)
        else:
            valid_flags.append(True)
            parsed_graphs.append(g)

    smiles = [s for s, v in zip(smiles, valid_flags) if v]
    targets = targets[valid_flags]
    graphs = [g for g, v in zip(parsed_graphs, valid_flags) if v]

    print("Extracting 2D RDKit descriptors...")
    X_raw, feature_names = extract_rdkit_2d_descriptors(smiles)

    seeds = [1, 2, 3, 4, 5]
    per_split_results: list[dict[str, Any]] = []

    # Load baseline direct_hybrid and residual_only from modularity ablation
    ablation_file = RESULTS_DIR / f"modularity_ablation_{dataset_name}.json"
    cached_ablation = {}
    if ablation_file.exists():
        with open(ablation_file, "r") as f:
            cached_ablation = json.load(f)["raw_results"]

    batch_size = 64

    for split_idx, seed in enumerate(seeds):
        print(f"\n--- Split {split_idx + 1}/5 (Seed {seed}) ---")
        split_file = REPO_ROOT / meta["splits"][f"seed_{seed}"]["split_file"]
        with open(split_file, "r") as f:
            split_dict = json.load(f)

        valid_indices = [i for i, v in enumerate(valid_flags) if v]
        orig_to_clean = {orig_i: clean_i for clean_i, orig_i in enumerate(valid_indices)}
        train_idx = [orig_to_clean[i] for i in split_dict["train"] if i in orig_to_clean]
        val_idx = [orig_to_clean[i] for i in split_dict["val"] if i in orig_to_clean]
        test_idx = [orig_to_clean[i] for i in split_dict["test"] if i in orig_to_clean]

        # Scaler fit strictly on train
        scaler = StandardScaler()
        X_clean = np.nan_to_num(X_raw, nan=0.0, posinf=1e4, neginf=-1e4)
        scaler.fit(X_clean[train_idx])
        X_all = scaler.transform(X_clean)
        X_all = np.nan_to_num(X_all, nan=0.0, posinf=3.0, neginf=-3.0)

        X_tr = X_all[train_idx]
        y_tr = targets[train_idx]
        X_val = X_all[val_idx]
        y_val = targets[val_idx]
        X_te = X_all[test_idx]
        y_te = targets[test_idx]

        graphs_tr = [graphs[i] for i in train_idx]
        graphs_val = [graphs[i] for i in val_idx]
        graphs_te = [graphs[i] for i in test_idx]

        # -------------------------------------------------------------
        # Step A: hybrid_rec selection (K* chosen on validation)
        # -------------------------------------------------------------
        best_val_score_rec = -float("inf") if is_classification else float("inf")
        best_k_rec = K_CANDIDATES[0]
        best_rec_test_metrics = None
        best_rec_indices = None

        print(f"  Selecting K* in {K_CANDIDATES} on validation set...")
        for k_cand in K_CANDIDATES:
            indices_cand = select_evidence_recommendation(
                X_train=X_tr,
                y_train=y_tr,
                feature_names=feature_names,
                k=k_cand,
                redundancy_threshold=0.85,
            )
            tr_ds = MoleculeGraphDataset(graphs_tr, X_tr[:, indices_cand], y_tr)
            val_ds = MoleculeGraphDataset(graphs_val, X_val[:, indices_cand], y_val)
            te_ds = MoleculeGraphDataset(graphs_te, X_te[:, indices_cand], y_te)

            tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_graph_batch)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_graph_batch)
            te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_graph_batch)

            model_cand = DirectHybridGNN(desc_dim=k_cand, hidden_dim=128, depth=3)
            val_s, te_m = train_and_eval_model(
                model_cand, tr_loader, val_loader, te_loader,
                is_classification=is_classification, seed=seed, epochs=40, lr=1e-3,
            )
            print(f"    - K={k_cand:<2}: val {primary_metric}={val_s:.4f}, test {primary_metric}={te_m[primary_metric]:.4f}")

            is_better = (val_s > best_val_score_rec) if is_classification else (val_s < best_val_score_rec)
            if is_better:
                best_val_score_rec = val_s
                best_k_rec = k_cand
                best_rec_test_metrics = te_m
                best_rec_indices = indices_cand

        print(f"  ★ Selected K*={best_k_rec} (val {primary_metric}={best_val_score_rec:.4f}, test {primary_metric}={best_rec_test_metrics[primary_metric]:.4f})")

        # -------------------------------------------------------------
        # Step B: hybrid_spearman with same K*
        # -------------------------------------------------------------
        indices_spearman = select_spearman_top_k(X_tr, y_tr, feature_names, best_k_rec)
        tr_ds_sp = MoleculeGraphDataset(graphs_tr, X_tr[:, indices_spearman], y_tr)
        val_ds_sp = MoleculeGraphDataset(graphs_val, X_val[:, indices_spearman], y_val)
        te_ds_sp = MoleculeGraphDataset(graphs_te, X_te[:, indices_spearman], y_te)
        tr_loader_sp = DataLoader(tr_ds_sp, batch_size=batch_size, shuffle=True, collate_fn=collate_graph_batch)
        val_loader_sp = DataLoader(val_ds_sp, batch_size=batch_size, shuffle=False, collate_fn=collate_graph_batch)
        te_loader_sp = DataLoader(te_ds_sp, batch_size=batch_size, shuffle=False, collate_fn=collate_graph_batch)

        m_sp = DirectHybridGNN(desc_dim=best_k_rec, hidden_dim=128, depth=3)
        _, sp_test_metrics = train_and_eval_model(
            m_sp, tr_loader_sp, val_loader_sp, te_loader_sp,
            is_classification=is_classification, seed=seed, epochs=40, lr=1e-3,
        )
        print(f"  ✓ hybrid_spearman (K*={best_k_rec}): test {primary_metric}={sp_test_metrics[primary_metric]:.4f}")

        # -------------------------------------------------------------
        # Step C: hybrid_mi with same K*
        # -------------------------------------------------------------
        indices_mi = select_mutual_info_top_k(X_tr, y_tr, best_k_rec, is_classification=is_classification, seed=seed)
        tr_ds_mi = MoleculeGraphDataset(graphs_tr, X_tr[:, indices_mi], y_tr)
        val_ds_mi = MoleculeGraphDataset(graphs_val, X_val[:, indices_mi], y_val)
        te_ds_mi = MoleculeGraphDataset(graphs_te, X_te[:, indices_mi], y_te)
        tr_loader_mi = DataLoader(tr_ds_mi, batch_size=batch_size, shuffle=True, collate_fn=collate_graph_batch)
        val_loader_mi = DataLoader(val_ds_mi, batch_size=batch_size, shuffle=False, collate_fn=collate_graph_batch)
        te_loader_mi = DataLoader(te_ds_mi, batch_size=batch_size, shuffle=False, collate_fn=collate_graph_batch)

        m_mi = DirectHybridGNN(desc_dim=best_k_rec, hidden_dim=128, depth=3)
        _, mi_test_metrics = train_and_eval_model(
            m_mi, tr_loader_mi, val_loader_mi, te_loader_mi,
            is_classification=is_classification, seed=seed, epochs=40, lr=1e-3,
        )
        print(f"  ✓ hybrid_mi (K*={best_k_rec}): test {primary_metric}={mi_test_metrics[primary_metric]:.4f}")

        # -------------------------------------------------------------
        # Step D: hybrid_random (3 random draws, averaged)
        # -------------------------------------------------------------
        rnd_metrics_list = []
        rng = np.random.RandomState(seed * 1000 + 7)
        for draw in range(RANDOM_DRAWS):
            rnd_indices = rng.choice(X_tr.shape[1], size=best_k_rec, replace=False).tolist()
            tr_ds_rnd = MoleculeGraphDataset(graphs_tr, X_tr[:, rnd_indices], y_tr)
            val_ds_rnd = MoleculeGraphDataset(graphs_val, X_val[:, rnd_indices], y_val)
            te_ds_rnd = MoleculeGraphDataset(graphs_te, X_te[:, rnd_indices], y_te)
            tr_loader_rnd = DataLoader(tr_ds_rnd, batch_size=batch_size, shuffle=True, collate_fn=collate_graph_batch)
            val_loader_rnd = DataLoader(val_ds_rnd, batch_size=batch_size, shuffle=False, collate_fn=collate_graph_batch)
            te_loader_rnd = DataLoader(te_ds_rnd, batch_size=batch_size, shuffle=False, collate_fn=collate_graph_batch)

            m_rnd = DirectHybridGNN(desc_dim=best_k_rec, hidden_dim=128, depth=3)
            _, rnd_te = train_and_eval_model(
                m_rnd, tr_loader_rnd, val_loader_rnd, te_loader_rnd,
                is_classification=is_classification, seed=seed + draw * 100, epochs=40, lr=1e-3,
            )
            rnd_metrics_list.append(rnd_te)

        avg_rnd_metrics = {
            m: float(np.mean([d[m] for d in rnd_metrics_list]))
            for m in rnd_metrics_list[0]
        }
        print(f"  ✓ hybrid_random (K*={best_k_rec}, 3 draws): test {primary_metric}={avg_rnd_metrics[primary_metric]:.4f}")

        # -------------------------------------------------------------
        # Step E: prior_rec (Prior + Residual on recommended top-K*)
        # -------------------------------------------------------------
        X_rec_tr = X_tr[:, best_rec_indices]
        X_rec_val = X_val[:, best_rec_indices]
        X_rec_te = X_te[:, best_rec_indices]

        if is_classification:
            prior_clf = LogisticRegression(C=1.0, random_state=seed, max_iter=500)
            prior_clf.fit(X_rec_tr, y_tr)
            p_train = prior_clf.decision_function(X_rec_tr)[:, None]
            p_val = prior_clf.decision_function(X_rec_val)[:, None]
            p_test = prior_clf.decision_function(X_rec_te)[:, None]
        else:
            prior_reg = Ridge(alpha=1.0, random_state=seed)
            prior_reg.fit(X_rec_tr, y_tr)
            p_train = prior_reg.predict(X_rec_tr)[:, None]
            p_val = prior_reg.predict(X_rec_val)[:, None]
            p_test = prior_reg.predict(X_rec_te)[:, None]

        tr_ds_pr = MoleculeGraphDataset(graphs_tr, X_rec_tr, y_tr, prior_preds=p_train)
        val_ds_pr = MoleculeGraphDataset(graphs_val, X_rec_val, y_val, prior_preds=p_val)
        te_ds_pr = MoleculeGraphDataset(graphs_te, X_rec_te, y_te, prior_preds=p_test)

        tr_loader_pr = DataLoader(tr_ds_pr, batch_size=batch_size, shuffle=True, collate_fn=collate_graph_batch)
        val_loader_pr = DataLoader(val_ds_pr, batch_size=batch_size, shuffle=False, collate_fn=collate_graph_batch)
        te_loader_pr = DataLoader(te_ds_pr, batch_size=batch_size, shuffle=False, collate_fn=collate_graph_batch)

        m_pr = PriorResidualGNN(hidden_dim=128, depth=3)
        _, pr_test_metrics = train_and_eval_model(
            m_pr, tr_loader_pr, val_loader_pr, te_loader_pr,
            is_classification=is_classification, seed=seed, epochs=40, lr=1e-3,
        )
        print(f"  ✓ prior_rec (K*={best_k_rec}): test {primary_metric}={pr_test_metrics[primary_metric]:.4f}")

        # Baseline references from cached modularity ablation
        dir_hyb_metric = cached_ablation.get("direct_hybrid", [{}])[split_idx].get(primary_metric, float("nan"))
        res_only_metric = cached_ablation.get("residual_only", [{}])[split_idx].get(primary_metric, float("nan"))

        split_record = {
            "split": split_idx + 1,
            "seed": seed,
            "best_k": best_k_rec,
            "hybrid_rec": best_rec_test_metrics,
            "hybrid_spearman": sp_test_metrics,
            "hybrid_mi": mi_test_metrics,
            "hybrid_random": avg_rnd_metrics,
            "prior_rec": pr_test_metrics,
            "direct_hybrid": {primary_metric: dir_hyb_metric},
            "residual_only": {primary_metric: res_only_metric},
        }
        per_split_results.append(split_record)

    # -------------------------------------------------------------
    # Aggregated Statistics & Paired Comparisons
    # -------------------------------------------------------------
    models_to_summarize = ["direct_hybrid", "residual_only", "hybrid_rec", "hybrid_spearman", "hybrid_mi", "hybrid_random", "prior_rec"]
    summary: dict[str, Any] = {}

    for m in models_to_summarize:
        vals = [r[m][primary_metric] for r in per_split_results]
        summary[m] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "per_split": vals,
        }

    # Paired comparisons against direct_hybrid and residual_only
    rec_vals = np.array(summary["hybrid_rec"]["per_split"])
    dir_vals = np.array(summary["direct_hybrid"]["per_split"])
    res_vals = np.array(summary["residual_only"]["per_split"])

    def compute_paired(a: np.ndarray, b: np.ndarray, higher_is_better: bool):
        diff = a - b
        delta_mean = float(np.mean(diff))
        delta_std = float(np.std(diff))
        n = len(diff)
        t_crit = stats.t.ppf(0.975, df=n - 1)
        ci_lower = delta_mean - t_crit * (delta_std / math.sqrt(n))
        ci_upper = delta_mean + t_crit * (delta_std / math.sqrt(n))
        wins = int(np.sum(diff > 0 if higher_is_better else diff < 0))
        return {
            "delta_mean": delta_mean,
            "delta_std": delta_std,
            "ci_95": [float(ci_lower), float(ci_upper)],
            "wins": f"{wins}/{n}",
        }

    higher_is_better = is_classification
    paired_vs_direct_hybrid = compute_paired(rec_vals, dir_vals, higher_is_better)
    paired_vs_residual_only = compute_paired(rec_vals, res_vals, higher_is_better)

    k_distribution = [r["best_k"] for r in per_split_results]

    out_payload = {
        "dataset": dataset_name,
        "is_classification": is_classification,
        "primary_metric": primary_metric,
        "k_candidates": K_CANDIDATES,
        "k_distribution": k_distribution,
        "summary": summary,
        "paired_vs_direct_hybrid": paired_vs_direct_hybrid,
        "paired_vs_residual_only": paired_vs_residual_only,
        "per_split_results": per_split_results,
    }

    out_file = RESULTS_DIR / f"evidence_routing_{dataset_name}.json"
    with open(out_file, "w") as f:
        json.dump(out_payload, f, indent=2)

    print(f"\n==================================================================")
    print(f"📊 Summary for {dataset_name.upper()} ({primary_metric.upper()}):")
    print(f"  - direct_hybrid (210 desc): {summary['direct_hybrid']['mean']:.4f} ± {summary['direct_hybrid']['std']:.4f}")
    print(f"  - residual_only (graph):    {summary['residual_only']['mean']:.4f} ± {summary['residual_only']['std']:.4f}")
    print(f"  - hybrid_rec (top-K*):      {summary['hybrid_rec']['mean']:.4f} ± {summary['hybrid_rec']['std']:.4f} (K* across splits: {k_distribution})")
    print(f"  - hybrid_spearman:          {summary['hybrid_spearman']['mean']:.4f} ± {summary['hybrid_spearman']['std']:.4f}")
    print(f"  - hybrid_mi:                {summary['hybrid_mi']['mean']:.4f} ± {summary['hybrid_mi']['std']:.4f}")
    print(f"  - hybrid_random:            {summary['hybrid_random']['mean']:.4f} ± {summary['hybrid_random']['std']:.4f}")
    print(f"  - prior_rec:                {summary['prior_rec']['mean']:.4f} ± {summary['prior_rec']['std']:.4f}")
    print(f"  ★ Paired Δ vs direct_hybrid: {paired_vs_direct_hybrid['delta_mean']:+.4f} (95% CI: [{paired_vs_direct_hybrid['ci_95'][0]:.4f}, {paired_vs_direct_hybrid['ci_95'][1]:.4f}], wins: {paired_vs_direct_hybrid['wins']})")
    print(f"  ★ Paired Δ vs residual_only: {paired_vs_residual_only['delta_mean']:+.4f} (95% CI: [{paired_vs_residual_only['ci_95'][0]:.4f}, {paired_vs_residual_only['ci_95'][1]:.4f}], wins: {paired_vs_residual_only['wins']})")
    print(f"Saved results to: {out_file}")

    return out_payload


def main():
    parser = argparse.ArgumentParser(description="Run Experiment E1: Evidence-Guided Descriptor Routing.")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "esol", "freesolv", "bbbp", "lipophilicity", "bace"])
    args = parser.parse_args()

    datasets = ["esol", "freesolv", "bbbp", "lipophilicity", "bace"] if args.dataset == "all" else [args.dataset]
    for d in datasets:
        run_evidence_routing_dataset(d)


if __name__ == "__main__":
    main()
