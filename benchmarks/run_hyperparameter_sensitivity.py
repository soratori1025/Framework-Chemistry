"""Hyper-parameter sensitivity of the descriptor effect (reviewer point M7).

The benchmark fixes its hyper-parameters a priori, so a critic can ask whether
the reported descriptor effect is an artefact of one setting.  This study
repeats the two configurations that define the effect -- graph only and the
direct graph-plus-descriptor hybrid -- across three hidden widths on the same
saved splits.  By default it runs FreeSolv, the endpoint with the largest
descriptor effect; BACE can be added on the command line.  Everything else is
held fixed and identical to the routing experiment, and the graph-only arm is
the same encoder-plus-shallow-head model (a zero prior in PriorResidualGNN).

    python benchmarks/run_hyperparameter_sensitivity.py
    python benchmarks/run_hyperparameter_sensitivity.py --dataset freesolv --splits 1 2 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.run_evidence_routing_experiment import (  # noqa: E402
    DirectHybridGNN,
    MoleculeGraphDataset,
    PriorResidualGNN,
    collate_graph_batch,
    train_and_eval_model,
)
from chemistry_framework.data.graph import mol_to_graph  # noqa: E402
from chemistry_framework.features.pharmacophore import extract_rdkit_2d_descriptors  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "moleculenet"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"

HIDDEN_DIMS = [64, 128, 256]


def run_dataset(dataset: str, splits: list[int], hidden_dims: list[int]) -> dict[str, Any]:
    manifest = json.loads((DATA_DIR / "dataset_manifest.json").read_text())[dataset]
    df = pd.read_csv(REPO_ROOT / manifest["clean_csv"])
    smiles = df[manifest["smiles_col"]].astype(str).tolist()
    targets = df[manifest["targets"][0]].values.astype(np.float32)
    is_cls = "classification" in manifest["task_type"]
    metric = "roc_auc" if is_cls else "rmse"

    valid = ~np.isnan(targets)
    smiles = [s for s, v in zip(smiles, valid) if v]
    targets = targets[valid]
    graphs = [mol_to_graph(s) for s in smiles]
    X_raw, _ = extract_rdkit_2d_descriptors(smiles)
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=1e4, neginf=-1e4)

    rows: list[dict[str, Any]] = []
    for split in splits:
        split_path = REPO_ROOT / manifest["splits"][f"seed_{split}"]["split_file"]
        sp = json.loads(split_path.read_text())
        tr, va, te = sp["train"], sp["val"], sp["test"]

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(X_raw[tr])
        X = np.nan_to_num(scaler.transform(X_raw), nan=0.0, posinf=3.0, neginf=-3.0)
        y = targets

        def loader(indices, shuffle):
            # a zero prior turns PriorResidualGNN into the graph-only model:
            # encoder + shallow head, identical to ResidualOnlyGNN
            prior = np.zeros((len(indices), 1), dtype=np.float32)
            ds = MoleculeGraphDataset([graphs[i] for i in indices], X[indices], y[indices],
                                      prior_preds=prior)
            return DataLoader(ds, batch_size=64, shuffle=shuffle, collate_fn=collate_graph_batch)

        for hidden in hidden_dims:
            for route in ["graph", "hybrid"]:
                model = (
                    PriorResidualGNN(hidden_dim=hidden, depth=3)
                    if route == "graph"
                    else DirectHybridGNN(desc_dim=X.shape[1], hidden_dim=hidden, depth=3)
                )
                val_score, metrics, epoch, val_loss = train_and_eval_model(
                    model,
                    loader(tr, True), loader(va, False), loader(te, False),
                    is_classification=is_cls, seed=split, epochs=40, lr=1e-3,
                )
                rows.append({
                    "dataset": dataset,
                    "split": split,
                    "hidden_dim": hidden,
                    "route": route,
                    "metric": metric,
                    "test_score": metrics[metric],
                    "val_score": val_score,
                    "best_epoch": epoch,
                    "best_val_loss": val_loss,
                })
                print(f"{dataset} split {split} hidden {hidden} {route}: "
                      f"{metric}={metrics[metric]:.4f} (epoch {epoch})", flush=True)

    # descriptor effect per hidden width and split
    effects: list[dict[str, Any]] = []
    for hidden in hidden_dims:
        for split in splits:
            g = next(r for r in rows if r["hidden_dim"] == hidden and r["split"] == split and r["route"] == "graph")
            h = next(r for r in rows if r["hidden_dim"] == hidden and r["split"] == split and r["route"] == "hybrid")
            if is_cls:
                rel = (h["test_score"] - g["test_score"]) / g["test_score"] * 100
            else:
                rel = (g["test_score"] - h["test_score"]) / g["test_score"] * 100
            effects.append({
                "hidden_dim": hidden, "split": split,
                "metric": metric, "descriptor_effect_percent": float(rel),
            })

    by_hidden = {}
    for hidden in hidden_dims:
        values = [e["descriptor_effect_percent"] for e in effects if e["hidden_dim"] == hidden]
        by_hidden[str(hidden)] = {
            "mean_effect_percent": float(np.mean(values)),
            "effects_per_split": values,
            "n_positive": int(np.sum(np.array(values) > 0)),
            "n_negative": int(np.sum(np.array(values) < 0)),
        }

    return {
        "dataset": dataset,
        "metric": metric,
        "splits": splits,
        "hidden_dims": hidden_dims,
        "runs": rows,
        "descriptor_effect": effects,
        "descriptor_effect_by_hidden_dim": by_hidden,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", default=["freesolv"])
    parser.add_argument("--splits", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=HIDDEN_DIMS)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    payload = {
        "note": "one-factor-at-a-time sensitivity of the descriptor effect to encoder width; "
                "all other settings as in run_evidence_routing_experiment.py",
        "datasets": {},
    }
    for dataset in args.dataset:
        payload["datasets"][dataset] = run_dataset(dataset, args.splits, args.hidden_dims)

    out = RESULTS_DIR / "hyperparameter_sensitivity.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"saved {out}")
    for dataset, block in payload["datasets"].items():
        print(f"{dataset}: " + ", ".join(
            f"hidden {h}: {v['mean_effect_percent']:+.1f}% ({v['n_positive']}+/{v['n_negative']}-)"
            for h, v in block["descriptor_effect_by_hidden_dim"].items()))


if __name__ == "__main__":
    main()
