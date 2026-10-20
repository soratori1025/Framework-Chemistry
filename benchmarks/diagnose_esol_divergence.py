"""Diagnose the two divergent runs visible in the per-split record.

Two runs in the benchmark stand out: the Chemprop v2.3.1 descriptor hybrid on
ESOL split 3 (test RMSE 2.76 while beating graph-only on the other nine splits)
and the framework ``prior_rec`` route on the same split (test RMSE 13.23).  Both
enter the model through a descriptor block, so the diagnosis tests three
explanations: (i) optimisation instability (does it recur with other seeds?),
(ii) descriptor extrapolation in that split, and (iii) an unstable baseline
prior.

Framework part (run with the benchmark environment):

    python benchmarks/diagnose_esol_divergence.py --part framework

Chemprop part (run with the environment that has Chemprop installed):

    python benchmarks/diagnose_esol_divergence.py --part chemprop
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
DATA_DIR = REPO_ROOT / "data" / "moleculenet"
SPLITS_DIR = REPO_ROOT / "benchmarks" / "splits"

SPLIT = 3  # the split on which both descriptor hybrids failed on ESOL


def _esol_split(split: int = SPLIT):
    from sklearn.preprocessing import StandardScaler
    from chemistry_framework.features.pharmacophore import extract_rdkit_2d_descriptors

    manifest = json.loads((DATA_DIR / "dataset_manifest.json").read_text())["esol"]
    df = pd.read_csv(REPO_ROOT / manifest["clean_csv"])
    smiles = df[manifest["smiles_col"]].astype(str).tolist()
    targets = df[manifest["targets"][0]].values.astype(np.float32)
    X_raw, feature_names = extract_rdkit_2d_descriptors(smiles)

    split_path = REPO_ROOT / manifest["splits"][f"seed_{split}"]["split_file"]
    sp = json.loads(split_path.read_text())
    train_idx, val_idx, test_idx = sp["train"], sp["val"], sp["test"]

    X_clean = np.nan_to_num(X_raw, nan=0.0, posinf=1e4, neginf=-1e4)
    scaler = StandardScaler().fit(X_clean[train_idx])
    X_all = np.nan_to_num(scaler.transform(X_clean), nan=0.0, posinf=3.0, neginf=-3.0)
    # unclipped variant, to expose the extrapolation that clipping hides
    X_unclipped = np.nan_to_num(scaler.transform(X_clean), nan=0.0, posinf=1e6, neginf=-1e6)
    return {
        "smiles": smiles,
        "targets": targets,
        "feature_names": feature_names,
        "X_all": X_all,
        "X_unclipped": X_unclipped,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }


def descriptor_extrapolation() -> dict[str, Any]:
    """How far outside the training range do the held-out partitions lie?"""
    data = _esol_split()
    X = data["X_unclipped"]
    tr, va, te = data["train_idx"], data["val_idx"], data["test_idx"]
    names = data["feature_names"]

    def exceed(idx: np.ndarray) -> tuple[float, str]:
        lo, hi = X[tr].min(axis=0), X[tr].max(axis=0)
        span = np.maximum(hi - lo, 1e-9)
        out = np.where(X[idx] < lo, (lo - X[idx]) / span, 0.0) + np.where(X[idx] > hi, (X[idx] - hi) / span, 0.0)
        j = int(np.unravel_index(np.argmax(out), out.shape)[1])
        return float(out.max()), names[j]

    worst_test, worst_test_name = exceed(te)
    worst_val, worst_val_name = exceed(va)

    per_split = {}
    for split in range(1, 11):
        d = _esol_split(split)
        Xs = d["X_unclipped"]
        trs = d["train_idx"]
        lo, hi = Xs[trs].min(axis=0), Xs[trs].max(axis=0)
        span = np.maximum(hi - lo, 1e-9)
        out = np.where(Xs[d["test_idx"]] < lo, (lo - Xs[d["test_idx"]]) / span, 0.0) + \
            np.where(Xs[d["test_idx"]] > hi, (Xs[d["test_idx"]] - hi) / span, 0.0)
        per_split[str(split)] = {
            "max_test_extrapolation_units": float(out.max()),
            "n_cells_beyond_one_training_span": int(np.sum(out > 1.0)),
        }

    # clipped features: what the model actually sees
    clipped_max = float(np.abs(data["X_all"][te]).max())
    return {
        "worst_held_out_extrapolation_units": worst_test,
        "worst_held_out_descriptor": worst_test_name,
        "worst_validation_extrapolation_units": worst_val,
        "worst_validation_descriptor": worst_val_name,
        "max_abs_clipped_standardised_test_value": clipped_max,
        "per_split_test_extrapolation": per_split,
    }


def extreme_standardised_inputs(threshold: float = 5.0) -> dict[str, Any]:
    """Which descriptors enter the model with extreme standardised values?

    ``StandardScaler`` divides by the training standard deviation, so a
    descriptor that is nearly constant in the training partition produces large
    standardised inputs for held-out molecules.  The benchmark harness clips
    infinities but not large finite values, so these magnitudes reach the
    descriptor branch unchanged.
    """
    per_split: dict[str, Any] = {}
    for split in range(1, 11):
        data = _esol_split(split)
        X = data["X_all"]
        te = data["test_idx"]
        names = data["feature_names"]
        tr = data["train_idx"]
        mag = np.abs(X[te])
        rows = []
        for j in range(mag.shape[1]):
            worst = int(np.argmax(mag[:, j]))
            if mag[worst, j] >= threshold:
                rows.append({
                    "feature": names[j],
                    "max_abs_standardised_test_value": float(mag[worst, j]),
                    "raw_train_std": float(np.std(data["X_unclipped"][tr, j])),
                    "n_test_molecules_above_threshold": int(np.sum(mag[:, j] >= threshold)),
                })
        rows.sort(key=lambda r: -r["max_abs_standardised_test_value"])
        per_split[str(split)] = {
            "n_features_above_threshold": len(rows),
            "n_test_molecules_affected": int(np.sum(mag >= threshold)),
            "max_abs_standardised_test_value": float(mag.max()),
            "top_features": rows[:8],
        }
    return {"threshold": threshold, "per_split": per_split}


def framework_part(seeds_test: list[int]) -> dict[str, Any]:
    import torch
    from sklearn.linear_model import Ridge
    from chemistry_framework.feature_analysis import select_evidence_recommendation
    from benchmarks.run_evidence_routing_experiment import (
        DirectHybridGNN,
        MoleculeGraphDataset,
        PriorResidualGNN,
        collate_graph_batch,
        train_and_eval_model,
    )
    from chemistry_framework.data.graph import mol_to_graph
    from torch.utils.data import DataLoader

    torch.set_num_threads(4)
    data = _esol_split()
    graphs = [mol_to_graph(s) for s in data["smiles"]]
    X, y = data["X_all"], np.asarray(data["targets"], dtype=np.float32)
    tr, va, te = data["train_idx"], data["val_idx"], data["test_idx"]

    routing = json.loads((RESULTS_DIR / "evidence_routing_esol.json").read_text())
    k_star = int(routing["k_distribution"][SPLIT - 1])

    idx = select_evidence_recommendation(
        X_train=X[tr], y_train=y[tr], feature_names=data["feature_names"], k=k_star,
        redundancy_threshold=0.85,
    )
    prior = Ridge(alpha=1.0, random_state=SPLIT).fit(X[tr][:, idx], y[tr])
    prior_preds = {
        "train": prior.predict(X[tr][:, idx])[:, None],
        "val": prior.predict(X[va][:, idx])[:, None],
        "test": prior.predict(X[te][:, idx])[:, None],
    }
    prior_stats = {
        "k_star": k_star,
        "coefficient_abs_max": float(np.abs(prior.coef_).max()),
        "train_abs_max": float(np.abs(prior_preds["train"]).max()),
        "val_abs_max": float(np.abs(prior_preds["val"]).max()),
        "test_abs_max": float(np.abs(prior_preds["test"]).max()),
        "test_prediction_range": [float(prior_preds["test"].min()), float(prior_preds["test"].max())],
        "target_train_range": [float(y[tr].min()), float(y[tr].max())],
    }

    def run(model_kind: str, seed: int) -> dict[str, Any]:
        if model_kind == "prior_rec":
            train_ds = MoleculeGraphDataset([graphs[i] for i in tr], X[tr][:, idx], y[tr], prior_preds=prior_preds["train"])
            val_ds = MoleculeGraphDataset([graphs[i] for i in va], X[va][:, idx], y[va], prior_preds=prior_preds["val"])
            test_ds = MoleculeGraphDataset([graphs[i] for i in te], X[te][:, idx], y[te], prior_preds=prior_preds["test"])
            model = PriorResidualGNN(hidden_dim=128, depth=3)
        else:
            train_ds = MoleculeGraphDataset([graphs[i] for i in tr], X[tr][:, idx], y[tr])
            val_ds = MoleculeGraphDataset([graphs[i] for i in va], X[va][:, idx], y[va])
            test_ds = MoleculeGraphDataset([graphs[i] for i in te], X[te][:, idx], y[te])
            model = DirectHybridGNN(desc_dim=len(idx), hidden_dim=128, depth=3)
        loaders = [
            DataLoader(ds, batch_size=64, shuffle=sh, collate_fn=collate_graph_batch)
            for ds, sh in ((train_ds, True), (val_ds, False), (test_ds, False))
        ]
        val_score, metrics, epoch, val_loss = train_and_eval_model(
            model, *loaders, is_classification=False, seed=seed, epochs=40, lr=1e-3,
        )
        model.eval()
        preds = []
        with torch.no_grad():
            for g, d, yy, pprior in loaders[2]:
                preds.append(model(g, d, pprior).cpu().numpy().ravel())
        p = np.concatenate(preds)
        return {
            "seed": seed,
            "test_rmse": metrics["rmse"],
            "val_metric": val_score,
            "best_epoch": epoch,
            "best_val_loss": val_loss,
            "test_prediction_abs_max": float(np.abs(p).max()),
            "test_prediction_min": float(p.min()),
            "test_prediction_max": float(p.max()),
            "target_test_range": [float(y[te].min()), float(y[te].max())],
        }

    out: dict[str, Any] = {"k_star": k_star, "prior_statistics": prior_stats, "runs": {}}
    for kind in ["prior_rec", "hybrid_rec"]:
        runs = [run(kind, s) for s in seeds_test]
        out["runs"][kind] = runs
        rmse = [r["test_rmse"] for r in runs]
        out[f"{kind}_summary"] = {
            "n_seeds": len(runs),
            "n_divergent_runs": int(np.sum(np.array(rmse) > 2.0)),
            "rmse_values": rmse,
        }
        print(f"{kind}: RMSE per seed {['%.3f' % v for v in rmse]} "
              f"(divergent: {out[f'{kind}_summary']['n_divergent_runs']}/{len(runs)})")
    return out


def chemprop_part(seeds_test: list[int], clip_z: float | None = None) -> dict[str, Any]:
    import torch
    from chemprop import data, models
    from lightning import pytorch as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import StandardScaler
    from benchmarks._descriptor_contract import select_contract_columns
    from benchmarks.run_chemprop_v2_benchmarks import build_chemprop_mpnn
    from chemistry_framework.features.pharmacophore import extract_rdkit_2d_descriptors

    torch.set_num_threads(4)
    df = pd.read_csv(DATA_DIR / "esol_clean.csv")
    smiles = df["smiles"].astype(str).tolist()
    raw_targets = df["measured log solubility in mols per litre"].values.astype(np.float32)
    desc_matrix, _ = select_contract_columns(*extract_rdkit_2d_descriptors(smiles))
    desc_matrix = np.nan_to_num(desc_matrix, nan=0.0, posinf=3.0, neginf=-3.0)

    sp = json.loads((SPLITS_DIR / f"esol_seed_{SPLIT}.json").read_text())
    tr = sp.get("train_indices") or sp["train"]
    va = sp.get("val_indices") or sp["val"]
    te = sp.get("test_indices") or sp["test"]

    scaler_y = StandardScaler().fit(raw_targets[tr, None])
    y_tr = scaler_y.transform(raw_targets[tr, None]).ravel()
    y_va = scaler_y.transform(raw_targets[va, None]).ravel()
    y_te = scaler_y.transform(raw_targets[te, None]).ravel()

    scaler_x = StandardScaler().fit(desc_matrix[tr])
    X_all = np.zeros_like(desc_matrix)
    X_all[tr] = scaler_x.transform(desc_matrix[tr])
    X_all[va] = scaler_x.transform(desc_matrix[va])
    X_all[te] = scaler_x.transform(desc_matrix[te])
    if clip_z is not None:
        # Intervention: winsorise the standardised inputs, leaving everything
        # else (data, split, model, seed) identical.
        X_all = np.clip(X_all, -clip_z, clip_z)

    runs = []
    for seed in seeds_test:
        L.seed_everything(seed, workers=True)

        def make(pts, y):
            points = [
                data.MoleculeDatapoint.from_smi(
                    smiles[i], y=[float(y[k])], x_d=X_all[i].astype(np.float32),
                )
                for k, i in enumerate(pts)
            ]
            return data.MoleculeDataset(points)

        ds_tr, ds_va, ds_te = make(tr, y_tr), make(va, y_va), make(te, y_te)
        loader_tr = data.build_dataloader(ds_tr, batch_size=64, shuffle=True)
        loader_va = data.build_dataloader(ds_va, batch_size=64, shuffle=False)
        loader_te = data.build_dataloader(ds_te, batch_size=64, shuffle=False)

        mpnn = build_chemprop_mpnn("regression", d_xd=X_all.shape[1], d_h=300, depth=3)
        early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=10)
        checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, save_weights_only=True)
        trainer = L.Trainer(
            max_epochs=50, accelerator="cpu", devices=1,
            callbacks=[early_stop, checkpoint], logger=False,
            enable_progress_bar=False, enable_model_summary=False,
        )
        trainer.fit(mpnn, loader_tr, loader_va)

        best = models.MPNN.load_from_checkpoint(checkpoint.best_model_path)
        preds = torch.cat(trainer.predict(best, loader_te), dim=0).cpu().numpy().ravel()
        preds_original = scaler_y.inverse_transform(preds[:, None]).ravel()
        rmse = float(np.sqrt(mean_squared_error(raw_targets[te], preds_original)))
        best_val = trainer.callback_metrics.get("val_loss")
        runs.append({
            "seed": seed,
            "test_rmse_original_units": rmse,
            "epochs_run": int(trainer.current_epoch),
            "best_val_loss": float(best_val) if best_val is not None else None,
            "test_prediction_abs_max_standardised": float(np.abs(preds).max()),
            "test_prediction_abs_max_original_units": float(np.abs(preds_original).max()),
        })
        print(f"chemprop seed {seed}: test RMSE={rmse:.3f} (original units), "
              f"epochs={trainer.current_epoch}, max|pred|={np.abs(preds_original).max():.2f}")

    rmse = [r["test_rmse_original_units"] for r in runs]
    return {
        "seeds": seeds_test,
        "clip_z": clip_z,
        "runs": runs,
        "n_divergent_runs": int(np.sum(np.array(rmse) > 2.0)),
        "rmse_values": rmse,
        "rmse_mean": float(np.mean(rmse)),
        "rmse_median": float(np.median(rmse)),
        "descriptor_standardised_abs_max_test": float(np.abs(X_all[te]).max()),
        "descriptor_standardised_abs_max_train": float(np.abs(X_all[tr]).max()),
        "target_original_test_range": [float(raw_targets[te].min()), float(raw_targets[te].max())],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["framework", "chemprop", "extrapolation"], required=True)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument(
        "--clip-z", type=float, default=None,
        help="chemprop only: winsorise standardised descriptors at +/- this value "
             "(intervention arm; the result is written to a separate file)",
    )
    parser.add_argument("--out-suffix", type=str, default="",
                        help="extra suffix for the output file name")
    args = parser.parse_args()

    suffix = "" if args.clip_z is None else f"_clip{args.clip_z:g}"
    suffix += args.out_suffix
    out_path = RESULTS_DIR / f"divergence_diagnosis_{args.part}{suffix}.json"
    if args.part == "extrapolation":
        payload = {
            "split_analysed": SPLIT,
            "descriptor_extrapolation": descriptor_extrapolation(),
            "extreme_standardised_inputs": extreme_standardised_inputs(),
        }
    elif args.part == "framework":
        seeds = args.seeds or [SPLIT, SPLIT + 100, SPLIT + 200]
        payload = {
            "split_analysed": SPLIT,
            "descriptor_extrapolation": descriptor_extrapolation(),
            "extreme_standardised_inputs": extreme_standardised_inputs(),
            "framework": framework_part(seeds),
        }
    else:
        seeds = args.seeds or [SPLIT, SPLIT + 100, SPLIT + 200]
        payload = {"split_analysed": SPLIT, "chemprop": chemprop_part(seeds, clip_z=args.clip_z)}

    out_path.write_text(json.dumps(payload, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
