from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chemistry_framework.analysis import DatasetAdvisor
from chemistry_framework.config import ExperimentConfig, load_config
from chemistry_framework.data import make_experiment_dataloaders
from chemistry_framework.evaluation import evaluate_model
from chemistry_framework.experiment import prepare_experiment
from chemistry_framework.models import build_hybrid_model
from chemistry_framework.training import train_model

PUBLISHED_REFERENCE = {
    "Chemprop (D-MPNN)": {"mae": 0.380, "rmse": 0.540, "r2": 0.580},
    "AttentiveFP": {"mae": 0.395, "rmse": 0.552, "r2": 0.560},
    "Random Forest (ECFP)": {"mae": 0.442, "rmse": 0.598, "r2": 0.490},
}

def run_experiment_config(
    base_config_path: Path,
    overrides: dict[str, Any],
    tag: str,
    output_dir: Path,
) -> dict[str, Any]:
    config = load_config(base_config_path)

    for k, v in overrides.items():
        if k in {"loss", "scheduler", "warmup_epochs", "epochs", "batch_size", "learning_rate", "patience"}:
            object.__setattr__(config.training, k, v)
        elif k == "descriptors_enabled":
            if hasattr(config.model, "descriptors"):
                object.__setattr__(config.model.descriptors, "enabled", v)
        elif k in {"virtual_node", "aggregation"}:
            if hasattr(config.model, "structure"):
                object.__setattr__(config.model.structure, k, v)

    if "spearman" not in config.evaluation.metrics:
        object.__setattr__(config.evaluation, "metrics", (*config.evaluation.metrics, "spearman"))

    t0 = time.time()
    prepared = prepare_experiment(config)

    task_name = next(iter(config.tasks))
    task_info = config.tasks[task_name]
    target_name = task_info["target"]
    target_cfg = config.dataset.targets[target_name]
    task_kind = task_info.get("kind", "scalar")
    out_dim = len(target_cfg.columns)

    route_name = task_info.get("route")
    desc_dim = len(prepared.route_features[route_name]) if route_name and route_name in prepared.route_features else len(prepared.features.names)

    loaders = make_experiment_dataloaders(prepared, target_name, task_name=task_name)

    model = build_hybrid_model(
        config.model,
        task_kind=task_kind,
        out_dim=out_dim,
        descriptor_dim=desc_dim,
        prior=None,
    )

    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"caco2_{tag}.pt"

    metadata = {
        "config_name": config.name,
        "tag": tag,
        "overrides": overrides,
    }

    res = train_model(
        model,
        loaders.train,
        loaders.validation,
        config.training,
        checkpoint_path=ckpt_path,
        checkpoint_metadata=metadata,
    )

    elapsed = time.time() - t0

    test_metrics, actuals, preds = evaluate_model(
        model,
        loaders.test,
        metrics=config.evaluation.metrics,
        device=config.training.device,
        task_kind="regression",
    )

    return {
        "tag": tag,
        "name": overrides.get("label", tag),
        "best_epoch": res.best_epoch,
        "best_val_loss": res.best_validation_loss,
        "test_metrics": test_metrics,
        "elapsed_seconds": elapsed,
        "overrides": overrides,
    }

def main():
    print("=" * 78)
    print("🧪 CACO-2 COMPREHENSIVE STUDY: ANALYSIS & CONFIGURATION COMPARISON")
    print("=" * 78)

    base_config_path = REPO_ROOT / "benchmarks" / "configs" / "caco2_dmpnn.yaml"
    output_dir = REPO_ROOT / "benchmarks" / "results" / "caco2_study"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- [PHASE 1] DATASET ADVISOR V2 ANALYSIS ---")
    advisor = DatasetAdvisor(seed=42)
    csv_path = REPO_ROOT / "data" / "caco2_wang.tab"
    report = advisor.profile_csv(csv_path, smiles_column="Drug", target_columns=["Y"], dataset_name="caco2_wang")

    console_report = advisor.format_console_report(report)
    print(console_report)

    analysis_dir = output_dir / "analysis"
    report.save_json(analysis_dir / "caco2_health_report.json")
    report.save_markdown(analysis_dir / "caco2_advisory_report.md")

    print("\n--- [PHASE 2] RUNNING MULTI-CONFIG TRAINING COMPARISON ---")

    configurations = [
        {
            "tag": "cfg1_baseline",
            "label": "1. Baseline (Naive D-MPNN, MSE, No Scheduler)",
            "loss": "mse",
            "scheduler": "none",
            "warmup_epochs": 0,
            "descriptors_enabled": False,
            "virtual_node": False,
            "aggregation": "mean",
            "epochs": 35,
            "batch_size": 32,
            "patience": 12,
        },
        {
            "tag": "cfg2_advisor_v1",
            "label": "2. Advisor v1 (Hybrid Features + Huber Loss + Cosine Warmup)",
            "loss": "huber",
            "scheduler": "cosine",
            "warmup_epochs": 3,
            "descriptors_enabled": True,
            "virtual_node": False,
            "aggregation": "mean",
            "epochs": 35,
            "batch_size": 32,
            "patience": 12,
        },
        {
            "tag": "cfg3_breakthrough_virtual_node",
            "label": "3. Breakthrough v2 (Global Virtual Node + Attention Readout)",
            "loss": "huber",
            "scheduler": "cosine",
            "warmup_epochs": 3,
            "descriptors_enabled": True,
            "virtual_node": True,
            "aggregation": "attention",
            "epochs": 35,
            "batch_size": 32,
            "patience": 12,
        },
        {
            "tag": "cfg4_breakthrough_pairwise_ranking",
            "label": "4. Breakthrough v2 (Pairwise Ranking Loss + Virtual Node + Attention)",
            "loss": "pairwise_ranking",
            "scheduler": "cosine",
            "warmup_epochs": 3,
            "descriptors_enabled": True,
            "virtual_node": True,
            "aggregation": "attention",
            "epochs": 35,
            "batch_size": 32,
            "patience": 12,
        },
    ]

    study_results = []
    for cfg in configurations:
        tag = cfg["tag"]
        label = cfg["label"]
        print(f"\n▶ Running: {label}...")
        res = run_experiment_config(base_config_path, cfg, tag, output_dir)
        study_results.append(res)
        m = res["test_metrics"]
        print(f"  ✓ Finished in {res['elapsed_seconds']:.1f}s (Best Val Epoch: {res['best_epoch']})")
        print(f"    Test MAE: {m.get('mae', 0):.4f} | RMSE: {m.get('rmse', 0):.4f} | R²: {m.get('r2', 0):.4f} | Spearman ρ: {m.get('spearman', 0):.4f}")

    print("\n" + "=" * 88)
    print(" 📊 FINAL BENCHMARK COMPARISON TABLE: CACO-2 PERMEABILITY")
    print("=" * 88)
    header = f"{'Configuration':<45} | {'MAE ↓':<8} | {'RMSE ↓':<8} | {'R² ↑':<8} | {'Spearman ρ ↑':<12}"
    print(header)
    print("-" * 88)

    for r in study_results:
        m = r["test_metrics"]
        print(f"{r['name']:<45} | {m.get('mae', 0):<8.4f} | {m.get('rmse', 0):<8.4f} | {m.get('r2', 0):<8.4f} | {m.get('spearman', 0):<12.4f}")

    print("-" * 88)
    print(f"{'[Ref] Published Chemprop (D-MPNN)':<45} | {0.380:<8.4f} | {0.540:<8.4f} | {0.580:<8.4f} | {'N/A':<12}")
    print(f"{'[Ref] Published AttentiveFP':<45} | {0.395:<8.4f} | {0.552:<8.4f} | {0.560:<8.4f} | {'N/A':<12}")
    print(f"{'[Ref] Published Random Forest (ECFP)':<45} | {0.442:<8.4f} | {0.598:<8.4f} | {0.490:<8.4f} | {'N/A':<12}")
    print("=" * 88)

    summary_file = output_dir / "caco2_study_summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "task": "caco2_wang",
                "results": study_results,
                "published_references": PUBLISHED_REFERENCE,
            },
            f,
            indent=2,
        )
    print(f"\nAll study results saved to: {summary_file}")

if __name__ == "__main__":
    main()
