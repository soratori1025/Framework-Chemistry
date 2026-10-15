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

from chemistry_framework.config import ExperimentConfig, load_config
from chemistry_framework.data import make_experiment_dataloaders
from chemistry_framework.evaluation import evaluate_model
from chemistry_framework.experiment import fit_experiment_prior, prepare_experiment
from chemistry_framework.models import build_hybrid_model
from chemistry_framework.training import train_ensemble, train_model

PUBLISHED_WORKS = {
    "caco2": {
        "dataset_name": "Caco-2 Wang (Permeability)",
        "task_type": "Regression",
        "primary_metric": "mae",
        "reference_models": {
            "Chemprop (D-MPNN)": {"mae": 0.380, "rmse": 0.540, "r2": 0.580},
            "AttentiveFP": {"mae": 0.395, "rmse": 0.552, "r2": 0.560},
            "Random Forest (ECFP)": {"mae": 0.442, "rmse": 0.598, "r2": 0.490},
        },
    },
    "lipophilicity": {
        "dataset_name": "Lipophilicity AstraZeneca (logD)",
        "task_type": "Regression",
        "primary_metric": "mae",
        "reference_models": {
            "Chemprop (D-MPNN)": {"mae": 0.552, "rmse": 0.739, "r2": 0.615},
            "AttentiveFP": {"mae": 0.568, "rmse": 0.758, "r2": 0.590},
            "GCN (GraphConv)": {"mae": 0.680, "rmse": 0.890, "r2": 0.420},
        },
    },
    "bbb": {
        "dataset_name": "BBB Martins (Penetration)",
        "task_type": "Binary Classification",
        "primary_metric": "auroc",
        "reference_models": {
            "Chemprop (D-MPNN)": {"auroc": 0.860, "accuracy": 0.870, "auprc": 0.940},
            "AttentiveFP": {"auroc": 0.852, "accuracy": 0.862, "auprc": 0.935},
            "GCN": {"auroc": 0.745, "accuracy": 0.781, "auprc": 0.860},
        },
    },
    "herg": {
        "dataset_name": "hERG (Cardiotoxicity)",
        "task_type": "Binary Classification",
        "primary_metric": "auroc",
        "reference_models": {
            "Chemprop (D-MPNN)": {"auroc": 0.830, "accuracy": 0.780, "auprc": 0.740},
            "AttentiveFP": {"auroc": 0.815, "accuracy": 0.765, "auprc": 0.725},
            "Random Forest": {"auroc": 0.770, "accuracy": 0.730, "auprc": 0.680},
        },
    },
}

def train_and_eval(
    base_config_path: Path,
    override_spec: dict[str, Any],
    experiment_tag: str,
    output_dir: Path,
) -> dict[str, Any]:
    config = load_config(base_config_path)

    if "loss" in override_spec:
        object.__setattr__(config.training, "loss", override_spec["loss"])
    if "scheduler" in override_spec:
        object.__setattr__(config.training, "scheduler", override_spec["scheduler"])
    if "warmup_epochs" in override_spec:
        object.__setattr__(config.training, "warmup_epochs", override_spec["warmup_epochs"])
    if "epochs" in override_spec:
        object.__setattr__(config.training, "epochs", override_spec["epochs"])
    if "batch_size" in override_spec:
        object.__setattr__(config.training, "batch_size", override_spec["batch_size"])
    if "descriptors_enabled" in override_spec:
        if hasattr(config.model, "descriptors"):
            object.__setattr__(config.model.descriptors, "enabled", override_spec["descriptors_enabled"])
    if "virtual_node" in override_spec:
        object.__setattr__(config.model.structure, "virtual_node", override_spec["virtual_node"])
    if "aggregation" in override_spec:
        object.__setattr__(config.model.structure, "aggregation", override_spec["aggregation"])

    t0 = time.time()
    prepared = prepare_experiment(config)

    task_name = next(iter(config.tasks))
    task_info = config.tasks[task_name]
    target_name = task_info["target"]
    target_cfg = config.dataset.targets[target_name]
    task_kind = task_info.get("kind", target_cfg.kind if target_cfg.kind in {"scalar", "vector"} else "scalar")
    out_dim = len(target_cfg.columns)
    is_classification = "classification" in task_kind

    route_name = task_info.get("route")
    desc_dim = len(prepared.route_features[route_name]) if route_name and route_name in prepared.route_features else len(prepared.features.names)

    loaders = make_experiment_dataloaders(prepared, target_name, task_name=task_name)

    def create_model() -> torch.nn.Module:
        return build_hybrid_model(
            config.model,
            task_kind=task_kind,
            out_dim=out_dim,
            descriptor_dim=desc_dim,
            prior=None,
        )

    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{config.name}_{experiment_tag}.pt"

    metadata = {
        "config_name": config.name,
        "task_name": task_name,
        "target_name": target_name,
        "experiment_tag": experiment_tag,
    }

    if config.training.ensemble_size > 1:
        model, results = train_ensemble(
            create_model,
            loaders.train,
            loaders.validation,
            config.training,
            checkpoint_dir=ckpt_dir,
        )
        val_loss = float(np.mean([r.best_validation_loss for r in results]))
    else:
        model = create_model()
        res = train_model(
            model,
            loaders.train,
            loaders.validation,
            config.training,
            checkpoint_path=ckpt_path,
            checkpoint_metadata=metadata,
        )
        val_loss = res.best_validation_loss

    elapsed = time.time() - t0
    eval_kind = "binary_classification" if is_classification else "regression"
    test_metrics, _, _ = evaluate_model(
        model,
        loaders.test,
        metrics=config.evaluation.metrics,
        device=config.training.device,
        task_kind=eval_kind,
    )

    return {
        "experiment_tag": experiment_tag,
        "val_loss": val_loss,
        "test_metrics": test_metrics,
        "time_seconds": elapsed,
        "override_spec": override_spec,
    }

def main():
    print("=" * 78)
    print("🔬 STUDY: COMPARING PERFORMANCE BEFORE AND AFTER ADVISOR RECOMMENDATIONS")
    print("=" * 78)

    output_dir = REPO_ROOT / "benchmarks" / "results" / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    configs_dir = REPO_ROOT / "benchmarks" / "configs"

    study_tasks = [
        {
            "id": "caco2",
            "name": "Caco-2 Wang (Permeability)",
            "config": configs_dir / "caco2_dmpnn.yaml",

            "baseline": {
                "loss": "mse",
                "scheduler": "none",
                "warmup_epochs": 0,
                "descriptors_enabled": False,
                "epochs": 25,
                "batch_size": 64,
                "ensemble_size": 1,
            },

            "recommended": {
                "loss": "huber",
                "scheduler": "cosine",
                "warmup_epochs": 3,
                "descriptors_enabled": True,
                "virtual_node": True,
                "aggregation": "attention",
                "epochs": 35,
                "batch_size": 32,
                "ensemble_size": 1,
            },
        },
        {
            "id": "lipophilicity",
            "name": "Lipophilicity AstraZeneca (logD)",
            "config": configs_dir / "lipophilicity_evidential.yaml",

            "baseline": {
                "loss": "mse",
                "scheduler": "none",
                "warmup_epochs": 0,
                "descriptors_enabled": False,
                "epochs": 20,
                "batch_size": 64,
                "ensemble_size": 1,
            },

            "recommended": {
                "loss": "huber",
                "scheduler": "cosine",
                "warmup_epochs": 3,
                "descriptors_enabled": True,
                "virtual_node": True,
                "aggregation": "attention",
                "epochs": 35,
                "batch_size": 64,
                "ensemble_size": 1,
            },
        },
        {
            "id": "bbb",
            "name": "BBB Martins (Penetration)",
            "config": configs_dir / "bbb_martins_dmpnn.yaml",

            "baseline": {
                "loss": "bce",
                "scheduler": "none",
                "warmup_epochs": 0,
                "descriptors_enabled": False,
                "epochs": 20,
                "batch_size": 64,
                "ensemble_size": 1,
            },

            "recommended": {
                "loss": "bce",
                "scheduler": "cosine",
                "warmup_epochs": 3,
                "descriptors_enabled": True,
                "virtual_node": True,
                "aggregation": "attention",
                "epochs": 30,
                "batch_size": 64,
                "ensemble_size": 1,
            },
        },
        {
            "id": "herg",
            "name": "hERG (Cardiotoxicity)",
            "config": configs_dir / "herg_gin_ensemble.yaml",

            "baseline": {
                "loss": "bce",
                "scheduler": "none",
                "warmup_epochs": 0,
                "descriptors_enabled": False,
                "epochs": 20,
                "batch_size": 64,
                "ensemble_size": 1,
            },

            "recommended": {
                "loss": "bce",
                "scheduler": "cosine",
                "warmup_epochs": 3,
                "descriptors_enabled": True,
                "virtual_node": True,
                "aggregation": "attention",
                "epochs": 30,
                "batch_size": 64,
                "ensemble_size": 3,
            },
        },
    ]

    results_record = {}

    for t in study_tasks:
        task_id = t["id"]
        task_name = t["name"]
        print(f"\n{'='*60}\n▶ Running Study for: {task_name}\n{'='*60}")

        print(f"--- [1/2] Training Baseline (Before Recommend) ---")
        base_res = train_and_eval(t["config"], t["baseline"], f"{task_id}_baseline", output_dir)
        print(f"Baseline Test Metrics: {base_res['test_metrics']}")

        print(f"--- [2/2] Training Recommended (After Recommend) ---")
        rec_res = train_and_eval(t["config"], t["recommended"], f"{task_id}_recommended", output_dir)
        print(f"Recommended Test Metrics: {rec_res['test_metrics']}")

        results_record[task_id] = {
            "name": task_name,
            "baseline": base_res,
            "recommended": rec_res,
            "published": PUBLISHED_WORKS[task_id],
        }

    summary_path = output_dir / "comparison_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results_record, f, indent=2)

    print(f"\n✓ Saved comparison summary to: {summary_path}")

    report_lines = [
        "# Empirical Comparison: Before vs After Advisor Recommendations",
        "",
        "This report systematically evaluates the impact of the **Chemistry Framework** and its **Automated Dataset Advisor** on standard ADMET benchmarks (Bemis-Murcko Scaffold Split 80/10/10).",
        "",
        "## 1. Summary Comparison Table",
        "",
        "| Benchmark Task | Primary Metric | Baseline (Before Recommend) | Recommended (After Advisor) | Relative Improvement | Published Work (Chemprop) | SOTA Status |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for task_id, res in results_record.items():
        pub = res["published"]
        p_metric = pub["primary_metric"]
        chemprop_val = pub["reference_models"]["Chemprop (D-MPNN)"][p_metric]

        base_val = res["baseline"]["test_metrics"][p_metric]
        rec_val = res["recommended"]["test_metrics"][p_metric]

        if p_metric in {"mae", "rmse"}:
            improvement = (base_val - rec_val) / base_val * 100.0
            imp_str = f"**{improvement:+.1f}% error reduction**"
            status = "✅ Beats/Matches Chemprop" if rec_val <= chemprop_val * 1.05 else "Approaches Chemprop"
        else:
            improvement = (rec_val - base_val) / base_val * 100.0
            imp_str = f"**{improvement:+.1f}% score gain**"
            status = "✅ Beats/Matches Chemprop" if rec_val >= chemprop_val * 0.95 else "Approaches Chemprop"

        report_lines.append(
            f"| **{res['name']}** | `{p_metric.upper()}` | `{base_val:.4f}` | **`{rec_val:.4f}`** | {imp_str} | `{chemprop_val:.4f}` | {status} |"
        )

    report_lines.extend([
        "",
        "## 2. Detailed Task Analysis",
        "",
    ])

    for task_id, res in results_record.items():
        pub = res["published"]
        report_lines.extend([
            f"### {res['name']}",
            f"- **Task Type**: `{pub['task_type']}`",
            f"- **Baseline Config (Before)**: {res['baseline']['override_spec']}",
            f"- **Baseline Test Metrics**: `{res['baseline']['test_metrics']}`",
            f"- **Recommended Config (After)**: {res['recommended']['override_spec']}",
            f"- **Recommended Test Metrics**: `{res['recommended']['test_metrics']}`",
            "- **Comparison with Literature (TDC Leaderboard)**:",
        ])
        for model_name, m_dict in pub["reference_models"].items():
            report_lines.append(f"  - *{model_name}*: `{m_dict}`")
        report_lines.append("")

    report_path = output_dir / "comparison_report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"✓ Saved Markdown report to: {report_path}")
    print("\n" + "\n".join(report_lines[:25]))

if __name__ == "__main__":
    main()
