from __future__ import annotations
import argparse
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

BENCHMARK_CONFIGS = [
    ("BBB Martins (Penetration)", "bbb_martins_dmpnn.yaml"),
    ("Caco-2 Wang (Permeability)", "caco2_dmpnn.yaml"),
    ("Lipophilicity (logD)", "lipophilicity_evidential.yaml"),
    ("hERG (Cardiotoxicity)", "herg_gin_ensemble.yaml"),
]

def run_single_benchmark(config_path: Path,results_dir: Path,device: str = "auto",) -> dict[str, Any]:
    print(f"🚀 Running Benchmark: {config_path.name}")
    config = load_config(config_path)
    if device != "auto":
        object.__setattr__(config.training, "device", device)
    ckpt_dir = results_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    prepared = prepare_experiment(config)
    dataset_load_time = time.time() - t0
    print(f"✓ Dataset loaded & split: {len(prepared.dataset)} molecules in {dataset_load_time:.2f}s")

    task_name = next(iter(config.tasks))
    task_info = config.tasks[task_name]
    target_name = task_info["target"]
    target_cfg = config.dataset.targets[target_name]
    task_kind = task_info.get("kind", target_cfg.kind if target_cfg.kind in {"scalar", "vector"} else "scalar")
    out_dim = len(target_cfg.columns)
    is_classification = "classification" in task_kind

    prior_obj = None
    prior_name = task_info.get("prior")
    if prior_name and prior_name in config.priors:
        prior_fit = fit_experiment_prior(prepared, prior_name, target_name=target_name)
        prior_obj = prior_fit.prior

    route_name = task_info.get("route")
    if route_name and route_name in prepared.route_features:
        desc_dim = len(prepared.route_features[route_name])
    else:
        desc_dim = len(prepared.features.names)

    loaders = make_experiment_dataloaders(prepared, target_name, task_name=task_name)

    def create_model() -> torch.nn.Module:
        return build_hybrid_model(
            config.model,
            task_kind=task_kind,
            out_dim=out_dim,
            descriptor_dim=desc_dim,
            prior=prior_obj,
        )
    ckpt_path = ckpt_dir / f"{config.name}_{task_name}.pt"
    train_start = time.time()
    metadata = {
        "config_name": config.name,
        "task_name": task_name,
        "target_name": target_name,
        "task_kind": task_kind,
        "model_spec": config.model,
    }
    if config.training.ensemble_size > 1:
        print(f"⚡ Training Deep Ensemble ({config.training.ensemble_size} members) with scheduler '{config.training.scheduler}'...")
        model, member_results = train_ensemble(
            create_model,
            loaders.train,
            loaders.validation,
            config.training,
            checkpoint_dir=ckpt_dir,
        )
        best_val_loss = float(np.mean([r.best_validation_loss for r in member_results]))
        torch.save(
            {
                "ensemble_state_dicts": [m.state_dict() for m in model.models],
                "metadata": metadata,
            },
            ckpt_path,
        )
    else:
        struct_enc = config.model.get("structure", {}).get("encoder", "mlp") if isinstance(config.model.get("structure"), dict) else "mlp"
        head_kind = config.model.get("head", "standard")
        print(f"⚡ Training Model [Encoder: {struct_enc.upper()}, Head: {head_kind}] for {config.training.epochs} epochs...")
        model = create_model()
        train_res = train_model(
            model,
            loaders.train,
            loaders.validation,
            config.training,
            checkpoint_path=ckpt_path,
            checkpoint_metadata=metadata,
        )
        best_val_loss = train_res.best_validation_loss
    training_time = time.time() - train_start
    print(f"Training complete in {training_time:.2f}s | Best Val Loss: {best_val_loss:.4f}")

    eval_kind = "binary_classification" if is_classification else "regression"
    test_metrics, actual, predicted = evaluate_model(
        model,
        loaders.test,
        metrics=config.evaluation.metrics,
        device=config.training.device,
        task_kind=eval_kind,
    )
    print(f"📊 Test Results: {json.dumps(test_metrics, indent=2)}")

    test_split_indices = prepared.split.test
    test_smiles = [prepared.dataset.smiles[i] for i in test_split_indices]
    pred_records = {
        "smiles": test_smiles[:len(actual)],
        "actual": actual.tolist(),
        "predicted": predicted.tolist(),
    }

    model.eval()
    dev = next(model.parameters()).device if hasattr(model, "parameters") and list(model.parameters()) else torch.device("cpu")
    all_aleatoric = []
    all_epistemic = []
    with torch.no_grad():
        for b_inputs, _ in loaders.test:
            if isinstance(b_inputs, dict):
                from chemistry_framework.training.engine import _transfer_to_device
                b_in = _transfer_to_device(b_inputs, dev)
                out = model(b_in)
                if isinstance(out, dict):
                    if "aleatoric_variance" in out:
                        all_aleatoric.extend(out["aleatoric_variance"].detach().cpu().reshape(-1).tolist())
                    if "epistemic_variance" in out:
                        all_epistemic.extend(out["epistemic_variance"].detach().cpu().reshape(-1).tolist())
    if all_aleatoric and len(all_aleatoric) == len(actual):
        pred_records["aleatoric_std"] = np.sqrt(np.clip(all_aleatoric, 0, None)).tolist()
    if all_epistemic and len(all_epistemic) == len(actual):
        pred_records["epistemic_std"] = np.sqrt(np.clip(all_epistemic, 0, None)).tolist()
    pred_df = pd.DataFrame(pred_records)
    pred_csv_path = results_dir / f"{config.name}_test_predictions.csv"
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"✓ Saved test predictions to: {pred_csv_path.name}")
    return {
        "name": config.name,
        "dataset_name": config.dataset.name,
        "task_name": task_name,
        "task_kind": task_kind,
        "num_molecules": len(prepared.dataset),
        "test_size": len(actual),
        "best_val_loss": round(float(best_val_loss), 4),
        "test_metrics": {k: round(float(v), 4) for k, v in test_metrics.items()},
        "training_time_sec": round(training_time, 2),
        "checkpoint": str(ckpt_path),
        "predictions_csv": str(pred_csv_path),
    }

def main() -> int:
    parser = argparse.ArgumentParser(description="Run chemical benchmarks with Chemistry Framework")
    parser.add_argument(
        "--configs-dir",
        type=str,
        default=str(REPO_ROOT / "benchmarks" / "configs"),
        help="Path to directory containing benchmark yaml files",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(REPO_ROOT / "benchmarks" / "results"),
        help="Path to directory where results and checkpoints will be saved",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to train on ('auto', 'cuda', 'cpu')",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter benchmark configs by keyword (e.g. 'bbb', 'lipo', 'caco2')",
    )
    args = parser.parse_args()
    configs_dir = Path(args.configs_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_list = []
    start_all = time.time()
    for friendly_name, filename in BENCHMARK_CONFIGS:
        if args.filter and args.filter.lower() not in filename.lower():
            continue
        cfg_file = configs_dir / filename
        if not cfg_file.exists():
            print(f"Warning: config file not found: {cfg_file}")
            continue

        res = run_single_benchmark(cfg_file, results_dir, device=args.device)
        res["benchmark_title"] = friendly_name
        summary_list.append(res)
    total_time = time.time() - start_all

    summary_json_path = results_dir / "benchmark_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_list, f, indent=2)

    print("ALL BENCHMARKS COMPLETED - PERFORMANCE SUMMARY")
    print(f"{'Benchmark Dataset':<30} | {'Type':<15} | {'Val Loss':<10} | {'Test Performance':<25} | {'Time (s)'}")
    print("-" * 90)
    for r in summary_list:
        perf_str = ", ".join([f"{k.upper()}: {v:.4f}" for k, v in r["test_metrics"].items() if k != "count"])
        print(f"{r['benchmark_title']:<30} | {r['task_kind']:<15} | {r['best_val_loss']:<10.4f} | {perf_str:<25} | {r['training_time_sec']:.1f}s")
    print(f"Total benchmark wall-clock time: {total_time:.2f}s")

    report_md_path = results_dir / "benchmark_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Chemistry Framework - Benchmark Evaluation Report\n\n")
        f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 1. Overview\n")
        f.write("This report provides empirical validation of the upgraded **Chemistry Framework** across standard Therapeutic Data Commons (TDC) ADMET benchmarks. ")
        f.write("The models evaluate both molecular graph representation learning (Chemprop-style D-MPNN, GIN), physics-informed feature fusion, Evidential Uncertainty Quantification, and Deep Ensembles.\n\n")
        f.write("## 2. Benchmark Summary Table\n\n")
        f.write("| Benchmark | Dataset | Task Type | Architecture | Best Val Loss | Test Performance | Time (s) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in summary_list:
            perf_str = ", ".join([f"**{k.upper()}**: {v:.4f}" for k, v in r["test_metrics"].items() if k != "count"])
            arch = "D-MPNN + Evidential" if "evidential" in r["name"] else ("GIN Ensemble (K=2)" if "ensemble" in r["name"] else "D-MPNN Hybrid")
            f.write(f"| {r['benchmark_title']} | `{r['dataset_name']}` | {r['task_kind']} | {arch} | {r['best_val_loss']} | {perf_str} | {r['training_time_sec']}s |\n")
        f.write("\n## 3. Detailed Dataset Results\n\n")
        for r in summary_list:
            f.write(f"### {r['benchmark_title']}\n")
            f.write(f"- **Dataset**: `{r['dataset_name']}` ({r['num_molecules']} compounds, {r['test_size']} test)\n")
            f.write(f"- **Task Kind**: `{r['task_kind']}`\n")
            f.write(f"- **Best Validation Loss**: `{r['best_val_loss']}`\n")
            f.write(f"- **Test Metrics**:\n")
            for m_k, m_v in r["test_metrics"].items():
                f.write(f"  - `{m_k}`: **{m_v}**\n")
            f.write(f"- **Saved Checkpoint**: `{r['checkpoint']}`\n")
            f.write(f"- **Test Predictions**: `{r['predictions_csv']}`\n\n")
    print(f"\nMarkdown benchmark report saved to: {report_md_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
