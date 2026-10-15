from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any
import numpy as np
import torch
from .config import load_config
from .data.graph import collate_mol_graphs
from .data.loaders import make_experiment_dataloaders
from .evaluation.metrics import evaluate_model, load_model_checkpoint
from .experiment import fit_experiment_prior, prepare_experiment
from .models.hybrid import build_hybrid_model
from .training.engine import train_ensemble, train_model
from .analysis.advisor import DatasetAdvisor

def _run_train(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    prepared = prepare_experiment(config)

    if args.task:
        task_name = args.task
        task_info = config.tasks.get(task_name)
        if task_info is None:
            raise KeyError(f"unknown task {task_name!r}")
        target_name = task_info["target"]
    elif args.target:
        target_name = args.target
        task_name = target_name
        task_info = config.tasks.get(task_name)
    else:

        if config.tasks:
            task_name = next(iter(config.tasks))
            task_info = config.tasks[task_name]
            target_name = task_info["target"]
        else:
            target_name = next(iter(config.dataset.targets))
            task_name = target_name
            task_info = {}
    target_cfg = config.dataset.targets[target_name]
    task_kind = task_info.get("kind", target_cfg.kind if target_cfg.kind in {"scalar", "vector"} else "scalar")
    out_dim = len(target_cfg.columns)

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

    loaders = make_experiment_dataloaders(
        prepared,
        target_name,
        task_name=task_name,
    )

    def create_model() -> torch.nn.Module:
        return build_hybrid_model(
            config.model,
            task_kind=task_kind,
            out_dim=out_dim,
            descriptor_dim=desc_dim,
            prior=prior_obj,
        )
    ckpt_dir = config.resolve_path(args.checkpoint_dir or config.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{config.name}_{task_name}.pt"
    metadata = {
        "config_name": config.name,
        "task_name": task_name,
        "target_name": target_name,
        "task_kind": task_kind,
        "out_dim": out_dim,
        "descriptor_dim": desc_dim,
        "model_spec": config.model,
    }
    if config.training.ensemble_size > 1:
        model, results = train_ensemble(
            create_model,
            loaders.train,
            loaders.validation,
            config.training,
            checkpoint_dir=ckpt_dir,
        )
        best_val_loss = float(np.mean([r.best_validation_loss for r in results]))

        torch.save(
            {
                "ensemble_state_dicts": [m.state_dict() for m in model.models],
                "metadata": metadata,
            },
            ckpt_path,
        )
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
        best_val_loss = res.best_validation_loss

    test_metrics, _, _ = evaluate_model(
        model,
        loaders.test,
        metrics=config.evaluation.metrics,
        device=config.training.device,
        task_kind="binary_classification" if "classification" in task_kind else "regression",
    )
    print(
        json.dumps(
            {
                "trained": True,
                "name": config.name,
                "task": task_name,
                "target": target_name,
                "best_validation_loss": best_val_loss,
                "test_metrics": test_metrics,
                "checkpoint": str(ckpt_path),
            },
            indent=2,
        )
    )
    return 0

def _run_predict(args: argparse.Namespace) -> int:
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint file not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata", {})
    model_spec = metadata.get("model_spec", {})
    task_kind = metadata.get("task_kind", "scalar")
    out_dim = metadata.get("out_dim", 1)
    desc_dim = metadata.get("descriptor_dim", 0)

    if "ensemble_state_dicts" in checkpoint:
        members = []
        for state in checkpoint["ensemble_state_dicts"]:
            m = build_hybrid_model(model_spec, task_kind=task_kind, out_dim=out_dim, descriptor_dim=desc_dim)
            m.load_state_dict(state)
            m.eval()
            members.append(m)
        from .models.ensemble import EnsembleModel
        model = EnsembleModel(members)
    else:
        model = build_hybrid_model(model_spec, task_kind=task_kind, out_dim=out_dim, descriptor_dim=desc_dim)
        load_model_checkpoint(model, ckpt_path)
        model.eval()

    input_path = Path(args.input).expanduser().resolve()
    smiles_col = args.smiles_column or "smiles"
    smiles_list: list[str] = []
    with input_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            val = (row.get(smiles_col) or "").strip()
            if val:
                smiles_list.append(val)
    if not smiles_list:
        raise ValueError(f"no molecules found in {input_path}")

    batch_size = args.batch_size or 64
    all_preds: list[float | list[float]] = []
    all_vars: list[float | None] = []
    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            chunk = smiles_list[i : i + batch_size]
            bg = collate_mol_graphs(chunk)
            out = model({"graph": bg})
            preds = out.get("prediction", out.get("mean"))
            if task_kind in {"binary_classification", "multilabel_classification"}:
                preds = torch.sigmoid(preds)
            preds_arr = preds.cpu().numpy()
            vars_arr = out["variance"].cpu().numpy() if "variance" in out else None
            for row_idx in range(len(chunk)):
                if out_dim == 1:
                    all_preds.append(float(preds_arr[row_idx].reshape(-1)[0]))
                else:
                    all_preds.append([float(x) for x in preds_arr[row_idx]])
                all_vars.append(float(vars_arr[row_idx].reshape(-1)[0]) if vars_arr is not None else None)

    output_path = Path(args.output).expanduser().resolve() if args.output else input_path.with_name(f"{input_path.stem}_preds.csv")
    has_var = any(v is not None for v in all_vars)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        header = ["smiles", "prediction"]
        if has_var:
            header.append("variance")
        writer.writerow(header)
        for s, p, v in zip(smiles_list, all_preds, all_vars):
            row_out = [s, json.dumps(p) if isinstance(p, list) else f"{p:.6g}"]
            if has_var:
                row_out.append(f"{v:.6g}" if v is not None else "")
            writer.writerow(row_out)
    print(json.dumps({"predicted": True, "count": len(smiles_list), "output": str(output_path)}, indent=2))
    return 0

def _run_inspect(args: argparse.Namespace) -> int:
    advisor = DatasetAdvisor()
    target_path = Path(args.input).expanduser().resolve()
    if not target_path.exists():
        raise FileNotFoundError(f"input file not found: {target_path}")
    if target_path.suffix.lower() in {".yaml", ".yml"}:
        report = advisor.profile_config(target_path)
    else:
        target_cols = [args.target] if args.target else None
        report = advisor.profile_csv(
            csv_path=target_path,
            smiles_column=args.smiles_column,
            target_columns=target_cols,
        )
    print(advisor.format_console_report(report))
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"{report.dataset_name}-health-report.json"
        md_path = out_dir / f"{report.dataset_name}-advisory-report.md"
        report.save_json(json_path)
        report.save_markdown(md_path)
        print(f"Saved analysis reports to: {out_dir}")
    if args.generate_config:
        cfg_path = Path(args.generate_config).expanduser().resolve()
        report.save_config(cfg_path)
        print(f"Generated recommended YAML training config to: {cfg_path}")
    return 0

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chemistry-framework")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate an experiment YAML file")
    validate.add_argument("config")
    prepare = commands.add_parser(
        "prepare",
        help="load data, build train-only feature vocabularies, split, cache, and analyze",
    )
    prepare.add_argument("config")
    analyze = commands.add_parser(
        "analyze",
        help="extract chemical features and write the feature evidence scorecard",
    )
    analyze.add_argument("config")
    analyze.add_argument("--inspect", action="store_true", help="run dataset profiling and advisory report")
    inspect_cmd = commands.add_parser(
        "inspect",
        help="profile dataset/features, evaluate data quality risks, and recommend training configs",
    )
    inspect_cmd.add_argument("input", help="path to dataset CSV file or experiment YAML config")
    inspect_cmd.add_argument("--smiles-column", default=None, help="SMILES column name (auto-detected if omitted)")
    inspect_cmd.add_argument("--target", default=None, help="target column name (auto-detected if omitted)")
    inspect_cmd.add_argument("--generate-config", default=None, help="path to save recommended YAML configuration")
    inspect_cmd.add_argument("--output-dir", default=None, help="directory to write JSON and Markdown reports")
    train_cmd = commands.add_parser(
        "train",
        help="train configured hybrid graph/descriptor/prior models",
    )
    train_cmd.add_argument("config")
    train_cmd.add_argument("--task", default=None, help="specific task to train")
    train_cmd.add_argument("--target", default=None, help="specific target to train")
    train_cmd.add_argument("--checkpoint-dir", default=None, help="checkpoint output directory")
    predict_cmd = commands.add_parser(
        "predict",
        help="predict properties for a file of SMILES using a trained checkpoint",
    )
    predict_cmd.add_argument("--checkpoint", required=True, help="path to .pt checkpoint")
    predict_cmd.add_argument("--input", required=True, help="path to input CSV file")
    predict_cmd.add_argument("--smiles-column", default="smiles", help="SMILES column name")
    predict_cmd.add_argument("--output", default=None, help="output CSV path")
    predict_cmd.add_argument("--batch-size", type=int, default=64, help="inference batch size")
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            return _run_inspect(args)
        if args.command == "analyze" and getattr(args, "inspect", False):
            args.input = args.config
            args.smiles_column = None
            args.target = None
            args.generate_config = None
            args.output_dir = None
            return _run_inspect(args)
        if args.command == "train":
            return _run_train(args)
        if args.command == "predict":
            return _run_predict(args)
            return _run_predict(args)
        config = load_config(args.config)
        if args.command == "validate":
            print(
                json.dumps(
                    {
                        "valid": True,
                        "name": config.name,
                        "targets": list(config.dataset.targets),
                        "feature_blocks": list(config.features),
                        "tasks": list(config.tasks),
                    },
                    indent=2,
                )
            )
            return 0
        prepared = prepare_experiment(config)
        report_path = None
        feature_evidence_paths = None
        if prepared.analysis is not None:
            output_dir = config.resolve_path(config.analysis.output_dir)
            report_path = str(output_dir / f"{config.name}-analysis.json")
            if prepared.feature_evidence is not None:
                feature_evidence_paths = {
                    "json": str(output_dir / f"{config.name}-feature-evidence.json"),
                    "csv": str(output_dir / f"{config.name}-feature-evidence.csv"),
                    "html": str(output_dir / f"{config.name}-feature-report.html"),
                    "heatmap": str(output_dir / f"{config.name}-correlation-heatmap.png"),
                }
        print(
            json.dumps(
                {
                    "prepared": True,
                    "name": config.name,
                    "dataset_rows": len(prepared.dataset),
                    "split_sizes": {
                        "train": len(prepared.split.train),
                        "validation": len(prepared.split.validation),
                        "test": len(prepared.split.test),
                    },
                    "feature_count": len(prepared.features.names),
                    "analysis_report": report_path,
                    "feature_evidence": feature_evidence_paths,
                    "feature_fingerprint": prepared.feature_fingerprint,
                },
                indent=2,
            )
        )
        return 0
    except (FileNotFoundError, ImportError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"chemistry-framework: error: {exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
