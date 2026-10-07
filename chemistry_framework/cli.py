"""Command-line entry points for validating and preparing experiment configs."""
from __future__ import annotations
import argparse
import json
import sys
from .config import load_config
from .experiment import prepare_experiment

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
    args = parser.parse_args(argv)
    try:
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
        if prepared.analysis is not None:
            report_path = str(
                config.resolve_path(config.analysis.output_dir)
                / f"{config.name}-analysis.json"
            )
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
