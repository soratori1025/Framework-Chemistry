"""Rebuild the combined routing checkpoint log from the per-dataset result files.

``run_evidence_routing_experiment.py`` writes one audit CSV per dataset
(``routing_checkpoint_audit_<dataset>.csv``) and one result JSON per dataset.
This script merges those records into the released combined log
``routing_checkpoint_audit.csv``, so the log can never mix rows from superseded
runs: it is derived from exactly the result files that the tables are computed
from.

Run::

    python benchmarks/rebuild_routing_audit.py --results-dir benchmarks/results

The columns are: dataset, split, model, route, K, validation_loss,
validation_metric, checkpoint_epoch, test_metric - identical to the historical
layout, where ``test_metric`` is the test RMSE (regression endpoints) or test
AUROC (classification endpoints).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = REPO_ROOT / "benchmarks" / "results"

DATASETS = ["esol", "freesolv", "lipophilicity", "bbbp", "bace"]
COLUMNS = [
    "dataset",
    "split",
    "model",
    "route",
    "K",
    "validation_loss",
    "validation_metric",
    "checkpoint_epoch",
    "test_metric",
]


def rows_from_json(payload: dict, dataset: str) -> list[dict]:
    rows = []
    for record in payload.get("per_split_results", []):
        split = record.get("split")
        for model_key, info in record.get("audit_details", {}).items():
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "model": model_key,
                    "route": model_key.replace("hybrid_", ""),
                    "K": info["K"],
                    "validation_loss": round(float(info["val_loss"]), 5),
                    "validation_metric": round(float(info["val_metric"]), 5),
                    "checkpoint_epoch": info["epoch"],
                    "test_metric": round(float(info["test_metric"]), 5),
                }
            )
    return rows


def rows_from_csv(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Combined log to write (default: <results-dir>/routing_checkpoint_audit.csv)",
    )
    args = parser.parse_args(argv)
    results_dir: Path = args.results_dir
    out_path = args.out or results_dir / "routing_checkpoint_audit.csv"

    rows: list[dict] = []
    missing: list[str] = []
    for dataset in DATASETS:
        json_path = results_dir / f"evidence_routing_{dataset}.json"
        csv_path = results_dir / f"routing_checkpoint_audit_{dataset}.csv"
        dataset_rows: list[dict] = []
        source = None
        if json_path.is_file():
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            dataset_rows = rows_from_json(payload, dataset)
            source = "json"
            if not dataset_rows:
                print(
                    f"{dataset:<15} result file has no checkpoint audit details "
                    "(older runner version)"
                )
        if not dataset_rows and csv_path.is_file():
            dataset_rows = rows_from_csv(csv_path)
            source = "per-dataset csv"
        if not dataset_rows:
            missing.append(dataset)
            continue
        print(f"{dataset:<15} {len(dataset_rows):>4} rows from {source}")
        rows.extend(dataset_rows)

    if not rows:
        print("no routing records found", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: (DATASETS.index(r["dataset"]), int(r["split"]), str(r["model"])))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"wrote {len(rows)} rows to {out_path}")
    datasets_covered = sorted(set(r["dataset"] for r in rows))
    print(f"datasets covered: {len(datasets_covered)} ({', '.join(datasets_covered)})")
    print(f"model runs logged: {len(rows)}")
    if missing:
        print(f"datasets without routing results: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
