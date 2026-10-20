"""Structure-level audit: duplicate/conflict inventory and range-based OOD counts.

This runner regenerates the structure-audit artefacts used by the manuscript:

``benchmarks/results/duplicate_audit_report.json``
    Per endpoint: record and unique-InChIKey counts, duplicated structures and
    records, the population SD of the target, the half-SD conflict threshold,
    the largest within-structure target span, the number of conflicting
    structures, and, for every scaffold manifest, the duplicated structures that
    cross the train/validation/test boundaries.

``benchmarks/results/duplicate_audit_summary.csv``
    The same inventory as a flat table (one row per endpoint), with the
    boundary-crossing counts per split seed.

``benchmarks/results/bbbp_duplicate_groups.csv``
    The duplicated BBBP structures with their labels, the conflict flag and the
    SMILES of the records in each group.  This is the group list that the
    duplicate-controlled sensitivity analysis removes or collapses.

``benchmarks/results/ood_shift_all_splits.json``
    Per split of ESOL, Lipophilicity and BBBP: the number of descriptors whose
    test values fall outside the training range, that number as a percentage of
    the descriptor block, and the percentage of test molecules carrying at least
    one such descriptor.

Definitions
-----------
* Structures are identified by standard InChIKey (``Chem.MolToInchiKey``); two
  records with the same key are the same chemical species even when they are
  written differently.
* A duplicated structure is *conflicting* when its labels disagree
  (classification) or when its target span exceeds half the population SD of
  the endpoint (regression).  The ordering of `targets` follows the record
  order of the cleaned dataset.
* A descriptor is out of distribution for a split when at least one test value
  falls outside the closed training range of that descriptor.  Descriptor
  values are the ones produced by
  :func:`chemistry_framework.features.pharmacophore.extract_rdkit_2d_descriptors`,
  which replaces non-finite values with zero.

The number of descriptors in the block is read from the extractor rather than
hard-coded, because ``rdkit.Chem.Descriptors.descList`` grows between RDKit
releases: the manuscript used RDKit 2023.09.6 (210 descriptors), while RDKit
2026.09.1 exposes 217.  The archived counts in the manuscript were produced with
the pinned version.

Usage::

    python benchmarks/run_structure_audit.py
    python benchmarks/run_structure_audit.py --seeds 1 2 3 4 5 6 7 8 9 10 --datasets esol lipophilicity bbbp
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks._seeds import DEFAULT_SEEDS, add_seed_argument, resolve_seeds  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "moleculenet"
SPLITS_DIR = REPO_ROOT / "benchmarks" / "splits"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"

#: Endpoints whose range shift is reported split by split (scaffold splits only).
OOD_DATASETS = ("esol", "lipophilicity", "bbbp")

#: Endpoints in the manuscript's order.
ALL_DATASETS = ("esol", "freesolv", "lipophilicity", "bbbp", "bace")

#: Standardised test value above which a descriptor input is reported as an
#: extrapolation hazard.  Not a validated cut-off: it is the magnitude at which
#: a linear component of a standardised model can dominate the prediction.
MAGNITUDE_THRESHOLD = 10.0


def load_metadata(dataset: str, data_dir: Path) -> dict[str, Any]:
    """Return smiles column, target columns, task type and clean CSV path."""
    manifest_path = data_dir / "dataset_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        if dataset in manifest:
            meta = dict(manifest[dataset])
            meta["clean_path"] = REPO_ROOT / meta["clean_csv"]
            return meta

    # Fallback for a bare directory of cleaned CSVs (used when replaying the
    # audit outside the full data-prep pipeline).
    from benchmarks.prepare_moleculenet_data import DATASET_CONFIGS

    cfg = DATASET_CONFIGS[dataset]
    return {
        "dataset": dataset,
        "smiles_col": cfg["smiles_col"],
        "targets": cfg["target_cols"],
        "task_type": cfg["task_type"],
        "clean_path": data_dir / f"{dataset}_clean.csv",
    }


def read_clean_frame(meta: dict[str, Any]) -> tuple[pd.DataFrame, str, str]:
    df = pd.read_csv(meta["clean_path"])
    smiles_col = meta["smiles_col"]
    if smiles_col not in df.columns:
        for candidate in ("smiles", "mol"):
            if candidate in df.columns:
                smiles_col = candidate
                break
        else:
            raise ValueError(
                f"{meta['dataset']}: no SMILES column in {meta['clean_path']}"
            )
    target_col = meta["targets"][0]
    if target_col not in df.columns:
        raise ValueError(
            f"{meta['dataset']}: target column {target_col!r} missing from "
            f"{meta['clean_path']}"
        )
    return df, smiles_col, target_col


def structure_groups(
    df: pd.DataFrame, smiles_col: str, target_col: str
) -> tuple[list[str], np.ndarray, dict[str, list[int]]]:
    """Canonical SMILES, targets and InChIKey groups in record order."""
    smiles, targets, keys = [], [], []
    for value, target in zip(df[smiles_col].astype(str), df[target_col].values):
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            continue
        key = Chem.MolToInchiKey(mol)
        if not key:
            continue
        smiles.append(Chem.MolToSmiles(mol))
        targets.append(float(target))
        keys.append(key)

    groups: dict[str, list[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    return smiles, np.asarray(targets, dtype=float), groups


def audit_duplicates(
    dataset: str,
    meta: dict[str, Any],
    seeds: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[int]]]:
    df, smiles_col, target_col = read_clean_frame(meta)
    is_classification = "classification" in meta["task_type"]
    smiles, targets, groups = structure_groups(df, smiles_col, target_col)

    duplicated = {key: idx for key, idx in groups.items() if len(idx) > 1}
    target_sd = float(np.std(targets))
    threshold = 0.5 * target_sd

    spans: dict[str, float] = {}
    conflicts: dict[str, bool] = {}
    for key, idx in duplicated.items():
        values = targets[idx]
        if is_classification:
            conflicts[key] = len(np.unique(values)) > 1
            spans[key] = 0.0
        else:
            span = float(values.max() - values.min())
            spans[key] = span
            conflicts[key] = span > threshold

    report = {
        "total_records": int(len(targets)),
        "unique_inchikeys": int(len(groups)),
        "duplicated_structures": int(len(duplicated)),
        "duplicated_records": int(sum(len(idx) for idx in duplicated.values())),
        "target_sd": target_sd,
        "conflict_threshold_half_sd": threshold,
        "max_span": float(max(spans.values())) if spans else 0.0,
        "conflicts_count": int(sum(conflicts.values())),
    }

    # Boundary crossings: duplicated structures that appear in more than one
    # partition of a scaffold manifest.
    leakage: dict[str, Any] = {}
    crossing_counts: list[int] = []
    for seed in seeds:
        split_path = SPLITS_DIR / f"{dataset}_seed_{seed}.json"
        if not split_path.exists():
            raise FileNotFoundError(
                f"missing split manifest {split_path}; prepare the splits or "
                "reduce --seeds"
            )
        with open(split_path, "r", encoding="utf-8") as fh:
            split = json.load(fh)
        memberships = {
            "train": set(split["train"]),
            "val": set(split["val"]),
            "test": set(split["test"]),
        }

        crossing_groups: list[dict[str, Any]] = []
        train_val = train_test = val_test = 0
        for key, idx in duplicated.items():
            present = {
                name: bool(set(idx) & indices) for name, indices in memberships.items()
            }
            if sum(present.values()) < 2:
                continue
            train_val += int(present["train"] and present["val"])
            train_test += int(present["train"] and present["test"])
            val_test += int(present["val"] and present["test"])
            crossing_groups.append(
                {
                    "inchikey": key,
                    "in_train": present["train"],
                    "in_val": present["val"],
                    "in_test": present["test"],
                    "has_conflict": conflicts[key],
                    "targets": [float(targets[i]) for i in idx],
                }
            )

        crossing_counts.append(len(crossing_groups))
        leakage[f"seed_{seed}"] = {
            "crossing_groups_count": len(crossing_groups),
            "train_val": train_val,
            "train_test": train_test,
            "val_test": val_test,
            "crossing_groups": crossing_groups,
        }

    report["split_leakage"] = leakage

    group_rows = [
        {
            "inchikey": key,
            "records_count": len(idx),
            "targets": [float(targets[i]) for i in idx],
            "has_conflict": conflicts[key],
            "smiles_list": " | ".join(smiles[i] for i in idx),
        }
        for key, idx in duplicated.items()
    ]
    group_rows.sort(key=lambda row: row["inchikey"])

    return report, group_rows, {"crossing_counts": crossing_counts}


def audit_ood_shift(dataset: str, meta: dict[str, Any], seeds: list[int]) -> list[dict[str, Any]]:
    # Imported lazily: the duplicate inventory above does not need the full
    # framework import chain, and the audit should stay runnable on a machine
    # where only RDKit and pandas are installed.
    from chemistry_framework.features.pharmacophore import extract_rdkit_2d_descriptors

    from benchmarks._descriptor_contract import select_contract_columns

    df, smiles_col, _ = read_clean_frame(meta)
    smiles = df[smiles_col].astype(str).tolist()
    raw_matrix, raw_names = extract_rdkit_2d_descriptors(smiles)
    matrix, names = select_contract_columns(raw_matrix, raw_names)
    n_descriptors = int(matrix.shape[1])

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        split_path = SPLITS_DIR / f"{dataset}_seed_{seed}.json"
        with open(split_path, "r", encoding="utf-8") as fh:
            split = json.load(fh)
        train_idx = np.asarray(split["train"], dtype=int)
        test_idx = np.asarray(split["test"], dtype=int)

        train = matrix[train_idx, :]
        test = matrix[test_idx, :]
        lo = train.min(axis=0)
        hi = train.max(axis=0)

        outside = (test < lo) | (test > hi)
        n_ood = int(np.any(outside, axis=0).sum())
        affected = int(np.any(outside, axis=1).sum())
        n_test = int(test.shape[0])

        # Magnitude-aware companion to the range count.  A descriptor one unit
        # outside the training range and a descriptor thousands of standard
        # deviations outside it are both "out of range", but only the second
        # can dominate a standardised model input.
        train_nan = np.nan_to_num(train, nan=0.0, posinf=0.0, neginf=0.0)
        test_nan = np.nan_to_num(test, nan=0.0, posinf=0.0, neginf=0.0)
        scale = np.nanstd(train_nan, axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        z = np.abs((test_nan - train_nan.mean(axis=0)) / scale)
        max_z = float(z.max()) if z.size else 0.0
        worst_descriptor = names[int(np.argmax(z.max(axis=0)))]
        n_molecules_above = int(np.any(z > MAGNITUDE_THRESHOLD, axis=1).sum())

        rows.append(
            {
                "split": seed,
                "n_ood_descriptors": n_ood,
                "pct_ood_descriptors": 100.0 * n_ood / n_descriptors,
                "pct_mols_affected": 100.0 * affected / n_test,
                "n_descriptors": n_descriptors,
                "n_test_molecules": n_test,
                "max_abs_standardised_test_value": max_z,
                "worst_magnitude_descriptor": worst_descriptor,
                "magnitude_threshold": MAGNITUDE_THRESHOLD,
                "n_molecules_above_magnitude_threshold": n_molecules_above,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Structure-level dataset audit")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(ALL_DATASETS),
        metavar="NAME",
        help="Endpoints to audit (default: all five manuscript endpoints).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory holding dataset_manifest.json and the cleaned CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for the audit artefacts.",
    )
    add_seed_argument(parser)
    args = parser.parse_args()

    seeds = resolve_seeds(args.seeds)
    data_dir = args.data_dir
    results_dir = args.output_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    reports: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        meta = load_metadata(dataset, data_dir)
        report, group_rows, extra = audit_duplicates(dataset, meta, seeds)
        reports[dataset] = report

        row = {
            "dataset": dataset,
            "total_records": report["total_records"],
            "unique_inchikeys": report["unique_inchikeys"],
            "duplicated_structures": report["duplicated_structures"],
            "duplicated_records": report["duplicated_records"],
            "target_sd": round(report["target_sd"], 4),
            "conflict_threshold_half_sd": round(report["conflict_threshold_half_sd"], 4),
            "max_span": round(report["max_span"], 4),
            "conflicts_count": report["conflicts_count"],
        }
        for seed, count in zip(seeds, extra["crossing_counts"]):
            row[f"seed_{seed}_boundary_crossings"] = count
        summary_rows.append(row)

        print(
            f"[{dataset.upper()}] records={report['total_records']} "
            f"unique={report['unique_inchikeys']} "
            f"duplicated={report['duplicated_structures']} "
            f"({report['duplicated_records']} records) "
            f"conflicts={report['conflicts_count']} "
            f"crossings={extra['crossing_counts']}"
        )

        if dataset == "bbbp":
            groups_path = results_dir / "bbbp_duplicate_groups.csv"
            pd.DataFrame(
                group_rows,
                columns=["inchikey", "records_count", "targets", "has_conflict", "smiles_list"],
            ).to_csv(groups_path, index=False)
            print(f"  wrote {groups_path}")

    report_path = results_dir / "duplicate_audit_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(reports, fh, indent=2)
    print(f"wrote {report_path}")

    summary_path = results_dir / "duplicate_audit_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")

    ood: dict[str, list[dict[str, Any]]] = {}
    for dataset in OOD_DATASETS:
        if dataset not in args.datasets:
            continue
        meta = load_metadata(dataset, data_dir)
        rows = audit_ood_shift(dataset, meta, seeds)
        ood[dataset] = rows
        for row in rows:
            print(
                f"[{dataset.upper()} split {row['split']}] "
                f"OOD descriptors {row['n_ood_descriptors']}/{row['n_descriptors']} "
                f"({row['pct_ood_descriptors']:.1f}%), "
                f"test molecules affected {row['pct_mols_affected']:.1f}%"
            )

    ood_path = results_dir / "ood_shift_all_splits.json"
    with open(ood_path, "w", encoding="utf-8") as fh:
        json.dump(ood, fh, indent=2)
    print(f"wrote {ood_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
