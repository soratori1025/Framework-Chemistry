from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "moleculenet"
SPLITS_DIR = REPO_ROOT / "benchmarks" / "splits"

SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
FRACTIONS = (0.8, 0.1, 0.1)

DATASET_CONFIGS = {
    "bbbp": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
        "filename": "bbbp.csv",
        "smiles_col": "smiles",
        "target_cols": ["p_np"],
        "task_type": "classification",
        "split_method": "scaffold",
    },
    "bace": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
        "filename": "bace.csv",
        "smiles_col": "mol",
        "target_cols": ["Class"],
        "task_type": "classification",
        "split_method": "scaffold",
    },
    "hiv": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
        "filename": "hiv.csv",
        "smiles_col": "smiles",
        "target_cols": ["HIV_active"],
        "task_type": "classification",
        "split_method": "scaffold",
    },
    "clintox": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
        "filename": "clintox.csv.gz",
        "smiles_col": "smiles",
        "target_cols": ["FDA_APPROVED", "CT_TOX"],
        "task_type": "multitask_classification",
        "split_method": "random",
    },
    "tox21": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
        "filename": "tox21.csv.gz",
        "smiles_col": "smiles",
        "target_cols": [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
            "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
        ],
        "task_type": "multitask_classification",
        "split_method": "random",
    },
    "esol": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
        "filename": "esol.csv",
        "smiles_col": "smiles",
        "target_cols": ["measured log solubility in mols per litre"],
        "task_type": "regression",
        "split_method": "scaffold",
    },
    "freesolv": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv",
        "filename": "freesolv.csv",
        "smiles_col": "smiles",
        "target_cols": ["expt"],
        "task_type": "regression",
        "split_method": "random",
    },
    "lipophilicity": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
        "filename": "lipophilicity.csv",
        "smiles_col": "smiles",
        "target_cols": ["exp"],
        "task_type": "regression",
        "split_method": "scaffold",
    },
}

def build_scaffold_groups(smiles: list[str]) -> list[list[int]]:
    groups: dict[str, list[int]] = {}
    for idx, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True)
            key = scaffold or f"acyclic_{Chem.MolToSmiles(mol)}"
        except Exception:
            key = f"raw_{smi}"
        groups.setdefault(key, []).append(idx)
    return list(groups.values())

def scaffold_split_from_groups(group_values: list[list[int]], n_total: int, fractions: tuple[float, float, float], seed: int) -> tuple[list[int], list[int], list[int]]:
    rng = np.random.default_rng(seed)
    shuffled_groups = [g for g in group_values]
    rng.shuffle(shuffled_groups)
    shuffled_groups.sort(key=len, reverse=True)

    targets = np.asarray(fractions) * n_total
    bins: list[list[int]] = [[], [], []]
    for group in shuffled_groups:
        ratios = [len(bins[i]) / max(targets[i], 1.0) for i in range(3)]
        selected = int(np.argmin(ratios))
        bins[selected].extend(group)

    return bins[0], bins[1], bins[2]

def random_split(n_total: int, fractions: tuple[float, float, float], seed: int) -> tuple[list[int], list[int], list[int]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n_total, dtype=int)
    rng.shuffle(indices)
    n_train = int(fractions[0] * n_total)
    n_val = int(fractions[1] * n_total)
    train_idx = indices[:n_train].tolist()
    val_idx = indices[n_train:n_train + n_val].tolist()
    test_idx = indices[n_train + n_val:].tolist()
    return train_idx, val_idx, test_idx

def prepare_dataset(name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = DATA_DIR / cfg["filename"]
    if not csv_path.exists():
        print(f"📥 Downloading {name.upper()} from {cfg['url']}...")
        df = pd.read_csv(cfg["url"])
        df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path)

    smiles_col = cfg["smiles_col"]
    if smiles_col not in df.columns:
        possible = [c for c in df.columns if "smiles" in c.lower() or c.lower() == "mol"]
        if not possible:
            raise ValueError(
                f"{name}: no SMILES column found. Expected {smiles_col!r}; "
                f"available columns are {sorted(df.columns)}. "
                "Refusing to emit a descriptor dataset without structures, "
                "because the resulting CSV would be silently unusable."
            )
        smiles_col = possible[0]

    # The target columns must survive into the cleaned frame; a typo in
    # DATASET_CONFIGS would otherwise only surface much later as a KeyError in
    # the loader, or worse, as a silently wrong target.
    missing_targets = [c for c in cfg["target_cols"] if c not in df.columns]
    if missing_targets:
        raise ValueError(
            f"{name}: target column(s) {missing_targets} absent from {csv_path.name}; "
            f"available columns are {sorted(df.columns)}"
        )

    valid_mask = []
    clean_smiles = []
    for val in df[smiles_col].astype(str):
        mol = Chem.MolFromSmiles(val)
        if mol is not None:
            valid_mask.append(True)
            clean_smiles.append(Chem.MolToSmiles(mol))
        else:
            valid_mask.append(False)
            clean_smiles.append(None)

    df_clean = df[valid_mask].copy().reset_index(drop=True)
    clean_smiles = [s for s in clean_smiles if s is not None]
    df_clean[smiles_col] = clean_smiles

    clean_file = DATA_DIR / f"{name}_clean.csv"
    df_clean.to_csv(clean_file, index=False)
    n_samples = len(df_clean)
    print(f"✓ {name.upper()}: {n_samples} valid molecules (dropped {len(df) - n_samples} invalid)")

    split_info = {}
    scaffold_groups = build_scaffold_groups(clean_smiles) if cfg["split_method"] == "scaffold" else []
    for seed in SEEDS:
        if cfg["split_method"] == "scaffold":
            train_idx, val_idx, test_idx = scaffold_split_from_groups(scaffold_groups, n_samples, FRACTIONS, seed=seed)
        else:
            train_idx, val_idx, test_idx = random_split(n_samples, FRACTIONS, seed=seed)

        split_dict = {
            "dataset": name,
            "seed": seed,
            "split_method": cfg["split_method"],
            "fractions": FRACTIONS,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_idx),
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        }
        split_path = SPLITS_DIR / f"{name}_seed_{seed}.json"
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split_dict, f, indent=2)
        split_info[f"seed_{seed}"] = {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx),
            "split_file": str(split_path.relative_to(REPO_ROOT)),
        }

    return {
        "dataset": name,
        "n_samples": n_samples,
        "smiles_col": smiles_col,
        "targets": cfg["target_cols"],
        "task_type": cfg["task_type"],
        "split_method": cfg["split_method"],
        "clean_csv": str(clean_file.relative_to(REPO_ROOT)),
        "splits": split_info,
    }

def main():
    print("=" * 70)
    print("MoleculeNet Benchmark Data & Canonical Split Preparation (80/10/10)")
    print("=" * 70)
    manifest = {}
    for name, cfg in DATASET_CONFIGS.items():
        try:
            info = prepare_dataset(name, cfg)
            manifest[name] = info
        except Exception as e:
            print(f"❌ Error preparing {name}: {e}")

    summary_path = DATA_DIR / "dataset_manifest.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n✅ All datasets and {len(SEEDS)}-seed split indices generated successfully!")
    print(f"Manifest saved to: {summary_path}")

if __name__ == "__main__":
    main()
