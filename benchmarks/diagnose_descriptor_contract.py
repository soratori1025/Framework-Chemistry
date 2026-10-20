"""Check that the RDKit 2D descriptor contract is stable across environments.

The framework benchmark and the Chemprop benchmark were executed in different
conda environments (RDKit 2023.09.6 and a newer RDKit), so this script exports
the contracted descriptor matrix for each dataset and compares two dumps
numerically.  Any drift would weaken the "same 210 descriptors" claim.

Usage (two environments, same script):

    python benchmarks/diagnose_descriptor_contract.py --dump out_a.npz
    python benchmarks/diagnose_descriptor_contract.py --compare out_a.npz out_b.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chemistry_framework.features.pharmacophore import extract_rdkit_2d_descriptors
from benchmarks._descriptor_contract import select_contract_columns

DATA_DIR = REPO_ROOT / "data" / "moleculenet"
DATASETS = ["esol", "freesolv", "lipophilicity", "bbbp", "bace"]


def build() -> dict[str, np.ndarray]:
    import rdkit
    payload: dict[str, np.ndarray] = {"__rdkit__": np.array([rdkit.__version__])}
    manifest = json.loads((DATA_DIR / "dataset_manifest.json").read_text())
    for name in DATASETS:
        meta = manifest[name]
        df = pd.read_csv(REPO_ROOT / meta["clean_csv"])
        smiles = df[meta["smiles_col"]].astype(str).tolist()
        matrix, names = select_contract_columns(*extract_rdkit_2d_descriptors(smiles))
        payload[name] = matrix
        payload[f"{name}__names"] = np.array(names)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", type=str)
    parser.add_argument("--compare", nargs=2, type=str)
    args = parser.parse_args()

    if args.dump:
        payload = build()
        np.savez_compressed(args.dump, **payload)
        print(f"dumped {args.dump}: rdkit {payload['__rdkit__'][0]}")
        return

    if args.compare:
        a = np.load(args.compare[0], allow_pickle=True)
        b = np.load(args.compare[1], allow_pickle=True)
        print(f"A rdkit {a['__rdkit__'][0]}  vs  B rdkit {b['__rdkit__'][0]}")
        for name in DATASETS:
            assert list(a[f"{name}__names"]) == list(b[f"{name}__names"]), f"feature names differ for {name}"
            x, y = a[name], b[name]
            scale = np.maximum(np.abs(x), np.abs(y)) + 1e-12
            rel = np.abs(x - y) / scale
            finite = np.isfinite(x) & np.isfinite(y)
            n_diff = int(np.sum(finite & (rel > 1e-9)))
            print(f"  {name:14s} cells={x.shape} differing(rel>1e-9)={n_diff} "
                  f"max_rel={np.max(rel[finite]) if finite.any() else 0:.3e}")
        return

    parser.error("pass --dump or --compare")


if __name__ == "__main__":
    main()
