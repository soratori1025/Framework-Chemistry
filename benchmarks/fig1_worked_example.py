"""Select one real ESOL molecule and derive every number Fig. 1 shows for it.

The figure is a worked example, so each value in it must come from the same
data and the same split as the benchmark.  This script reads the committed
MoleculeNet data and the split manifest, re-runs the framework recommendation
rule on the training partition, and prints:

* the partition of the chosen molecule and its measured value;
* support, |Spearman rho| and rank for a few descriptors, with the molecule's
  own value and the training range;
* the recommended budget K* for the first split (from the routing result);
* test RMSE of the full-descriptor hybrid and of the compact route,
  and the prior estimate for the molecule.

Run from the repository root with the framework environment (the published
results used RDKit 2023.09.6):

    python benchmarks/fig1_worked_example.py

Add ``--write`` to store the numbers as ``benchmarks/results/fig1_worked_example.json``
so the figure script can read them without recomputing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "benchmarks" / "results"
DATA = REPO_ROOT / "data" / "moleculenet" / "esol_clean.csv"
SPLIT = REPO_ROOT / "benchmarks" / "splits" / "esol_seed_1.json"

MOLECULE = "phenacetin"
TARGET = "measured log solubility in mols per litre"
PRIOR_DESCRIPTORS = ("MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds")
SHOW = ("MolLogP", "MolWt", "NumRotatableBonds", "TPSA", "NumHDonors")
SUPPORTED_RHO = 0.15     # threshold used by the case-study scorecard
REDUNDANCY_R = 0.85      # framework recommender and case study


def descriptor_values(smiles: list[str]) -> tuple[np.ndarray, list[str]]:
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    mols = [Chem.MolFromSmiles(s) for s in smiles]
    names = [name for name, _ in Descriptors.descList]
    fns = [fn for _, fn in Descriptors.descList]
    matrix = np.zeros((len(mols), len(names)))
    for i, mol in enumerate(mols):
        if mol is None:
            continue
        for j, fn in enumerate(fns):
            try:
                value = float(fn(mol))
                matrix[i, j] = value if np.isfinite(value) else 0.0
            except Exception:
                matrix[i, j] = 0.0
    return matrix, names


def recommend(X_tr: np.ndarray, y_tr: np.ndarray, k: int) -> tuple[list[int], np.ndarray]:
    """The framework rule: |Spearman| ranking, greedy |Pearson| <= 0.85 pruning."""
    scores = np.zeros(X_tr.shape[1])
    for j in range(X_tr.shape[1]):
        if np.std(X_tr[:, j]) < 1e-6:
            continue
        r, _ = stats.spearmanr(X_tr[:, j], y_tr)
        scores[j] = abs(r) if np.isfinite(r) else 0.0
    selected: list[int] = []
    for j in np.argsort(scores)[::-1]:
        if scores[j] < 1e-4:
            break
        if any(kept_collinear(X_tr, j, k_)[1] for k_ in selected):
            continue
        selected.append(int(j))
        if len(selected) == k:
            break
    return selected, scores


def kept_collinear(X_tr: np.ndarray, j: int, reference: int) -> tuple[float, bool]:
    """|Pearson r| between two columns; constant columns are never flagged."""
    if np.std(X_tr[:, j]) < 1e-12 or np.std(X_tr[:, reference]) < 1e-12:
        return 0.0, False
    r = abs(float(stats.pearsonr(X_tr[:, j], X_tr[:, reference])[0]))
    return r, r > REDUNDANCY_R


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="store the numbers as JSON")
    args = parser.parse_args()

    df = pd.read_csv(DATA)
    y = df[TARGET].to_numpy(dtype=float)
    X, names = descriptor_values(df["smiles"].tolist())
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    split = json.loads(SPLIT.read_text())
    tr, va, te = (np.array(split["train"]), np.array(split["val"]), np.array(split["test"]))

    row = int(np.where(df["Compound ID"].str.strip().str.lower() == MOLECULE)[0][0])
    partition = "train" if row in set(tr.tolist()) else ("test" if row in set(te.tolist()) else "validation")

    k_candidates = (10, 20, 50)
    selected, scores = recommend(X[tr], y[tr], max(k_candidates))
    order = {int(j): r + 1 for r, j in enumerate(np.argsort(scores)[::-1])}

    print(f"descriptors: {len(names)}   train {len(tr)}   val {len(va)}   test {len(te)}")
    print(f"\n{MOLECULE}: row {row}, measured log10 S = {y[row]:.2f}, {partition} partition (split 1)")

    table = []
    print(f"\n{'descriptor':<20}{'|rho|':>7}{'rank':>6}{'support':>9}{'value':>10}   training range")
    for j, name in enumerate(names):
        if name not in SHOW:
            continue
        lo, hi = X[tr, j].min(), X[tr, j].max()
        inside = lo <= X[row, j] <= hi
        # the scorecard's ordered rule, applied in its own order
        if np.count_nonzero(X[tr, j]) == 0:
            evidence = "unsupported"
        elif any(
            kept_collinear(X[tr], j, k_)[1] and scores[k_] > scores[j]
            for k_ in range(len(names)) if k_ != j
        ):
            evidence = "redundant"
        elif scores[j] >= SUPPORTED_RHO:
            evidence = "supported"
        else:
            evidence = "weak"
        table.append({"name": name, "abs_rho": round(float(scores[j]), 2), "rank": order[j],
                      "support": int(np.count_nonzero(X[tr, j])), "value": round(float(X[row, j]), 2),
                      "range": [round(float(lo), 2), round(float(hi), 2)], "inside": bool(inside),
                      "evidence": evidence, "in_route": name in [names[j2] for j2 in selected]})
        print(f"{name:<20}{scores[j]:7.2f}{order[j]:6d}{int(np.count_nonzero(X[tr, j])):9d}{X[row, j]:10.2f}   "
              f"[{lo:.2f}, {hi:.2f}]  {evidence}")

    route = [names[j] for j in selected]

    out_of_range = sum(
        bool(np.any((X[te, j] < X[tr, j].min()) | (X[te, j] > X[tr, j].max()))) for j in range(len(names))
    )

    routing = json.loads((RESULTS / "evidence_routing_esol.json").read_text())
    s1 = next(r for r in routing["per_split_results"] if r["split"] == 1)
    k_star = int(s1["best_k"])

    # prior estimate for the molecule: ridge on six descriptors, fitted on train only
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    p_idx = [names.index(n) for n in PRIOR_DESCRIPTORS]
    scaler = StandardScaler().fit(X[tr][:, p_idx])
    prior = Ridge(alpha=1.0).fit(scaler.transform(X[tr][:, p_idx]), y[tr])
    prior_value = float(prior.predict(scaler.transform(X[[row]][:, p_idx]))[0])

    result = {
        "molecule": MOLECULE,
        "smiles": "CCOc1ccc(NC(C)=O)cc1",
        "inchikey": "CPJSUEIXXCENMM-UHFFFAOYSA-N",
        "dataset": "ESOL",
        "split_seed": 1,
        "partition": partition,
        "measured": round(float(y[row]), 2),
        "prior_estimate": round(prior_value, 2),
        "descriptors_in_pool": len(names),
        "train_size": int(len(tr)),
        "test_size": int(len(te)),
        "descriptors_outside_train_range_test": out_of_range,
        "k_candidates": list(k_candidates),
        "k_star": k_star,
        "route": route[:k_star],
        "k10_route": route[:10],
        "table": table,
        "test_rmse_full_descriptors": round(float(s1["direct_hybrid"]["rmse"]), 3),
        "test_rmse_compact_route": round(float(s1["hybrid_rec"]["rmse"]), 3),
        "test_rmse_graph_only": round(float(s1["residual_only"]["rmse"]), 3),
    }

    print(f"\nrecommended route at K*={k_star}: {', '.join(result['route'])}")
    print(f"descriptors with test values outside the training range: {out_of_range}/{len(names)}")
    print(f"prior estimate for {MOLECULE}: {prior_value:.2f} (measured {y[row]:.2f})")
    print(f"test RMSE on split 1: graph only {result['test_rmse_graph_only']}, "
          f"all descriptors {result['test_rmse_full_descriptors']}, "
          f"compact route {result['test_rmse_compact_route']}")

    if args.write:
        path = RESULTS / "fig1_worked_example.json"
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
