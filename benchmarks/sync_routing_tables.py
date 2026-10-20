"""Synchronise the routing tables (manuscript Table 7 and Additional file 1,
Table S7) with the archived ten-split routing results, and report the counts
used in the checkpoint audit sentence.

Run after ``rebuild_routing_audit.py`` has merged the per-dataset audit files.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
MAN = ROOT / "manuscript" / "sections" / "results.tex"
SI = ROOT / "manuscript" / "supplementary" / "additional_file_1.tex"

ROUTES = [
    ("residual_only", "Graph only"),
    ("direct_hybrid", "All 210"),
    ("hybrid_rec", "Recommender"),
    ("hybrid_spearman", "Spearman"),
    ("hybrid_mi", "MI"),
    ("hybrid_random", "Random"),
]
ORDER = ["esol", "freesolv", "lipophilicity", "bbbp", "bace"]
NAME = {"esol": "ESOL", "freesolv": "FreeSolv", "lipophilicity": "Lipophilicity",
        "bbbp": "BBBP", "bace": "BACE"}
CLASSIFICATION = {"bbbp", "bace"}


def rows() -> list[str]:
    out = []
    for ds in ORDER:
        j = json.loads((RESULTS / f"evidence_routing_{ds}.json").read_text())
        n = len(j["per_split_results"])
        assert n == 10, f"{ds}: {n} splits"
        cells, means = [], []
        for key, _ in ROUTES:
            s = j["summary"][key]
            means.append(float(s["mean"]))
            cells.append(f"${s['mean']:.3f}\\pm{s['std']:.3f}$")
        best = (max(means) if ds in CLASSIFICATION else min(means))
        idx = means.index(best)
        s = j["summary"][ROUTES[idx][0]]
        cells[idx] = f"$\\mathbf{{{s['mean']:.3f}\\pm{s['std']:.3f}}}$"
        metric = "AUROC $\\uparrow$" if ds in CLASSIFICATION else "RMSE $\\downarrow$"
        out.append(f"{NAME[ds]} & {metric} & " + " & ".join(cells) + r" \\")
    return out


def patch(path: Path, lines: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    start = text.index("Dataset & Metric & Graph only & All 210")
    head = text.index("\n", start) + 1
    end = text.index("\\bottomrule", head)
    body = "\n".join(lines) + "\n"
    path.write_text(text[:head] + body + text[end:], encoding="utf-8")
    print("patched", path)


def audit_count() -> int:
    with (RESULTS / "routing_checkpoint_audit.csv").open(encoding="utf-8-sig") as fh:
        return sum(1 for _ in csv.DictReader(fh))


if __name__ == "__main__":
    new_rows = rows()
    print("\n".join(new_rows))
    patch(MAN, new_rows)
    patch(SI, new_rows)
    n = audit_count()
    print("checkpoint audit rows:", n)

    for p in (MAN, SI):
        t = p.read_text(encoding="utf-8")
        t, k = re.subn(r"\b\d+ model runs\b", f"{n} model runs", t)
        p.write_text(t, encoding="utf-8")
        print(f"  {k} audit-count occurrence(s) updated in {p.name}")
    print("done")
