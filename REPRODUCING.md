# Reproducing the manuscript

This document describes how every number, table and figure in

> *Evidence-Guided Molecular Property Modelling with a Configuration-Driven Modular Framework*

can be regenerated from this repository. The archived per-split results are
committed under `benchmarks/results/`, so the statistics, tables and figures can
be checked **without retraining a single model**; retraining is only needed if
you want to extend the analysis (for example to more splits).

---

## 1. Repository layout

| Path | Contents |
| :--- | :--- |
| `chemistry_framework/` | The framework package: features, encoders, training, evaluation, advisor, feature analysis. |
| `benchmarks/` | Runners that produce every reported result, the statistics scripts and the split manifests. |
| `benchmarks/splits/` | 20 Bemis-Murcko scaffold manifests per endpoint (`<dataset>_seed_<k>.json`), plus the duplicate-controlled BBBP manifests under `bbbp_sensitivity/`. |
| `benchmarks/results/` | Archived per-split results (JSON/CSV) for the figures and tables of the manuscript, the recomputed statistics (`paper_statistics.json`) and the generated LaTeX table bodies (`paper_tables/`). |
| `tests/` | Unit tests for the framework package. |

The runners default to **five splits** (`--seeds 1 2 3 4 5`), which is what the
manuscript reports. Every runner accepts `--seeds` so the same experiment can be
repeated on any subset of the prepared manifests.

---

## 2. Environment

| Component | Version used for the manuscript | Notes |
| :--- | :--- | :--- |
| Python | 3.10 – 3.12 | The CI matrix covers these three versions. |
| PyTorch | ≥ 2.0 | `torch>=2.0`; CPU is sufficient, a CUDA GPU only shortens the wall clock. |
| **RDKit** | **2023.09.6 (pinned)** | **Important:** the 210-descriptor block is built from `rdkit.Chem.Descriptors.descList`, whose length depends on the RDKit release. Rerunning the descriptor-based steps with a different RDKit version silently changes the feature set. Pin this version. |
| scikit-learn | ≥ 1.2 | Random Forest baselines, selectors. |
| chemprop | 2.3.1 | Only for the external comparison in Table S6 (`runs/chemprop_v2_*.json`) and the Chemprop BBBP sensitivity runs. |
| PyTDC | ≥ 0.4 | Only for `prepare_moleculenet_data.py`. |

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -e ".[chemistry,test]"
pip install "rdkit==2023.09.6"
pip install chemprop==2.3.1          # optional, external comparison only
```

---

## 3. Data preparation

`prepare_moleculenet_data.py` downloads the MoleculeNet endpoints through
PyTDC, applies the cleaning described in the manuscript, writes
`data/moleculenet/` and generates the 20 scaffold manifests per endpoint:

```bash
python benchmarks/prepare_moleculenet_data.py
```

The manifests committed in `benchmarks/splits/` are already the ones used in
the manuscript; regenerating them is only necessary for a fresh checkout of the
raw data, and the generated manifest identifiers are checked by
`tests/test_datasets.py`-style assertions in the runners (a missing manifest
raises instead of silently skipping a split).

---

## 4. The primary analysis (five splits)

Run from the repository root. Each command writes its output to
`benchmarks/results/`.

| Manuscript artefact | Command | Output |
| :--- | :--- | :--- |
| Tables 2 and 3 (baselines) | `python benchmarks/run_canonical_baselines.py --dataset all --all-seeds` | `baselines_<dataset>_aggregated.json` |
| Tables 4 and S7 (role ablation) | see loop below (`run_modularity_ablation.py` takes one endpoint per invocation) | `modularity_ablation_<dataset>.json` |
| Table 5 and S5 (internal pairs) | statistics only — derived from the two rows above | `paper_statistics.json` |
| Table 6 (routing) and S13 audit | `python benchmarks/run_evidence_routing_experiment.py --dataset all` | `evidence_routing_<dataset>.json`, `routing_checkpoint_audit.csv` |
| Table S3 (recommended vs random/full) | `python benchmarks/run_feature_evidence_study.py --dataset esol` (repeat for `freesolv`, `bbbp`) | `feature_study_<dataset>.json` |
| Table S4 (selectors) | `python benchmarks/run_multivariate_feature_study.py --dataset all` | `multivariate_feature_study_<dataset>.json` |
| Table S6 (external comparison) | `python benchmarks/run_chemprop_v2_benchmarks.py --dataset all` | `chemprop_v2_<dataset>.json` |
| Table S8 (training-seed variance) | `python benchmarks/run_seed_variance_decomposition.py --dataset bbbp --split-seed 1` | `seed_variance_bbbp_split_1.json` |
| Table S9/S10 (case study) | `python benchmarks/run_scorecard_case_study.py` | `scorecard_case_study.json` |
| Table S12 (BBBP sensitivity) | `python benchmarks/run_bbbp_sensitivity_study.py` and `python benchmarks/run_chemprop_bbbp_sensitivity.py` | `bbbp_sensitivity_study.json`, `chemprop_bbbp_sensitivity.json` |
| Table S4b and the range-based OOD counts | `python benchmarks/run_structure_audit.py` | `ood_shift_all_splits.json` |
| Table 5 (structure audit) | `python benchmarks/run_structure_audit.py` | `duplicate_audit_report.json`, `duplicate_audit_summary.csv`, `bbbp_duplicate_groups.csv` |

Loops for the two single-endpoint runners:

```bash
# Linux/macOS
for d in esol freesolv lipophilicity bbbp bace; do
  python benchmarks/run_modularity_ablation.py --dataset "$d"
  python benchmarks/run_multivariate_feature_study.py --dataset "$d"   # if not using --dataset all
done

# PowerShell
foreach ($d in "esol","freesolv","lipophilicity","bbbp","bace") {
  python benchmarks/run_modularity_ablation.py --dataset $d
}
```

`generate_manuscript_tables.py` collects the archived results into
`benchmarks/results/manifest.json` and prints the manuscript tables to the
terminal:

```bash
python benchmarks/generate_manuscript_tables.py
```

---

## 5. Statistics (no retraining required)

```bash
python benchmarks/paper_statistics.py        # -> benchmarks/results/paper_statistics.json
python benchmarks/generate_si_tables.py      # -> benchmarks/results/paper_tables/*.tex
python benchmarks/validate_paper_statistics.py   # regression check against the published values
```

* `paper_statistics.py` recomputes **every** paired comparison in the main text
  and in Additional file 1 from the archived per-split scores. It does not read
  any value from the manuscript.
* The primary interval is the **Nadeau–Bengio corrected resampled *t*
  interval**: because the *k* splits are independent partitions of the same
  finite molecule pool, their training sets overlap, and the naive paired
  interval is anticonservative. The variance is inflated by
  `1/k + n_test/n_train` and the *t* quantile uses `k - 1` degrees of freedom.
  The nominal interval is also reported for continuity with the first version of
  the manuscript.
* Multiplicity is controlled with **Holm–Bonferroni** within the family of
  comparisons printed in one table (or one dataset block of a table). The family
  sizes are stated in each caption; the adjusted `p`-values are in the
  `holm_p` field and in the captions of the generated tables.
* `validate_paper_statistics.py` recomputes the 30 published values from the
  transcription of Additional file 1 and verifies that the archived results
  reproduce them exactly. It is a check on the *archived five-split* results: if
  you overwrite the results with a longer run, this script is expected to report
  differences — that is the signal that the manuscript numbers must be updated
  too.

---

## 6. Extending to more splits

```bash
python benchmarks/run_canonical_baselines.py --dataset all --seeds 1 2 3 4 5 6 7 8 9 10
for d in esol freesolv lipophilicity bbbp bace; do
  python benchmarks/run_modularity_ablation.py  --dataset "$d" --seeds 1 2 3 4 5 6 7 8 9 10
  python benchmarks/run_multivariate_feature_study.py --dataset "$d" --seeds 1 2 3 4 5 6 7 8 9 10
done
python benchmarks/run_evidence_routing_experiment.py --dataset all --seeds 1 2 3 4 5 6 7 8 9 10
python benchmarks/run_chemprop_v2_benchmarks.py --dataset all --seeds 1 2 3 4 5 6 7 8 9 10
python benchmarks/paper_statistics.py
python benchmarks/generate_si_tables.py
```

`paper_statistics.py` determines the number of splits from the archived files
(`n_seeds`, the `seeds` field written by the runners, or the length of the
per-split lists), so the corrected intervals, the *t* degrees of freedom and the
Holm adjustment all follow automatically. Prepared manifests exist for seeds
1–20; `--seeds` rejects any index without a manifest instead of skipping it.

Wall-clock guidance: on a single mid-range CPU, the five-split baseline suite
(`run_canonical_baselines.py --dataset all --all-seeds`) takes on the order of
a day; the routing experiment is the most expensive runner because each split
trains seven models and selects the budget `K*` on the validation partition.

---

## 7. What reproduces exactly, and what does not

* **Exact:** everything computed from the archived per-split scores — the
  paired comparisons, the corrected and nominal intervals, the *p*-values,
  the Holm adjustment, the minimum detectable differences, and the LaTeX bodies
  of the tables changed in this version.
* **Exact (structure audit):** `benchmarks/run_structure_audit.py` regenerates
  `duplicate_audit_report.json`, `duplicate_audit_summary.csv` and
  `bbbp_duplicate_groups.csv` field-for-field from the raw MoleculeNet releases
  — including the per-split boundary-crossing inventory of duplicated InChIKeys
  (BBBP crossings: 1, 0, 1, 1, 2 records for seeds 1–5).
* **Version-sensitive:** the RDKit 2D descriptor block (pin RDKit 2023.09.6),
  and anything downstream of it (OOD descriptor counts, scorecard evidence
  classes, recommender selections). Re-running the audit with RDKit 2026.09.1
  (217 descriptors in `descList` instead of 210) changes the denominator of the
  OOD percentages and, on 6 of the 15 archived endpoint/split combinations,
  adds 1–3 out-of-range descriptors; the percentage of affected test molecules
  is unchanged on 14 of 15 splits. Reported OOD counts must therefore always be
  quoted together with the RDKit version.
* **Non-deterministic:** the neural models are seeded per split
  (`torch.manual_seed(seed)`, `L.seed_everything(seed, workers=True)`), but GPU
  kernels, cuDNN autotuning and thread scheduling can shift individual runs by
  fractions of a metric point. Expect small drift in retrained numbers, not
  exact bit-for-bit equality. The Random Forest and descriptor-MLP baselines are
  deterministic given the split.
* **Not part of this release:** the `caco2_study/` results are a development
  artefact from earlier work on the same package; they are kept for provenance
  and are not used in the manuscript.

## 8. Issues

Please open an issue at <https://github.com/soratori1025/Framework-Chemistry/issues>
for any mismatch between this repository and the manuscript.
