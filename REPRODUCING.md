# Reproducing the manuscript

This document describes how every number, table and figure in

> *Evidence before prediction: a configuration-driven framework for testing how
> molecular descriptors shape property models*

can be regenerated from this repository. The archived per-split results are
committed under `benchmarks/results/`, so the statistics, tables and figures can
be checked **without retraining a single model**; retraining is only needed if
you want to extend the analysis (for example to more splits).

The manuscript reports **ten scaffold splits** (seeds 1–10). The archived
results committed in `benchmarks/results/` are those ten splits, and every
table and figure in the manuscript is generated from them. The archived files
are the authoritative source: if a table and a result file disagree, the result
file is correct and the table should be regenerated with the commands below.

---

## 1. Repository layout

| Path | Contents |
| :--- | :--- |
| `chemistry_framework/` | The framework package: features, encoders, training, evaluation, advisor, feature analysis. |
| `benchmarks/` | Runners that produce every reported result, the statistics scripts and the split manifests. |
| `benchmarks/splits/` | 20 Bemis-Murcko scaffold manifests per endpoint (`<dataset>_seed_<k>.json`), plus the duplicate-controlled BBBP manifests under `bbbp_sensitivity/`. |
| `benchmarks/results/` | Archived per-split results (JSON/CSV) for the figures and tables of the manuscript, the recomputed statistics (`paper_statistics.json`) and the generated LaTeX table bodies (`paper_tables/`). |
| `tests/` | Unit tests for the framework package. |

The runners default to **ten splits** (`--seeds 1 2 3 4 5 6 7 8 9 10`), which is
what the manuscript reports. Every runner accepts `--seeds` so the same
experiment can be repeated on any subset of the prepared manifests.

---

## 2. Environments

Two Python environments are used. The framework runners and every statistics
script run in the first; the two Chemprop runners need their own environment
because Chemprop 2.3.1 depends on a newer RDKit than the one the published
descriptor block was built with.

| Component | Version used for the manuscript | Notes |
| :--- | :--- | :--- |
| Python | 3.11 | The CI matrix covers 3.10–3.12. |
| PyTorch | ≥ 2.0, CPU build | CPU is sufficient; a CUDA GPU only shortens the wall clock. |
| **RDKit (framework env)** | **2023.09.6 (pinned)** | The 210-descriptor block is built from `rdkit.Chem.Descriptors.descList`, whose length depends on the RDKit release. `benchmarks/_descriptor_contract.py` pins the 210 column *names* and raises if one is missing, but the archived numbers were produced with this version. |
| scikit-learn | ≥ 1.2 | Random Forest baselines, selectors. |
| chemprop (second env) | 2.3.1 | External comparison and the Chemprop BBBP sensitivity runs. Its own RDKit (`≥ 2026.03`) is fine: the runners reindex the descriptor block to the 210 published columns. |
| PyTDC | ≥ 0.4 | Only for `prepare_moleculenet_data.py`. |

```bash
# framework environment
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -e ".[chemistry,test]"
pip install "rdkit==2023.09.6"

# chemprop environment (separate; external comparison only)
python -m venv .venv-chemprop
.venv-chemprop/bin/pip install chemprop==2.3.1
```

Every Chemprop command below is run with the interpreter of the second
environment (`.../python benchmarks/run_chemprop_v2_benchmarks.py ...`); the
framework runners use the first.

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
raw data, and the generated manifest identifiers are checked by the runners (a
missing manifest raises instead of silently skipping a split).

---

## 4. The primary analysis (ten splits)

Run from the repository root. Each command writes its output to
`benchmarks/results/`.

| Manuscript artefact | Command | Output |
| :--- | :--- | :--- |
| Table 3/4 (baselines) | `python benchmarks/run_canonical_baselines.py --dataset all` | `baselines_<dataset>_aggregated.json` |
| Table 8 (role ablation) | `run_modularity_ablation.py`, one endpoint per invocation (loop below) | `modularity_ablation_<dataset>.json` |
| Table 3/4 (external comparator) | `python benchmarks/run_chemprop_v2_benchmarks.py --dataset all` (chemprop env) | `chemprop_v2_<dataset>.json` |
| Table 5 and S4/S4b (selectors) | `python benchmarks/run_multivariate_feature_study.py --dataset all` | `multivariate_feature_study_<dataset>.json` |
| Table S3 (recommended vs random/full) | `python benchmarks/run_feature_evidence_study.py --dataset esol` (repeat for `freesolv`, `bbbp`) | `feature_study_<dataset>.json` |
| Table 7, S7/S7b (routing) and S13 (audit) | `python benchmarks/run_evidence_routing_experiment.py --dataset all` then `python benchmarks/rebuild_routing_audit.py` | `evidence_routing_<dataset>.json`, `routing_checkpoint_audit_<dataset>.csv`, `routing_checkpoint_audit.csv` |
| Table 6 and S2 (structure audit and range shift) | `python benchmarks/run_structure_audit.py` | `duplicate_audit_report.json`, `duplicate_audit_summary.csv`, `bbbp_duplicate_groups.csv`, `ood_shift_all_splits.json` |
| Table S8 (training-seed variance) | `python benchmarks/run_seed_variance_decomposition.py --dataset bbbp --split-seed 1` | `seed_variance_bbbp_split_1.json` |
| Table S11 (Caco-2 case study) | `python benchmarks/run_caco2_study.py` | `caco2_study/` results |
| Tables S12/S12b (BBBP sensitivity) | `python benchmarks/run_bbbp_sensitivity_study.py` and `python benchmarks/run_chemprop_bbbp_sensitivity.py` (chemprop env) | `bbbp_sensitivity_study.json`, `chemprop_bbbp_sensitivity.json` |
| Figures 1–4 | `python benchmarks/make_manuscript_figures.py` | `manuscript/figs/fig{1..4}_*.pdf` |

Everything else — Tables S1a/S1b, S2, S5/S5b/S6, S9/S10 — is computed from the
files above by the statistics and reporting tools in section 5.

Loops for the single-endpoint runners:

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

## 5. Statistics and manuscript numbers (no retraining required)

```bash
python benchmarks/paper_statistics.py            # -> benchmarks/results/paper_statistics.json
python benchmarks/generate_si_tables.py          # -> benchmarks/results/paper_tables/*.tex
python benchmarks/validate_paper_statistics.py   # consistency checks on that file
python benchmarks/report_manuscript_numbers.py   # every number quoted in the text
python benchmarks/report_manuscript_numbers.py --only latex   # table bodies
python benchmarks/sync_routing_tables.py         # rewrite the routing rows in manuscript/ and supplementary/
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
  the manuscript. The minimum detectable difference at 80 % power is computed
  from the corrected standard error.
* Multiplicity is controlled with **Holm–Bonferroni** within the family of
  comparisons printed in one table (or one dataset block of a table). The family
  sizes are stated in each caption; the adjusted `p`-values are in the
  `holm_p` field and in the captions of the generated tables.
* `generate_si_tables.py` renders the Additional-file tables whose content is
  fully determined by the statistics (`tab_int`, `tab_ext`, `tab_mde`,
  `tab_rec_random`, `tab_selector_stats`, `tab_routing_stats`,
  `tab_sensitivity_stats`). The remaining descriptive tables are transcribed
  from the numbers that `report_manuscript_numbers.py` prints.
* `validate_paper_statistics.py` has two modes. `--mode published` checks the 30
  values transcribed from the published Additional file 1 of the current
  ten-split study; the default `auto`/`consistency` mode recomputes every
  derived quantity in the statistics file — paired means and SDs from the stored
  per-split scores, both intervals, both *p*-values, the minimum detectable
  difference and the Holm adjustment — and fails if any stored field disagrees.
* `compare_split_runs.py` compares a rerun against the published snapshot,
  value by value, for the seeds both runs share:

```bash
python benchmarks/compare_split_runs.py \
    --archive <earlier snapshot> --new benchmarks/results --seeds 1 2 3 4 5
```

---

## 6. Changing the number of splits

```bash
python benchmarks/run_canonical_baselines.py --dataset all --seeds 1 2 3 4 5 6 7 8 9 10
for d in esol freesolv lipophilicity bbbp bace; do
  python benchmarks/run_modularity_ablation.py --dataset "$d" --seeds 1 2 3 4 5 6 7 8 9 10
done
python benchmarks/paper_statistics.py
```

`paper_statistics.py` determines the number of splits from the archived files
(`seeds` field written by the runners, or the length of the per-split lists), so
the corrected intervals, the *t* degrees of freedom, the Holm adjustment and the
minimum detectable differences all follow automatically. Prepared manifests
exist for seeds 1–20; `--seeds` rejects any index without a manifest instead of
skipping it.

Wall-clock guidance: on a single mid-range CPU (16 cores, no GPU) the ten-split
run of all framework experiments takes about two working days; the routing experiment is
the most expensive runner because each split trains several models and selects
the budget `K*` on the validation partition, and the Chemprop runs are slower
still per model. The runners are single-process; running one group per endpoint
in parallel is the easiest speed-up.

---

## 7. What reproduces exactly, and what does not

* **Exact:** everything computed from the archived per-split scores — the
  paired comparisons, the corrected and nominal intervals, the *p*-values, the
  Holm adjustment, the minimum detectable differences, the selector ranks and
  the LaTeX bodies of the generated tables.
* **Exact (audits):** `benchmarks/run_structure_audit.py` regenerates the
  duplicate/conflict inventory and the range-based OOD counts field-for-field
  (BBBP: 64 duplicated structures in 132 records, 11 conflicting; ESOL: 11
  duplicated structures below the conflict threshold), and
  `benchmarks/rebuild_routing_audit.py` rebuilds `routing_checkpoint_audit.csv`
  from the routing result files, so the released log can never mix rows from
  superseded runs.
* **Version-sensitive:** the RDKit 2D descriptor block. The contract in
  `benchmarks/_descriptor_contract.py` pins the 210 published columns by name,
  so a rerun with a newer RDKit (2026.09.1 exposes 217 entries) keeps the same
  feature block; without the contract, the OOD percentages would change
  denominator and 1–3 descriptors would flip to out-of-range on 6 of the 15
  archived endpoint/split combinations. List-valued and count-based descriptors
  that RDKit changed between releases are covered by the same contract.
* **Non-deterministic:** the neural models are seeded per split
  (`torch.manual_seed(seed)`, `L.seed_everything(seed, workers=True)`), but CPU
  and GPU kernels, thread scheduling and library versions can shift individual
  runs by fractions of a metric point. Expect small drift in retrained numbers,
  not exact bit-for-bit equality; `compare_split_runs.py` quantifies it against
  the published snapshot. The Random Forest and descriptor-MLP baselines are
  deterministic given the split, and the Chemprop runs use fixed seeds.
* **Not part of this release:** the `caco2_study/` outputs are a development
  artefact from earlier work on the same package; they are kept for provenance
  and are not used in the manuscript.

## 8. Issues

Please open an issue at <https://github.com/soratori1025/Framework-Chemistry/issues>
for any mismatch between this repository and the manuscript.
