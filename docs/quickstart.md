# Quick start

`chemistry-framework` turns one plain-text configuration file into a trained,
recorded experiment. This page is the five-minute path; the manuscript benchmark
suite is described in `../REPRODUCING.md`, and every configuration key is listed
in `configuration.md`.

## 1. Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -e ".[chemistry,test]"
pip install "rdkit==2023.09.6"        # the version behind the published descriptor block
```

`requirements-lock.txt` freezes the environment used for the manuscript;
`environment.yml` is the conda equivalent.

## 2. Look at the data before modelling it

```bash
chemistry-framework inspect my_table.csv --smiles-column smiles --target logS --output-dir reports/
```

`inspect` reports SMILES validity, repeated SMILES, scaffold and fingerprint
diversity, target skew and outliers, class balance, and a dataset health score.
It writes `reports/my_table-health-report.json` and
`reports/my_table-advisory-report.md`, and can draft a starting configuration
with `--generate-config my_experiment.yaml`.

## 3. Declare the experiment

```yaml
# my_experiment.yaml
data:
  path: my_table.csv
  smiles_column: smiles
  targets: [logS]

features:
  physchem:
    type: rdkit_descriptors
    names: [MolWt, MolLogP, TPSA, NumHDonors, NumHAcceptors, NumRotatableBonds]
  rdkit2d:
    type: rdkit_descriptors
    families: [all]           # the 210-descriptor block

priors:
  logS_prior: {use: [physchem], fit: ridge}

routes:
  logS: [rdkit2d]             # a route maps feature blocks onto a task

tasks:
  logS: {target: logS, kind: scalar, route: logS, prior: logS_prior}

split: {method: scaffold, fractions: [0.8, 0.1, 0.1], seed: 1}

model:
  structure: {encoder: dmpnn, hidden_dim: 128, depth: 3}
  descriptors: {enabled: true}
  fusion: concat
  prior: frozen
```

Everything is validated against a typed schema before any computation runs:

```bash
chemistry-framework validate my_experiment.yaml
```

Unknown keys, unknown feature names, and routes that reference undefined blocks
are errors, not silent defaults.

## 4. Run it

```bash
chemistry-framework prepare my_experiment.yaml            # featurise, cache, hash the definitions
chemistry-framework train   my_experiment.yaml            # fit, evaluate, write the run record
chemistry-framework predict --checkpoint <run>/model.pt  --input new_table.csv --output predictions.csv
```

`train` writes the checkpoint together with the resolved configuration, its
hash, the split seed, the per-split predictions and the metric table. Feature
statistics, scalers and priors are fitted on the training partition only; the
test partition is read once, for the final evaluation.

## 5. Read the evidence

```bash
chemistry-framework analyze my_experiment.yaml --inspect
```

This writes, next to the run record:

| File | Contents |
| :--- | :--- |
| `<name>-feature-evidence.json` / `.csv` | Per-feature provenance, training support, range, target association, collinear partners, and the assigned evidence class |
| `<name>-feature-report.html` | The same scorecard as a readable report |
| `<name>-correlation-heatmap.png` | Feature--feature correlation map used by the redundancy rule |
| `<name>-analysis.json` | The compact subset proposed by the recommender, with the reason each retained feature was kept and each dropped feature was flagged |

The scorecard and the recommendation are diagnostics for a decision you still
make: the recommender proposes a subset under a fixed relevance-and-redundancy
rule and never changes the model on its own.

## What the framework is not

It is not a model zoo and not a hyper-parameter search. It makes the choices
that are usually implicit -- data audit, feature evidence, which descriptors are
used, and what role they play -- explicit, recorded, and cheap to change.
