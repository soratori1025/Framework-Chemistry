# Configuration reference

An experiment is one YAML file validated against the typed schema in
`chemistry_framework/config/schema.py`. Validation runs before any computation;
an unknown key, an unknown feature name, or a route that points at a feature
block that does not exist is reported as an error.

Top-level sections:

| Section | Purpose |
| :--- | :--- |
| `name` | Experiment name; used as the stem for every artefact written by the run. |
| `dataset` | Input table, SMILES column, targets, task kind (`scalar` or `binary`), and the file provider. |
| `features` | Named feature blocks (the *what*). |
| `priors` | Baseline components fitted on the training partition (the *baseline*). |
| `routes` | Which feature blocks a task consumes (the *what is used where*). |
| `tasks` | Target, task kind, route, prior, and any per-task overrides. |
| `split` | Split strategy, fractions, and seed. |
| `model` | Encoder, fusion, descriptors on/off, and the role of each input. |
| `training` | Optimiser, loss, learning rate, epochs, batch size, early stopping. |
| `evaluation` | Primary and secondary metrics, checkpoint rule. |
| `analysis` | Scorecard, recommender and stability settings. |
| `cache` | Feature-cache directory and hashing of feature definitions. |
| `graph` | Graph construction options (atom/bond features, directed or undirected). |

## Feature blocks

```yaml
features:
  elements:     {type: element_counts}
  bonds:        {type: bond_pair_counts}
  fragments:    {type: smarts_patterns, patterns: [...]}
  ecfp:         {type: morgan_fingerprint, radius: 2, n_bits: 2048}
  physchem:     {type: rdkit_descriptors, names: [MolWt, MolLogP, TPSA, ...]}
  rdkit2d:      {type: rdkit_descriptors, families: [all]}
```

Each block may carry `group`, `description`, `source` and `definition`. The
resolved feature names and definitions are hashed, so a cached matrix can always
be traced back to the exact definitions that produced it.

## Routes and roles

A **route** assigns feature blocks to a task; a **role** says how the routed
features enter the predictor. Three roles are supported:

| Role | Where it is declared | Behaviour |
| :--- | :--- | :--- |
| Direct | `routes` + `model.descriptors.enabled` | Encoded by an MLP; concatenated with the graph representation when a graph encoder is active. |
| Prior | `priors.<name>` + `tasks.<task>.prior` | A baseline (ridge, least squares, or logistic regression on the training partition), optionally frozen, added to the network output. |
| Residual | `model.prior: frozen` with a graph branch | The graph network learns a correction to the baseline, as in delta-machine-learning. |

Disabling the graph encoder gives a descriptor-only model; disabling the
descriptor branch gives a graph-only model. The five manuscript configurations
(`prior_only`, `direct_only`, `residual_only`, `direct_hybrid`,
`prior_residual`) are exactly these switches.

## Splits

```yaml
split: {method: scaffold, fractions: [0.8, 0.1, 0.1], seed: 1}
```

`method` is `scaffold`, `random`, or `size`. Scaffold splits group molecules by
Bemis--Murcko scaffold, treating acyclic molecules as singleton groups; the
groups are shuffled with the split seed, sorted by decreasing size, and each is
assigned to the partition whose current size is lowest relative to its target
fraction. Large groups can therefore land in the validation or test partition.
The benchmark reuses the same saved manifests for every model, so the split
seed is the analysis unit.

## Analysis

```yaml
analysis:
  output_dir: analysis
  cv_folds: 5                 # folds for prior-coefficient bootstrap
  bootstrap_rounds: 20
  stability_splits: [random, scaffold, size]
  nonlinear_model: random_forest
  seed: 0
```

The scorecard is always computed on the training partition. Out-of-range
evidence uses the held-out values and is therefore post-hoc: it is reported but
never used for selection.

## Where to look next

* `../README.md` -- what the framework is, and what the manuscript found.
* `../REPRODUCING.md` -- how to regenerate every number, table and figure.
* `quickstart.md` -- a worked example from CSV to recorded run.
