# Chemistry Framework

Chemistry Framework is a standalone Python library for composing molecular-property
models from replaceable blocks. Its source does not import from the paper
implementations or other project folders.

## Install

Install the package with configuration and chemistry support:

```bash
pip install "chemistry-framework[config,chemistry] @ git+https://github.com/soratori1025/Framework-Chemistry.git"
```

For development:

```bash
git clone https://github.com/soratori1025/Framework-Chemistry.git
cd Framework-Chemistry
pip install -e ".[config,chemistry,test]"
```

PyTorch and NumPy are core dependencies. PyYAML (`config`) enables YAML
experiments; RDKit (`chemistry`) enables the built-in SMILES/SMARTS features,
scaffold/size splits, and graph SMARTS flags. Tensor-based components can be
used without RDKit. Install `benchmark` to load datasets through PyTDC:

```bash
pip install "chemistry-framework[config,chemistry,benchmark]"
```

The TDC adapter supports individual `ADME` and `Tox` datasets by name. It
downloads/loads the dataset through PyTDC and lets the experiment's configured
splitter create train/validation/test partitions. This is useful for framework
experiments, but it is **not** the fixed split distributed by the TDC
`BenchmarkGroup`; use that official protocol when claiming leaderboard
comparability.

## Public library API and design principles

The public package is intentionally centered on a reusable, importable API rather than on a single hard-coded task script. The supported pattern is:

```python
from chemistry_framework import ArchitectureRecipe, FeatureRole, FusionStrategy

recipe = (
    ArchitectureRecipe(
        name="prior_plus_graph",
        prior_features=("element_counts", "bond_pairs"),
        residual_features=("graph_flags",),
        fusion_strategy=FusionStrategy.ADDITIVE,
        prior_model="linear",
        residual_model="mlp",
        description="Explicit descriptor prior plus learnable residual encoder.",
    )
    .with_assignment("element_counts", FeatureRole.PRIOR, ("element_counts",), "Compositional prior")
    .with_assignment("graph_flags", FeatureRole.RESIDUAL, ("graph_flags",), "Residual graph signal")
)
```

This matches the framework intent: users can assign feature families to a `prior`, `residual`, `auxiliary`, or `direct` role, and they can change the fusion strategy without rewriting the entire modeling stack. The public library stays clean and task-agnostic; all analysis/evaluation artifacts remain local to a separate benchmarking workspace.

## Configure an experiment

Start from [`chemistry_framework/presets/general_regression.yaml`](chemistry_framework/presets/general_regression.yaml)
and edit the dataset path, target columns, features, tasks, and training
settings. Paths such as `dataset.path`, `cache.directory`, and
`analysis.output_dir` are resolved relative to the YAML file.

```yaml
name: my-enthalpy-task
dataset:
  path: data/molecules.csv
  smiles_column: smiles
  targets:
    enthalpy:
      column: H_f_kJ_mol
      unit: kJ/mol

features:
  composition:
    type: elements
    elements: [H, C, N, O, S]
  bonds:
    type: bond_pairs
    elements: [H, C, N, O, S]
    bond_orders: ["1", "2", "3", "a"]
  ester:
    type: smarts
    pattern: "[CX3](=O)[OX2][#6]"
    mode: count
    group: functional_groups
    description: Counts ester fragments
    interpretation: Ester functional-group abundance

priors:
  enthalpy:
    use: [composition, bonds]
    fit: ridge
routes:
  enthalpy: [ester]
tasks:
  enthalpy:
    target: enthalpy
    kind: scalar
    prior: enthalpy
    route: enthalpy

split:
  method: scaffold
  fractions: [0.8, 0.1, 0.1]
  seed: 42
training:
  epochs: 100
  batch_size: 64
  learning_rate: 0.001
  weight_decay: 0.0001
  loss: huber
  patience: 15
  device: auto
evaluation:
  metrics: [mae, rmse, r2]
analysis:
  enabled: true
  support_threshold: 20
  correlation_threshold: 0.95
  bins: 20
cache:
  enabled: true
  directory: .chemistry_cache/features
```

### TDC benchmark presets

Four starter configurations cover distinct property/task combinations:

| Preset | TDC source | Task | Split |
| --- | --- | --- | --- |
| `chemistry_framework/presets/admet_caco2.yaml` | `ADME / Caco2_Wang` | Regression | Scaffold |
| `chemistry_framework/presets/admet_lipophilicity.yaml` | `ADME / Lipophilicity_AstraZeneca` | Regression | Scaffold |
| `chemistry_framework/presets/admet_bbb.yaml` | `ADME / BBB_Martins` | Binary classification | Scaffold |
| `chemistry_framework/presets/admet_herg.yaml` | `Tox / hERG` | Binary classification | Scaffold |

Validate a preset before using it:

```bash
chemistry-framework validate chemistry_framework/presets/admet_caco2.yaml
chemistry-framework validate chemistry_framework/presets/admet_bbb.yaml
```

TDC data is loaded lazily by `prepare`, which requires the `benchmark` extra and
network/cache access on first use. Presets use the public `Drug` and `Y`
columns. Check dataset licenses and endpoint/assay definitions before
redistributing data or interpreting results. In particular, results from a
locally generated scaffold split must not be reported as TDC official
benchmark results.

Classification tasks declare `kind: binary_classification` or
`kind: multilabel_classification` and use `training.loss: bce`. Models return
logits during training; `evaluate_model(..., task_kind="binary_classification")`
applies sigmoid and reports configured classification metrics. Missing
multi-label targets are masked per label. For classification, use the task kind
explicitly when evaluating:

```python
metrics, actual, probabilities = evaluate_model(
    model,
    loaders.test,
    config.evaluation.metrics,
    device=config.training.device,
    task_kind="binary_classification",
)
```

To support meaningful comparisons, hold split seeds and model/training
settings fixed across feature ablations, retain a validation set for model
selection, and report repeated-run mean and standard deviation. Use at least
five independent seeds for benchmark claims; a single run is only a smoke
test. The framework does not claim that one prior or feature set improves all
ADMET endpoints.

The configuration has separate responsibilities:

- `dataset` declares the input CSV, SMILES column, and scalar or multi-column
  targets. Blank target cells are represented as missing values; vector-task
  training uses rows whose complete target vector is finite.
- `features` declares reusable blocks: element counts, generated bond-pair
  counts, SMARTS counts/presence flags, trusted Python calculators, and
  versioned user-supplied Benson-group providers.
- `priors` selects additive feature blocks and can fix individual expanded
  coefficients. `routes` selects descriptors for each task's neural branch.
  A feature can be used in both places.
- `tasks` maps targets to their task kind, prior, route, and loss weight.
- `split` supports `random`, `size`, `scaffold`, and SMARTS test holdout. For
  example, `method: smarts` with `test_smarts: "[CX3](=O)[OX2][#6]"` puts all
  matching molecules in test and partitions the remainder into train and
  validation.
-   `training` and `evaluation` hold optimizer/loop settings and regression or
  classification metrics. `analysis` configures support thresholds, correlation checks,
  distribution histograms, and report output. `cache` stores feature rows by
  canonical molecule and feature-schema fingerprint.
- `graph.atom_smarts_flags` and `graph.bond_smarts_flags` define local graph
  input flags. Changing them changes graph input dimensions and requires
  training a new encoder/checkpoint.

SMARTS and Python plugins execute chemistry/user logic. Python plugins must be
treated as trusted code; YAML `safe_load` does not make imported plugins safe.
A Benson provider is explicitly user supplied and should include a `version`
that changes whenever its group rules change. The framework does not silently
approximate Benson rules.

## Validate, prepare, train, evaluate

Validate the YAML and feature/task references before running:

```bash
chemistry-framework validate experiment.yaml
chemistry-framework prepare experiment.yaml
```

`prepare` loads the dataset, creates the split, builds feature columns (the
Benson vocabulary is selected using train rows only), applies the
configuration-keyed cache, and writes the configured feature/target analysis
JSON. It does not instantiate a task-specific graph model or silently train
one: users retain control of their encoder, loss, and graph batching.
When analysis is enabled, preparation also writes a model-independent feature
evidence scorecard (JSON and CSV), an HTML report, and a PNG training-set
correlation heatmap:

- `*-feature-evidence.json` preserves feature support, train/test ranges,
  train-only target associations, collinearity, provenance, and available
  prior/model evidence.
- `*-feature-evidence.csv` is the tabular scorecard for downstream analysis.
- `*-feature-report.html` summarizes the scorecard and group contributions.
- `*-correlation-heatmap.png` visualizes the training-feature correlation
  matrix; row/column order matches the CSV feature order.

The `group`, `description`, `interpretation`, `source`, and `definition`
fields are optional per-feature metadata. If omitted, the feature block/domain
and calculator metadata are used where possible; the framework does not invent
a chemistry explanation for a feature whose definition is unknown. The CLI
`analyze` command is a convenient alias for preparing the configured data and
writing these diagnostics:

```bash
chemistry-framework analyze experiment.yaml
```

Feature evidence is deliberately split into distinct kinds:

- **Intrinsic evidence** is model-independent: support, range coverage,
  train-only Spearman association, provenance, and redundancy indicators.
- **Prior evidence** reports fitted coefficient stability across supplied
  runs. Bootstrap intervals are descriptive with few runs, not guarantees.
- **Model-dependent evidence** can be obtained with grouped test-set
  permutation or leave-group-out retraining utilities in
  `chemistry_framework.feature_analysis`. Positive metric deltas mean the
  complete model outperformed the perturbed/reduced model. These are
  predictive, model-specific measurements and do not imply causal effects.

The scorecard avoids interpreting a near-zero coefficient as proof of no
chemical effect. A feature is only labelled `supported` when sufficient
training support, stable prior coefficients, and consistent positive
model-specific group evidence are all present. Without prior/model evidence,
the default class remains `weak_uncertain`; high collinearity is labelled
`redundant`, and no/limited support or test-range extrapolation is flagged.
These labels describe the tested data and protocol, not universal feature
validity. Feature matrices currently require finite values, so their reported
feature missing count is zero; missing target values are still represented
and summarized separately.

For descriptor-based tasks, the library provides loaders, a generic PyTorch
training loop, prior fitting, and evaluation:

```python
from chemistry_framework.config import load_config
from chemistry_framework.data import make_tensor_dataloaders
from chemistry_framework.evaluation import evaluate_model, load_model_checkpoint
from chemistry_framework.experiment import fit_experiment_prior, prepare_experiment
from chemistry_framework.training import train_model

config = load_config("experiment.yaml")
experiment = prepare_experiment(config)
loaders = make_tensor_dataloaders(experiment, "enthalpy")
prior_fit = fit_experiment_prior(experiment, "enthalpy")

# Assemble a torch.nn.Module compatible with your configured route and prior.
model = make_my_model(config, experiment, prior_fit.prior)
checkpoint_path = config.resolve_path(config.training.checkpoint_dir) / "enthalpy.pt"
result = train_model(
    model,
    loaders.train,
    loaders.validation,
    config.training,
    checkpoint_path=checkpoint_path,
)
if config.evaluation.checkpoint:
    load_model_checkpoint(
        result.model,
        config.resolve_path(config.evaluation.checkpoint),
        device=config.training.device,
    )
metrics, actual, predicted = evaluate_model(
    result.model, loaders.test, config.evaluation.metrics, device=config.training.device
)
```

`make_tensor_dataloaders` applies the task's configured descriptor route.
`fit_experiment_prior` fits scalar or vector linear priors only on training
rows and keeps fixed coefficients unchanged (a scalar fixed coefficient is
shared across the prior outputs). If multiple configured tasks share a target,
pass `task_name` to `make_tensor_dataloaders` to select that task's route. For
graph models, construct loaders that carry graph objects and
pass them to your own encoder/training integration; graph SMARTS flags are
available from `chemistry_framework.encoders`.

The analysis report includes per-feature train support and histograms,
constant/low-support columns, collinear pairs, test values outside train
ranges, and per-split target distributions. For molecules supplied at
inference time, `analyze_prediction_molecules` compares their descriptors
against the train range. Use the training partition for feature selection;
the validation/test report is diagnostic and must not be used to fit the
vocabulary or prior.

## Architecture and extension points

```text
chemistry_framework/
├── analysis/       # feature/target reports and prediction-range checks
├── config/         # validated YAML schema and loader
├── data/           # CSV records and tensor dataloaders
├── diagnostics/    # ranges, support, and collinearity
├── encoders/       # MLP blocks and configurable graph flags
├── evaluation/     # regression metrics
├── features/       # feature contracts, registry, cache, calculators
├── heads/          # scalar/vector and NASA-7 output blocks
├── models/         # model composition
├── priors/         # additive, physics, composite, and fitted priors
├── splits/         # random, size, scaffold, and SMARTS splitters
├── tasks/          # task metadata
└── training/       # generic PyTorch loop and checkpointing
```

Import from a focused layer package when extending it:

```python
from chemistry_framework.features import FeatureRegistry, FeatureSpec
from chemistry_framework.priors import BaselineBlock
from chemistry_framework.encoders import MLPEncoder
from chemistry_framework.heads import ScalarOutputHead
from chemistry_framework.models import ResidualPropertyModel
```

Add calculators under `features/`, subclasses of `priors.BaselineBlock` under
`priors/`, encoder/residual modules under `encoders/`, and output transforms
under `heads/`. Keep each component independently testable, export its public
API from that layer's `__init__.py`, and compose it with
`ResidualPropertyModel`. `NASA7OutputBlock` is available when a task predicts
the coefficient/reference-property vector expected by that block.

The generic prior is additive by design; graph encoders can use sum
aggregation to preserve size-additive behavior. Other task/model choices are
user supplied. Exact v15 numerical compatibility is **not claimed** by the
generic preset: establishing it requires porting the complete v15 feature
definition and adding golden-vector tests against the paper implementation.
The framework is independent of both paper folders, so entropy or other
physics-specific terms can be supplied as a trusted plugin or a
`CallablePhysicsPrior` without changing their paper implementation.

## Development

```bash
pytest
python -m pip wheel . --no-deps --no-build-isolation --wheel-dir /tmp/chemistry-framework-wheel
```

CI runs the package tests on the supported Python versions.
