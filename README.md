# TEB Framework

TEB Framework is a Python library for building molecular-property models from
independent, replaceable blocks. The package is self-contained: it has no
runtime imports from other projects or research folders.

## Install

Install the latest version directly from GitHub:

```bash
pip install "git+https://github.com/soratori1025/Framework-Chemistry.git"
```

Install with molecular SMILES/SMARTS feature support:

```bash
pip install "teb-framework[chemistry] @ git+https://github.com/soratori1025/Framework-Chemistry.git"
```

For development, clone this repository and install editable extras:

```bash
git clone https://github.com/soratori1025/Framework-Chemistry.git
cd Framework-Chemistry
pip install -e ".[chemistry,test]"
```

The core dependencies are PyTorch and NumPy. RDKit is an optional dependency
used by the built-in SMILES feature registry. Models, priors, task metadata,
and diagnostics can also be used with caller-provided tensors without RDKit.

## Architecture

```text
SMILES / molecular graph / 3-D input
                 |
       Feature registry (optional)
                 |
     +-----------+------------+
     |                        |
Baseline / physics prior   Representation encoder
     |                        |
     |                   Residual network
     +----------- add --------+
                 |
       Property output head
                 |
       Output / constraints
                 |
     Feature diagnostics
```

The core does not assign special meaning to entropy, enthalpy, heat capacity, or
a fixed feature-vector size. A task selects the feature routes, prior, encoder,
residual, and output block it needs. A custom graph encoder can be passed as an
ordinary `torch.nn.Module`; graph batching stays the responsibility of the
selected encoder and input pipeline. `NASA7OutputBlock` can be configured with a
temperature grid and plugged in as the output block for a 12-value prediction
(10 Cp coefficients followed by reference H and S), or called directly with
separate coefficient and reference-property tensors.

## Minimal example

```python
import torch

from teb_framework import (
    LinearAdditivePrior,
    MLPEncoder,
    MLPResidual,
    MoleculeFeatureBlock,
    ResidualPropertyModel,
    ScalarOutputHead,
    molecular_feature_registry,
)

registry = molecular_feature_registry()
registry.add_smarts("n_ester", "C(=O)O")
features = MoleculeFeatureBlock(registry, ["n_C", "n_O", "n_ester"])
batch = features.transform(["CCO", "CC(=O)OC"])

model = ResidualPropertyModel(
    encoder=MLPEncoder(input_dim=3, representation_dim=32),
    prior=LinearAdditivePrior([0.2, -0.1, 0.5], intercept=1.0),
    residual=MLPResidual(representation_dim=32, output_dim=1),
    output_head=ScalarOutputHead(),
)
prediction = model(torch.as_tensor(batch.values, dtype=torch.float32))["prediction"]
```

## Train a custom task

`ResidualPropertyModel` is a regular PyTorch module. Use the user's own dataset,
split strategy, loss, and optimizer in a standard training loop; for example:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

for epoch in range(epochs):
    model.train()
    for feature_batch, target_batch in train_loader:
        optimizer.zero_grad()
        prediction = model(feature_batch)["prediction"]
        loss = loss_fn(prediction, target_batch)
        loss.backward()
        optimizer.step()
```

The `PropertyTask` dataclass stores target metadata and feature routes; it does
not prescribe a dataset, split, loss, or training policy. Use `torch.utils.data`
or your preferred data pipeline to keep these task-specific decisions explicit.

For a physics-informed baseline, implement `BaselineBlock` or wrap a
differentiable tensor function with `CallablePhysicsPrior`. `CompositePrior`
sums compatible prior blocks; this allows a physical term and a learned/frozen
additive contribution to remain independently replaceable.

`diagnose_features(train, test, names)` reports train ranges, test extrapolation,
constant-in-train features, and strongly correlated feature pairs. Run it after
the split is fixed so diagnostics do not leak test data into feature selection.

## Package structure

Each architectural layer has its own Python package. Implementations live in
focused modules, and each layer's `__init__.py` exposes its supported public
API:

```text
teb_framework/
├── features/       # feature contracts, registry, calculators, matrix builder
├── priors/         # baseline contract, additive/physics/composite priors
├── encoders/       # representation and residual neural blocks
├── models/         # composition of encoder, prior, residual, and output block
├── heads/          # scalar/vector and domain-specific output transformations
├── tasks/          # property/task metadata
└── diagnostics/    # feature and model-input checks
```

Import from the layer package when working on a specific component:

```python
from teb_framework.features import FeatureRegistry, FeatureSpec
from teb_framework.priors import BaselineBlock
from teb_framework.encoders import MLPEncoder
from teb_framework.heads import ScalarOutputHead
from teb_framework.models import ResidualPropertyModel
```

The top-level `teb_framework` exports common components as a convenience and
compatibility API. Prefer imports from a layer package in library internals so
the dependency direction remains clear.

## Extending a layer

- **Feature:** add a calculator in `features/`, describe it with `FeatureSpec`,
  and register it through `FeatureRegistry`. Keep chemistry-library imports
  local to calculators that require them.
- **Prior:** subclass `priors.BaselineBlock` and implement `forward(features)`.
  Return a tensor shaped `(batch, output_dim)` and declare `output_dim`.
- **Encoder or residual:** implement an `nn.Module` in `encoders/` that returns
  a batched tensor with a documented representation/output dimension.
- **Output:** add an `nn.Module` under `heads/`. Keep generic shape validation
  separate from property-specific transformations.
- **Task metadata / diagnostics:** add the behavior under its corresponding
  package rather than coupling it to model construction.

Export supported additions from that layer's `__init__.py`, then add tests for
the focused module and for composition through `ResidualPropertyModel`. Avoid
editing the top-level API unless the new component is intended as a common
convenience import.

## Development

Run the tests from the repository root:

```bash
pytest
```

CI checks the package and tests on supported Python versions. This library does
not impose a task-specific training CLI, dataset format, checkpoint policy, or
graph batching layer; callers can provide those independently for their task.
