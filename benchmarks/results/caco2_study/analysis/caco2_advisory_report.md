# Chemical Dataset Analysis & Advisory Report: `caco2_wang`

**Health Score**: `95/100` (EXCELLENT)  
**Total Records**: `910` | **Valid Molecules**: `910` | **Unique Scaffolds**: `449`  
**Tanimoto Diversity**: `0.87` (Mean similarity: `0.13`)

## 1. Executive Summary & Health Check
The dataset contains **910** compounds with **49.3%** Bemis-Murcko scaffold diversity and **0.87** chemical space diversity. 
Overall health rating is **EXCELLENT** (95/100).

### Risk Assessment Table
| Severity | Category | Risk Factor | Recommendation |
| :--- | :--- | :--- | :--- |
| `LOW` | `data_quality` | **Moderate Sample Size (300 <= N < 1000)**: Sample size of 910 compounds is well-suited for D-MPNN or GIN with warm-up learning rate schedule. | Train with Cosine annealing and early stopping patience=15. |
| `LOW` | `architecture` | **High Chemical Diversity & Scaffold Hopping**: Mean pairwise Tanimoto similarity is 0.13 (Diversity: 0.87). | Global Virtual Supernode and Graph Attention readout enabled to prevent over-squashing across distant pharmacophores. |

## 2. Chemical Property Profiling

| Molecular Property | Mean ± Std | 25% | Median | 75% | Range [Min, Max] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Molecular Weight (MW)** | 414.7 ± 156.0 | 307.4 | 400.4 | 514.7 | [32.0, 967.0] |
| **LogP** | 2.1 ± 2.2 | 1.0 | 2.5 | 3.5 | [-7.6, 8.7] |
| **Heavy Atoms** | 29.4 ± 11.1 | 22.0 | 28.0 | 36.0 | [2.0, 68.0] |
| **H-Bond Donors** | 2.4 ± 2.0 | 1.0 | 2.0 | 3.0 | [0.0, 14.0] |
| **H-Bond Acceptors** | 5.8 ± 2.9 | 4.0 | 6.0 | 7.0 | [1.0, 23.0] |
| **TPSA (Å²)** | 102.5 ± 53.2 | 64.8 | 95.9 | 133.1 | [3.2, 374.1] |
| **Rotatable Bonds** | 5.7 ± 4.0 | 3.0 | 6.0 | 8.0 | [0.0, 23.0] |

## 4. Target Distribution Analysis

### Target: `Y`
- **Task Type**: `scalar_regression`
- **Valid Entries**: `910` / 910 (0.0% missing)
- **Distribution**: Mean `-5.24` ± `0.78` | Median `-5.13`
- **Skewness**: `-0.66` | **Outliers**: `4` (0.4%)

## 5. Prescriptive Action Plan & Recommended Training Recipe

Based on the profiling above, the automated advisor recommends the following configuration:

1. **Validation Split**: `scaffold` split to evaluate out-of-scaffold generalization.
2. **Representation Architecture**: `DMPNN` structure encoder combined with `hybrid` descriptor fusion.
3. **Optimization & Head**: Output head `standard`, Loss `pairwise_ranking`.
4. **Learning Rate & Scheduler**: `Cosine` schedule with 3 warmup epochs.
5. **Uncertainty & Ensembling**: `1` models in ensemble.

### Recommended YAML Configuration
```yaml
name: caco2_wang-recommended
dataset:
  path: /home/nvsinh1/brats_segmentation/teb-net-model/framework/data/caco2_wang.tab
  smiles_column: Drug
  targets:
    Y:
      column: Y
features:
  composition:
    type: elements
    elements:
    - H
    - C
    - N
    - O
    - S
    - P
    - F
    - Cl
    - Br
    - I
  bonds:
    type: bond_pairs
    elements:
    - H
    - C
    - N
    - O
    - S
    - P
    - F
    - Cl
    - Br
    - I
    bond_orders:
    - '1'
    - '2'
    - '3'
    - a
  aromatic:
    type: smarts
    pattern: a
    mode: count
routes:
  Y:
  - composition
  - bonds
  - aromatic
tasks:
  Y:
    target: Y
    kind: scalar_regression
    route: Y
split:
  method: scaffold
  fractions:
  - 0.8
  - 0.1
  - 0.1
  seed: 42
model:
  structure:
    encoder: dmpnn
    hidden_dim: 64
    depth: 3
    aggregation: attention
    dropout: 0.2
    virtual_node: true
  descriptors:
    enabled: true
    hidden_dim: 64
    depth: 1
    dropout: 0.1
  fusion: concat
  ffn:
    hidden_dim: 128
    depth: 2
    dropout: 0.1
  head: standard
training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  loss: pairwise_ranking
  scheduler: cosine
  warmup_epochs: 3
  grad_clip: 1.0
  patience: 15
  ensemble_size: 1
  device: auto
  checkpoint_dir: outputs/checkpoints
evaluation:
  metrics:
  - mae
  - rmse
  - r2
  - spearman
analysis:
  enabled: true
  output_dir: outputs/analysis
cache:
  enabled: true
  directory: .chemistry_cache/features
```
