import csv
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from chemistry_framework.config import ExperimentConfig, ModelSpec, TrainingConfig, load_config
from chemistry_framework.inputs import (
    ComplexAdapter,
    ComplexObject,
    MixtureAdapter,
    MixtureObject,
    available_input_formats,
    get_input_adapter,
)
from chemistry_framework.data import (
    BatchMolGraph,
    ChemistryDataset,
    MolGraph,
    collate_mol_graphs,
    mol_to_graph,
    make_experiment_dataloaders,
)
from chemistry_framework.encoders import (
    DMPNNEncoder,
    EGNNEncoder,
    GINEncoder,
    GraphAggregation,
    ProteinEncoder,
)
from chemistry_framework.experiment import prepare_experiment
from chemistry_framework.heads import (
    EvidentialRegressionHead,
    GaussianNLLHead,
    MultiTaskHead,
    NodeLevelHead,
)
from chemistry_framework.models import (
    EnsembleModel,
    HybridChemistryModel,
    build_hybrid_model,
)
from chemistry_framework.training import TrainingResult, train_ensemble, train_model
from chemistry_framework.training.engine import _evidential_loss, _gaussian_nll_loss
from chemistry_framework.cli import main as cli_main

def test_input_adapters():
    """Test Mixture and Complex adapters and registry."""
    formats = available_input_formats()
    assert "mixture" in formats
    assert "complex" in formats

    mix_adapter = get_input_adapter("mixture")
    assert isinstance(mix_adapter, MixtureAdapter)
    mix_obj = mix_adapter.parse("CCO:0.7;O:0.3")
    assert isinstance(mix_obj, MixtureObject)
    assert len(mix_obj.components) == 2
    assert pytest.approx(sum(mix_obj.fractions)) == 1.0
    assert mix_obj.smiles == "CCO:0.7;O:0.3"

    complex_adapter = get_input_adapter("complex")
    assert isinstance(complex_adapter, ComplexAdapter)
    comp_obj = complex_adapter.parse("CC(=O)O|MVLSPADKTN")
    assert isinstance(comp_obj, ComplexObject)
    assert comp_obj.ligand.smiles == "CC(=O)O"
    assert comp_obj.protein_sequence == "MVLSPADKTN"
    assert comp_obj.ligand.mol is not None

def test_graph_featurizer_and_batching():
    """Test mol_to_graph and collate_mol_graphs."""
    smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
    graphs = [mol_to_graph(s) for s in smiles_list]
    assert len(graphs) == 3
    for g in graphs:
        assert isinstance(g, MolGraph)
        assert g.num_atoms > 0
        assert g.num_bonds >= 0
        assert g.atom_features.shape[0] == g.num_atoms
        assert g.edge_features.shape[0] == g.num_edges

    batch = collate_mol_graphs(graphs)
    assert isinstance(batch, BatchMolGraph)
    total_atoms = sum(g.num_atoms for g in graphs)
    total_edges = sum(g.num_edges for g in graphs)
    assert batch.num_atoms == total_atoms
    assert batch.num_edges == total_edges
    assert len(batch.batch) == total_atoms
    assert batch.a2b.shape[0] == total_atoms
    assert batch.b2rev.shape[0] == total_edges

def test_encoders_forward():
    """Test forward passes for DMPNN, GIN, EGNN, and Protein encoders."""
    graphs = [mol_to_graph("CC"), mol_to_graph("CCC")]
    batch = collate_mol_graphs(graphs)

    dmpnn = DMPNNEncoder(hidden_dim=32, depth=3, aggregation="mean")
    out_dmpnn = dmpnn(batch)
    assert out_dmpnn.shape == (2, 32)

    gin = GINEncoder(hidden_dim=32, depth=3, aggregation="sum")
    out_gin = gin(batch)
    assert out_gin.shape == (2, 32)

    egnn = EGNNEncoder(hidden_dim=32, depth=2)
    out_egnn = egnn(batch)
    assert out_egnn.shape == (2, 32)

    prot_encoder = ProteinEncoder(embedding_dim=16, channels=32)
    seqs = ["MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSA", "ACDEFGHIKLMNPQRSTVWY"]
    from chemistry_framework.data.loaders import _tokenize_protein
    tokens = torch.stack([_tokenize_protein(s) for s in seqs])
    out_prot = prot_encoder(tokens)
    assert out_prot.shape == (2, 32)

def test_specialized_heads_and_losses():
    """Test Gaussian NLL and Evidential Regression heads and loss functions."""

    head_gauss = GaussianNLLHead(in_features=32, out_targets=1)
    x = torch.randn(4, 32)
    y = torch.randn(4, 1)
    res_gauss = head_gauss(x)
    assert "mean" in res_gauss
    assert "variance" in res_gauss
    loss_gauss = _gaussian_nll_loss(res_gauss, y).mean()
    assert loss_gauss.dim() == 0
    assert not torch.isnan(loss_gauss)

    head_evid = EvidentialRegressionHead(in_features=32, out_targets=1)
    res_evid = head_evid(x)
    assert "prediction" in res_evid
    assert "aleatoric_variance" in res_evid
    assert "epistemic_variance" in res_evid
    loss_evid = _evidential_loss(res_evid, y, coeff=0.1).mean()
    assert loss_evid.dim() == 0
    assert not torch.isnan(loss_evid)

    head_multi = MultiTaskHead(in_dim=32, task_dims={"solubility": 1, "lipophilicity": 1, "clearance": 2})
    res_multi = head_multi(x)
    assert res_multi["solubility"].shape == (4, 1)
    assert res_multi["lipophilicity"].shape == (4, 1)
    assert res_multi["clearance"].shape == (4, 2)

def test_hybrid_chemistry_model():
    """Test build_hybrid_model for pure graph and hybrid setups."""

    spec_graph = ModelSpec(
        structure={"encoder": "dmpnn", "hidden_dim": 32, "depth": 2, "aggregation": "mean"},
        ffn={"hidden_dim": 32, "depth": 1, "dropout": 0.0},
        head="scalar",
    )
    model_graph = build_hybrid_model(spec_graph)
    graphs = [mol_to_graph("CCO"), mol_to_graph("CC(=O)O")]
    batch = collate_mol_graphs(graphs)
    pred_graph = model_graph(batch_graph=batch)
    assert isinstance(pred_graph, dict)
    assert pred_graph["prediction"].shape == (2,)

    spec_hybrid = ModelSpec(
        structure={"encoder": "gin", "hidden_dim": 32, "depth": 2},
        descriptors={"hidden_dim": 32, "depth": 2},
        fusion="concat",
        ffn={"hidden_dim": 32, "depth": 1},
        head="evidential",
    )
    model_hybrid = build_hybrid_model(spec_hybrid, descriptor_dim=10)
    desc = torch.randn(2, 10)
    pred_hybrid = model_hybrid(batch_graph=batch, descriptors=desc)
    assert isinstance(pred_hybrid, dict)
    assert pred_hybrid["prediction"].shape == (2, 1)
    assert pred_hybrid["epistemic_variance"].shape == (2, 1)

def test_ensemble_model():
    """Test deep ensemble predictions and variance decomposition."""
    spec = ModelSpec(
        structure={"encoder": "dmpnn", "hidden_dim": 16, "depth": 2},
        ffn={"hidden_dim": 16, "depth": 1},
        head="gaussian",
    )
    m1 = build_hybrid_model(spec)
    m2 = build_hybrid_model(spec)
    ensemble = EnsembleModel([m1, m2])

    graphs = [mol_to_graph("CCO"), mol_to_graph("CC")]
    batch = collate_mol_graphs(graphs)
    res = ensemble(batch_graph=batch)
    assert "mean" in res
    assert "epistemic_variance" in res
    assert "total_variance" in res
    assert res["mean"].shape == (2, 1)
    assert (res["total_variance"] >= 0).all()

def _make_test_experiment(tmp_path: Path):
    """Helper to create a minimal PreparedExperiment with CSV dataset."""
    csv_path = tmp_path / "data.csv"
    smiles = ["C", "CC", "CCC", "CCCC", "CCO", "CCCO", "CCN", "CCCN"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["smiles", "target"])
        writer.writeheader()
        for idx, s in enumerate(smiles):
            writer.writerow({"smiles": s, "target": float(idx * 1.5)})

    config_data = {
        "name": "test_exp",
        "dataset": {
            "path": csv_path.name,
            "smiles_column": "smiles",
            "targets": {
                "target": {"column": "target"},
            },
        },
        "features": {
            "composition": {"type": "elements", "elements": ["C", "O", "N"]},
        },
        "priors": {},
        "routes": {},
        "tasks": {
            "target": {
                "target": "target",
                "kind": "scalar",
            }
        },
        "split": {"method": "random", "fractions": [0.6, 0.2, 0.2], "seed": 42},
        "training": {
            "epochs": 3,
            "batch_size": 2,
            "learning_rate": 0.01,
            "scheduler": "cosine",
            "warmup_epochs": 1,
            "grad_clip": 1.0,
            "ensemble_size": 2,
        },
        "evaluation": {"metrics": ["mae", "rmse"]},
        "analysis": {"enabled": False},
        "cache": {"enabled": False},
        "model": {
            "structure": {"encoder": "dmpnn", "hidden_dim": 16, "depth": 2},
            "descriptors": {"hidden_dim": 16, "depth": 1},
            "ffn": {"hidden_dim": 16, "depth": 1},
            "head": "regression",
        },
    }
    cfg = ExperimentConfig.parse(config_data, source_path=tmp_path / "experiment.yaml")
    return prepare_experiment(cfg)

def test_chemistry_dataset_and_loaders(tmp_path: Path):
    """Test ChemistryDataset and make_experiment_dataloaders."""
    experiment = _make_test_experiment(tmp_path)
    loaders = make_experiment_dataloaders(experiment, "target")
    assert len(loaders.train) > 0
    batch_inputs, batch_targets = next(iter(loaders.train))
    assert "graph" in batch_inputs
    assert isinstance(batch_inputs["graph"], BatchMolGraph)
    assert "descriptors" in batch_inputs
    assert batch_targets.ndim >= 1

def test_training_engine_schedulers_and_ensemble(tmp_path: Path):
    """Test train_model with warmup/cosine scheduler and train_ensemble."""
    experiment = _make_test_experiment(tmp_path)
    loaders = make_experiment_dataloaders(experiment, "target")

    spec = ModelSpec.parse(experiment.config.model)

    def factory():
        return build_hybrid_model(spec, descriptor_dim=len(experiment.features.names))

    result = train_model(
        model=factory(),
        train_loader=loaders.train,
        validation_loader=loaders.validation,
        config=experiment.config.training,
    )
    assert isinstance(result, TrainingResult)
    assert len(result.history) == 3
    assert result.best_epoch > 0

    ensemble, member_results = train_ensemble(
        model_factory=factory,
        train_loader=loaders.train,
        validation_loader=loaders.validation,
        config=experiment.config.training,
    )
    assert len(member_results) == 2
    assert isinstance(ensemble, EnsembleModel)

def test_yaml_config_with_model_spec(tmp_path: Path):
    """Test loading and validating complete YAML config with new model spec."""
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("smiles,solubility\nCCO,1.0\nCCC,2.0\n")

    yaml_content = f"""
name: test_dataset_exp
dataset:
  path: {csv_file.name}
  smiles_column: smiles
  targets:
    solubility:
      column: solubility
features:
  comp:
    type: elements
    elements: [C, O]
priors: {{}}
routes: {{}}
tasks:
  solubility:
    target: solubility
    kind: scalar
training:
  epochs: 5
  batch_size: 16
  learning_rate: 0.001
  scheduler: cosine
  warmup_epochs: 1
  grad_clip: 1.0
  amp: false
  ensemble_size: 3
  evidential_lambda: 0.2
model:
  structure:
    encoder: dmpnn
    hidden_dim: 64
    depth: 3
    aggregation: mean
  descriptors:
    hidden_dim: 64
    depth: 2
  fusion: concat
  ffn:
    hidden_dim: 64
    depth: 2
    dropout: 0.1
  head: evidential
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)

    cfg = load_config(config_file)
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.training.scheduler == "cosine"
    assert cfg.training.ensemble_size == 3
    assert cfg.training.evidential_lambda == 0.2
    assert cfg.model_spec is not None
    assert cfg.model_spec.structure.encoder == "dmpnn"
    assert cfg.model_spec.head == "evidential"

def test_cli_validate_and_train_mock(tmp_path: Path):
    """Test CLI validate command."""
    csv_file = tmp_path / "toy.csv"
    csv_file.write_text("smiles,logP\nCCO,1.0\nCCC,2.0\n")

    yaml_content = f"""
name: toy_exp
dataset:
  path: {csv_file.name}
  smiles_column: smiles
  targets:
    logP:
      column: logP
features:
  comp:
    type: elements
    elements: [C, O]
priors: {{}}
routes: {{}}
tasks:
  logP:
    target: logP
    kind: scalar
model:
  structure:
    encoder: gin
    hidden_dim: 32
"""
    config_file = tmp_path / "toy_config.yaml"
    config_file.write_text(yaml_content)

    exit_code = cli_main(["validate", str(config_file)])
    assert exit_code == 0
