from __future__ import annotations
import numpy as np
import pytest
import torch
from rdkit import Chem
from chemistry_framework.analysis import DatasetAdvisor
from chemistry_framework.config import ExperimentConfig, ModelSpec
from chemistry_framework.data.graph import BatchMolGraph, mol_to_graph, collate_mol_graphs
from chemistry_framework.encoders.gnn import DMPNNEncoder, GINEncoder
from chemistry_framework.evaluation.metrics import regression_metrics
from chemistry_framework.features.pharmacophore import (
    AutoPharmacophoreBlock,
    extract_morgan_fingerprints,
    extract_rdkit_2d_descriptors,
    select_top_features,
)
from chemistry_framework.models.hybrid import build_hybrid_model
from chemistry_framework.training.engine import _focal_loss, _pairwise_ranking_loss, train_model

def test_extract_morgan_fingerprints():
    smiles = ["c1ccccc1", "c1ccccc1O", "CC(=O)O"]
    fps, names = extract_morgan_fingerprints(smiles, radius=2, n_bits=1024, use_counts=False)
    assert fps.shape == (3, 1024)
    assert len(names) == 1024
    assert names[0] == "ecfp_2_1024.bit_0"

    assert np.sum(fps[0]) > 0

def test_extract_rdkit_2d_descriptors():
    smiles = ["c1ccccc1", "c1ccccc1O", "CC(=O)O"]
    descs, names = extract_rdkit_2d_descriptors(smiles)
    assert descs.shape[0] == 3
    assert descs.shape[1] >= 100
    assert len(names) == descs.shape[1]
    assert np.all(np.isfinite(descs))

def test_select_top_features():
    np.random.seed(42)

    X = np.random.randn(50, 20).astype(np.float32)

    X[:, 0] = 5.0

    y = X[:, 1] * 3.0 + np.random.randn(50) * 0.1
    names = [f"f_{i}" for i in range(20)]

    X_sel, sel_names, idxs = select_top_features(X, names, y, task_kind="scalar_regression", max_features=5)
    assert X_sel.shape == (50, 5)
    assert len(sel_names) == 5

    assert "f_0" not in sel_names

    assert "f_1" in sel_names

def test_auto_pharmacophore_block():
    train_smiles = ["c1ccccc1", "c1ccccc1O", "c1ccccc1N", "CC(=O)O", "CCO", "CCCO"] * 5
    y_train = np.linspace(1.0, 10.0, len(train_smiles))
    test_smiles = ["c1ccccc1", "CCO"]

    block = AutoPharmacophoreBlock(top_k=25, use_morgan=True, use_rdkit_2d=True)
    X_train, names = block.fit_transform(train_smiles, y_train, task_kind="scalar_regression")
    assert X_train.shape[0] == len(train_smiles)
    assert X_train.shape[1] == 25
    assert len(block.selected_feature_names) == 25
    assert len(names) == 25

    X_test = block.transform(test_smiles)
    assert X_test.shape == (len(test_smiles), 25)
    assert np.all(np.isfinite(X_test))

def test_dmpnn_with_virtual_node():
    smiles_list = ["c1ccccc1", "c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O"]
    graphs = [mol_to_graph(Chem.MolFromSmiles(s)) for s in smiles_list]
    batch = collate_mol_graphs(graphs)

    encoder = DMPNNEncoder(hidden_dim=64, depth=3, use_virtual_node=True)
    out = encoder(batch)
    assert out.shape == (3, 64)

    loss = out.sum()
    loss.backward()
    assert encoder.vn_embedding.grad is not None
    assert torch.all(torch.isfinite(encoder.vn_embedding.grad))

def test_gin_with_virtual_node():
    smiles_list = ["c1ccccc1", "c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O"]
    graphs = [mol_to_graph(Chem.MolFromSmiles(s)) for s in smiles_list]
    batch = collate_mol_graphs(graphs)

    encoder = GINEncoder(hidden_dim=64, depth=3, use_virtual_node=True)
    out = encoder(batch)
    assert out.shape == (3, 64)

    loss = out.sum()
    loss.backward()
    assert encoder.vn_embedding.grad is not None
    assert torch.all(torch.isfinite(encoder.vn_embedding.grad))

def test_hybrid_model_with_virtual_node():
    spec_dict = {
        "structure": {
            "encoder": "dmpnn",
            "hidden_dim": 64,
            "depth": 3,
            "virtual_node": True,
            "aggregation": "attention",
        },
        "descriptors": {"enabled": False},
        "ffn": {"hidden_dim": 64, "depth": 1},
        "head": "regression",
    }
    spec = ModelSpec.parse(spec_dict)
    assert spec.structure.virtual_node is True
    assert spec.structure.aggregation == "attention"

    model = build_hybrid_model(spec, task_kind="scalar", out_dim=1)
    smiles = ["c1ccccc1", "CCO"]
    graphs = [mol_to_graph(Chem.MolFromSmiles(s)) for s in smiles]
    bg = collate_mol_graphs(graphs)

    out = model({"structure": bg})
    assert "prediction" in out
    assert out["prediction"].shape in {(2,), (2, 1)}

def test_focal_loss_formulation():

    logits = torch.tensor([2.0, -3.0, -4.0, -2.5, -5.0, -3.2, -4.1, -2.9, -3.5, -4.8], requires_grad=True)
    targets = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    loss = _focal_loss(logits, targets, gamma=2.0, alpha=0.25).mean()
    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))

def test_pairwise_ranking_loss():

    preds = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    targets = torch.tensor([1.0, 2.5, 3.1, 4.5])
    base_loss = torch.nn.functional.mse_loss(preds, targets)

    total_loss = _pairwise_ranking_loss(preds, targets, base_loss, margin=0.1, rank_weight=0.5)
    assert total_loss.item() >= base_loss.item()
    total_loss.backward()
    assert preds.grad is not None

def test_spearman_ranking_metric():
    actual = [1.0, 2.0, 3.0, 4.0, 5.0]
    predicted = [1.2, 2.3, 2.9, 4.1, 5.2]
    res = regression_metrics(actual, predicted, metrics=("mae", "rmse", "r2", "spearman"))
    assert res["count"] == 5
    assert res["spearman"] == pytest.approx(1.0, abs=1e-3)

def test_advisor_v2_tanimoto_and_breakthrough_recommendations():
    advisor = DatasetAdvisor()

    smiles = [
        "c1ccccc1",
        "c1ccccc1O",
        "CC(=O)Oc1ccccc1C(=O)O",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "CCN(CC)CC",
        "C1CCOC1",
        "CC#N",
    ] * 20

    targets = {"activity": np.linspace(0.5, 9.5, len(smiles)).tolist()}
    report = advisor.profile_dataset(smiles, targets, dataset_name="diverse_assay")

    chem = report.chemical_profile
    assert chem.mean_tanimoto_similarity > 0.0
    assert chem.tanimoto_diversity_score > 0.0

    rec = report.recommendation

    assert "spearman" in rec.recommended_metrics

    assert rec.config_dict["model"]["structure"]["virtual_node"] is True
    assert rec.config_dict["model"]["structure"]["aggregation"] == "attention"

    console_report = advisor.format_console_report(report)
    assert "Mean Tanimoto Sim" in console_report
    assert "Virtual Supernode: ON" in console_report
