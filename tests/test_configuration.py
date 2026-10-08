from __future__ import annotations

import csv
from dataclasses import asdict
import sys
from types import ModuleType

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from chemistry_framework.analysis import analyze_prediction_molecules
from chemistry_framework.config import ExperimentConfig
from chemistry_framework.config.schema import FeatureConfig
from chemistry_framework.data import make_tensor_dataloaders
from chemistry_framework.encoders import AtomSMARTSFlags, BondSMARTSFlags
from chemistry_framework.evaluation import evaluate_model, load_model_checkpoint
from chemistry_framework.experiment import fit_experiment_prior, prepare_experiment
from chemistry_framework.features import FeatureCache, MoleculeFeatureBlock, build_feature_registry
from chemistry_framework.priors import fit_linear_prior
from chemistry_framework.splits import split_indices
from chemistry_framework.training import train_model


def _write_dataset(path) -> None:
    smiles = [
        "C", "CC", "CCC", "CCCC", "CCO", "CCCO", "CCN", "CCCN",
        "CC(=O)OC", "CCC(=O)OC", "CC(=O)OCC", "CCCC(=O)OC",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["smiles", "h", "s", "active", "tox_a", "tox_b"],
        )
        writer.writeheader()
        for index, value in enumerate(smiles):
            writer.writerow(
                {
                    "smiles": value,
                    "h": index * 2.0 + 1,
                    "s": index * -1.5 + 4,
                    "active": index % 2,
                    "tox_a": index % 2,
                    "tox_b": "" if index == 0 else (index // 2) % 2,
                }
            )


def _config(tmp_path, split=None) -> ExperimentConfig:
    dataset_path = tmp_path / "molecules.csv"
    _write_dataset(dataset_path)
    data = {
        "name": "fixture",
        "dataset": {
            "path": dataset_path.name,
            "smiles_column": "smiles",
            "targets": {
                "enthalpy": {"column": "h", "unit": "kJ/mol"},
                "entropy": {"column": "s", "unit": "J/(mol K)"},
            },
        },
        "features": {
            "composition": {"type": "elements", "elements": ["C", "O"]},
            "bonds": {"type": "bond_pairs", "elements": ["C", "O"], "bond_orders": ["1"]},
            "ester": {"type": "smarts", "pattern": "[CX3](=O)[OX2][#6]"},
        },
        "priors": {"enthalpy": {"use": ["composition"], "fit": "ridge"}},
        "routes": {"enthalpy": ["ester"]},
        "tasks": {
            "enthalpy": {
                "target": "enthalpy",
                "kind": "scalar",
                "prior": "enthalpy",
                "route": "enthalpy",
            }
        },
        "split": split or {"method": "random", "fractions": [0.6, 0.2, 0.2], "seed": 19},
        "training": {"epochs": 5, "batch_size": 4, "learning_rate": 0.01, "patience": 2},
        "evaluation": {"metrics": ["mae", "rmse", "r2"]},
        "analysis": {
            "enabled": True,
            "support_threshold": 2,
            "correlation_threshold": 0.95,
            "bins": 5,
            "output_dir": "analysis",
        },
        "cache": {"enabled": True, "directory": ".cache/features"},
        "graph": {
            "aggregation": "sum",
            "atom_smarts_flags": [{"name": "carbonyl", "pattern": "[CX3]=O"}],
            "bond_smarts_flags": [{"name": "carbonyl_bond", "pattern": "[CX3]=O"}],
        },
    }
    return ExperimentConfig.parse(data, source_path=tmp_path / "experiment.yaml")

def _classification_config(tmp_path, *, multilabel=False) -> ExperimentConfig:
    config = asdict(_config(tmp_path))
    config.pop("source_path")
    config["features"] = {
        name: {"type": feature["type"], **feature["options"]}
        for name, feature in config["features"].items()
    }
    config["routes"] = {
        name: list(feature_names)
        for name, feature_names in config["routes"].items()
    }
    config["priors"] = {
        name: {
            **prior,
            "use": list(prior["use"]),
        }
        for name, prior in config["priors"].items()
    }
    config["evaluation"]["metrics"] = list(config["evaluation"]["metrics"])
    config["dataset"]["targets"] = {
        name: {
            "column": target["columns"][0],
            "unit": target["unit"],
            "kind": target["kind"],
        }
        for name, target in config["dataset"]["targets"].items()
    }
    if multilabel:
        config["dataset"]["targets"] = {
            "activity": {
                "columns": ["tox_a", "tox_b"],
                "kind": "vector",
            }
        }
        config["tasks"] = {
            "activity": {
                "target": "activity",
                "kind": "multilabel_classification",
            }
        }
    else:
        config["dataset"]["targets"] = {
            "activity": {"column": "active"},
        }
        config["tasks"] = {
            "activity": {
                "target": "activity",
                "kind": "binary_classification",
            }
        }
    config["training"]["loss"] = "bce"
    return ExperimentConfig.parse(config, source_path=tmp_path / "experiment.yaml")


def test_configured_experiment_loads_splits_caches_and_writes_analysis(tmp_path) -> None:
    experiment = prepare_experiment(_config(tmp_path))
    assert len(experiment.dataset) == 12
    assert sum(map(len, (experiment.split.train, experiment.split.validation, experiment.split.test))) == 12
    assert experiment.features.values.shape[0] == 12
    assert experiment.feature_blocks["composition"] == ("composition.C", "composition.O")
    assert experiment.prior_features["enthalpy"] == ("composition.C", "composition.O")
    assert experiment.route_features["enthalpy"] == ("ester",)
    assert experiment.analysis is not None
    assert experiment.feature_evidence is not None
    assert experiment.analysis.split_sizes["train"] == len(experiment.split.train)
    assert (tmp_path / "analysis" / "fixture-analysis.json").is_file()
    assert (tmp_path / "analysis" / "fixture-feature-evidence.csv").is_file()
    assert (tmp_path / "analysis" / "fixture-feature-report.html").is_file()
    assert (tmp_path / "analysis" / "fixture-correlation-heatmap.png").is_file()
    cached_files = list((tmp_path / ".cache" / "features").glob("*.npz"))
    assert len(cached_files) == len(experiment.dataset)


def test_feature_cache_is_keyed_by_configuration_and_molecule(tmp_path) -> None:
    from rdkit import Chem

    registry, resolved = build_feature_registry(
        _config(tmp_path).features,
        ["CCO"],
        [0],
    )
    block = MoleculeFeatureBlock(registry, resolved["composition"])
    cache = FeatureCache(tmp_path / "cache", "schema-a")
    first = cache.transform(block, ["CCO"])
    second = cache.transform(block, [Chem.MolFromSmiles("OCC")])
    np.testing.assert_array_equal(first.values, second.values)
    FeatureCache(tmp_path / "cache", "schema-b").transform(block, ["CCO"])
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 2


def test_feature_annotations_are_kept_as_provenance(tmp_path) -> None:
    feature = FeatureConfig.parse(
        {
            "type": "smarts",
            "pattern": "[CX3](=O)[OX2][#6]",
            "group": "functional_groups",
            "description": "Ester fragment count",
            "interpretation": "Counts ester motifs",
            "source": "smarts",
            "definition": "Ester SMARTS query",
        },
        "ester",
    )
    registry, _ = build_feature_registry({"ester": feature}, ["CC(=O)OC"], [0])
    spec = registry.get("ester")
    assert spec.group == "functional_groups"
    assert spec.interpretation == "Counts ester motifs"
    assert spec.definition == "Ester SMARTS query"


def test_implicit_hydrogens_only_contribute_to_single_bond_pairs(tmp_path) -> None:
    config = FeatureConfig.parse(
        {"type": "bond_pairs", "elements": ["H", "C"], "bond_orders": ["1", "2", "3", "a"]},
        "bonds",
    )
    registry, blocks = build_feature_registry({"bonds": config}, ["C=C"], [0])
    matrix = MoleculeFeatureBlock(registry, blocks["bonds"]).transform(["C=C"])
    values = dict(zip(matrix.names, matrix.values[0], strict=True))
    assert values["bonds.HC1"] == 4.0
    assert values["bonds.HC2"] == 0.0
    assert values["bonds.HC3"] == 0.0
    assert values["bonds.HCa"] == 0.0


def test_smarts_holdout_split_keeps_all_matching_molecules_in_test() -> None:
    smiles = ["CC", "CCC", "CCO", "CCCO", "CC(=O)OC", "CCC(=O)OC", "CCN", "CCCN", "C", "CO"]
    from chemistry_framework.config import SplitConfig

    split = split_indices(
        smiles,
        SplitConfig(
            method="smarts",
            test_smarts="[CX3](=O)[OX2][#6]",
            fractions=(0.7, 0.15, 0.15),
            seed=3,
        ),
    )
    assert len(split.test) == 2
    assert len(split.train) + len(split.validation) + len(split.test) == len(smiles)


def test_graph_smarts_flags_are_per_atom_and_per_bond() -> None:
    from rdkit import Chem

    molecule = Chem.MolFromSmiles("CC(=O)OC")
    atom_flags = AtomSMARTSFlags.from_config(
        [{"name": "carbonyl", "pattern": "[CX3]=O"}]
    )
    bond_flags = BondSMARTSFlags.from_config(
        [{"name": "carbonyl", "pattern": "[CX3]=O"}]
    )
    atom_values = atom_flags(molecule)
    bond_values = bond_flags(molecule)
    assert atom_values.shape == (molecule.GetNumAtoms(), 1)
    assert atom_values.sum() == 2
    assert bond_values.shape == (molecule.GetNumBonds(), 1)
    assert bond_values.sum() == 1


def test_fixed_prior_coefficients_remain_fixed() -> None:
    x = np.array([[0, 1], [1, 1], [2, 1], [3, 1]], dtype=float)
    y = 2 * x[:, 0] + 3 * x[:, 1] + 5
    result = fit_linear_prior(
        x,
        y,
        ["slope", "known"],
        fixed={"known": 3.0},
        method="ridge",
    )
    assert result.train_count == 4
    assert result.prior.coefficients[1, 0].item() == 3.0
    assert result.residual_std < 0.01


def test_linear_prior_supports_multi_output_targets() -> None:
    x = np.arange(8, dtype=float).reshape(-1, 1)
    y = np.column_stack((2 * x[:, 0] + 1, -3 * x[:, 0] + 4))
    result = fit_linear_prior(x, y, ["composition.C"], method="least_squares")
    prediction = result.prior(torch.as_tensor(x, dtype=torch.float32)).detach().numpy()
    np.testing.assert_allclose(prediction, y, atol=1e-5)
    assert result.prior.output_dim == 2


def test_prepared_prior_uses_training_rows_and_external_analysis(tmp_path) -> None:
    experiment = prepare_experiment(_config(tmp_path))
    fit = fit_experiment_prior(experiment, "enthalpy")
    assert fit.train_count == len(experiment.split.train)
    report = analyze_prediction_molecules(experiment, ["CCCCCCCCCCCC"])
    carbon = next(item for item in report.features if item.name == "composition.C")
    assert carbon.out_of_range_count == 1


def test_tensor_loaders_apply_task_descriptor_route(tmp_path) -> None:
    experiment = prepare_experiment(_config(tmp_path))
    loaders = make_tensor_dataloaders(experiment, "enthalpy")
    assert loaders.train.dataset.tensors[0].shape[1] == 1
    assert experiment.features.values.shape[1] > 1

def test_classification_config_and_loader_support_binary_and_missing_multilabels(tmp_path) -> None:
    from chemistry_framework.data import make_tensor_dataloaders

    binary_experiment = prepare_experiment(_classification_config(tmp_path))
    binary_loaders = make_tensor_dataloaders(binary_experiment, "activity")
    assert binary_experiment.config.tasks["activity"]["kind"] == "binary_classification"
    assert binary_loaders.train.dataset.tensors[1].ndim == 1

    multilabel_experiment = prepare_experiment(
        _classification_config(tmp_path, multilabel=True)
    )
    multilabel_loaders = make_tensor_dataloaders(multilabel_experiment, "activity")
    train_targets = multilabel_loaders.train.dataset.tensors[1]
    assert train_targets.ndim == 2
    assert torch.isnan(train_targets).any()

def test_tdc_adapter_loads_configured_columns_without_installing_tdc(monkeypatch) -> None:
    from chemistry_framework.data import load_tdc_dataset
    from chemistry_framework.config import DatasetConfig

    class Frame:
        columns = ["Drug", "Y"]

        def __getitem__(self, column):
            return np.asarray({"Drug": ["CC", "CCO"], "Y": [0, 1]}[column])

    class ADME:
        def __init__(self, name):
            assert name == "HIA_Hou"

        def get_data(self):
            return Frame()

    tdc_package = ModuleType("tdc")
    tdc_package.__path__ = []
    tdc_single_pred = ModuleType("tdc.single_pred")
    tdc_single_pred.ADME = ADME
    tdc_single_pred.Tox = ADME
    monkeypatch.setitem(sys.modules, "tdc", tdc_package)
    monkeypatch.setitem(sys.modules, "tdc.single_pred", tdc_single_pred)
    config = DatasetConfig.parse(
        {
            "provider": "tdc_adme",
            "name": "HIA_Hou",
            "targets": {"activity": {"column": "Y"}},
        }
    )

    loaded = load_tdc_dataset(config)
    assert loaded.smiles == ("CC", "CCO")
    assert loaded.targets["activity"] == (0.0, 1.0)
    assert loaded.source_path is None


def test_training_and_evaluation_work_with_configured_settings(tmp_path) -> None:
    x = torch.linspace(-1.0, 1.0, 20).reshape(-1, 1)
    y = 3 * x.squeeze(1) - 2
    train_loader = DataLoader(TensorDataset(x[:14], y[:14]), batch_size=4, shuffle=False)
    validation_loader = DataLoader(TensorDataset(x[14:], y[14:]), batch_size=3)
    model = torch.nn.Linear(1, 1)
    from chemistry_framework.config import TrainingConfig

    result = train_model(
        model,
        train_loader,
        validation_loader,
        TrainingConfig(epochs=100, batch_size=4, learning_rate=0.03, patience=15),
        checkpoint_path=tmp_path / "best.pt",
    )
    metrics, actual, predicted = evaluate_model(
        result.model, validation_loader, ("mae", "rmse", "r2"), device="auto"
    )
    restored = load_model_checkpoint(torch.nn.Linear(1, 1), tmp_path / "best.pt")
    restored_metrics, _, _ = evaluate_model(
        restored, validation_loader, ("mae", "rmse", "r2")
    )
    assert result.best_epoch > 0
    assert result.checkpoint_path == tmp_path / "best.pt"
    assert actual.shape == predicted.shape == (6,)
    assert metrics["mae"] < 3.0
    assert restored_metrics["mae"] == pytest.approx(metrics["mae"])

def test_binary_training_and_classification_metrics_support_missing_labels() -> None:
    from chemistry_framework.config import TrainingConfig
    from chemistry_framework.evaluation import evaluate_model

    inputs = torch.tensor([[-2.0], [-1.0], [1.0], [2.0], [3.0]])
    targets = torch.tensor([0.0, 0.0, 1.0, 1.0, float("nan")])
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    result = train_model(
        torch.nn.Linear(1, 1),
        loader,
        loader,
        TrainingConfig(
            epochs=5,
            batch_size=2,
            learning_rate=0.03,
            loss="bce",
            patience=2,
        ),
    )
    metrics, actual, predicted = evaluate_model(
        torch.nn.Sequential(torch.nn.Identity()),
        DataLoader(
            TensorDataset(
                torch.tensor([[-2.0], [-1.0], [1.0], [2.0], [0.0]]),
                targets,
            ),
            batch_size=2,
        ),
        ("auroc", "auprc", "accuracy"),
        task_kind="binary_classification",
    )
    assert result.best_epoch > 0
    assert metrics["count"] == 4
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["auprc"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert actual.shape == predicted.shape == (5,)


def test_auprc_groups_examples_with_tied_scores() -> None:
    from chemistry_framework.evaluation import classification_metrics

    metrics = classification_metrics(
        [1, 0, 1, 0],
        [0.5, 0.5, 0.5, 0.1],
        ("auprc",),
    )

    assert metrics["auprc"] == pytest.approx(2 / 3)
