from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
import pytest
from chemistry_framework.analysis import (
    AdvisorRecommendation,
    ChemicalProfile,
    DatasetAdvisor,
    DatasetReport,
    TabularFeatureProfile,
    TargetProfile,
)
from chemistry_framework.cli import main
from chemistry_framework.config.schema import ExperimentConfig

def test_advisor_profiles_classification_dataset():
    advisor = DatasetAdvisor()
    smiles = [
        "c1ccccc1",
        "c1ccccc1O",
        "c1ccccc1N",
        "CC(=O)Oc1ccccc1C(=O)O",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    ] * 20

    targets = {
        "active": [1.0] * 20 + [0.0] * 100
    }

    report = advisor.profile_dataset(smiles, targets, dataset_name="toy_assay")
    assert report.dataset_name == "toy_assay"
    assert report.chemical_profile.total_records == 120
    assert report.chemical_profile.valid_smiles_count == 120
    assert report.chemical_profile.invalid_smiles_count == 0
    assert report.chemical_profile.unique_smiles_count == 6
    assert report.chemical_profile.duplicate_count == 114
    assert len(report.chemical_profile.property_stats) >= 5
    target_prof = report.target_profiles["active"]
    assert target_prof.task_kind == "binary_classification"
    assert target_prof.class_counts == {"0": 100, "1": 20}
    assert target_prof.class_imbalance_ratio == pytest.approx(5.0)
    rec = report.recommendation
    assert rec.recommended_loss == "bce"
    assert rec.recommended_head == "classification"
    assert "auroc" in rec.recommended_metrics
    assert "auprc" in rec.recommended_metrics
    assert rec.config_dict["tasks"]["active"]["kind"] == "binary_classification"

def test_advisor_profiles_regression_with_skew_and_outliers():
    advisor = DatasetAdvisor()
    smiles = ["CCCC", "CCCCC", "CCCCCC", "c1ccccc1", "c1ccccc1CC"] * 40
    np.random.seed(42)
    reg_vals = np.random.exponential(scale=2.0, size=200).tolist()
    reg_vals[0] = 50.0
    reg_vals[1] = 45.0
    report = advisor.profile_dataset(smiles, {"solubility": reg_vals}, dataset_name="toy_solubility")
    target_prof = report.target_profiles["solubility"]
    assert target_prof.task_kind == "scalar_regression"
    assert target_prof.skewness is not None and target_prof.skewness > 1.5
    assert target_prof.outliers_count >= 2
    rec = report.recommendation
    assert rec.recommended_loss == "huber"
    assert "mae" in rec.recommended_metrics
    assert "rmse" in rec.recommended_metrics
    risk_titles = [r.title for r in rec.risks]
    assert any("Skewness" in t or "Outliers" in t for t in risk_titles)

def test_advisor_profiles_tabular_features():
    advisor = DatasetAdvisor()
    smiles = ["c1ccccc1", "c1ccccc1O", "c1ccccc1N", "CC(=O)O", "CCO", "CCCO"] * 20
    n = len(smiles)
    f_normal = (np.sin(np.linspace(0.1, 10.0, n)) * 20.0 + 30.0).tolist()
    f_const = [5.0] * n
    f_missing = [1.0 if i % 4 == 0 else np.nan for i in range(n)]
    f_collin1 = np.arange(n, dtype=float).tolist()
    f_collin2 = (np.arange(n, dtype=float) * 2.0 + 1.0).tolist()
    tabular = {
        "normal_feat": f_normal,
        "const_feat": f_const,
        "missing_feat": f_missing,
        "collin_a": f_collin1,
        "collin_b": f_collin2,
    }
    targets = {"activity": [1.0 if i % 2 == 0 else 0.0 for i in range(n)]}
    report = advisor.profile_dataset(smiles, targets, tabular_features=tabular, dataset_name="feat_test")
    assert report.tabular_profile is not None
    tab = report.tabular_profile
    assert tab.feature_count == 5
    assert "const_feat" in tab.constant_columns
    assert "missing_feat" in tab.high_missing_columns
    assert len(tab.collinear_pairs) >= 1
    collin_names = {tab.collinear_pairs[0][0], tab.collinear_pairs[0][1]}
    assert collin_names == {"collin_a", "collin_b"}
    assert "const_feat" not in tab.recommended_keep_features
    assert "normal_feat" in tab.recommended_keep_features
    rec = report.recommendation
    assert rec.feature_imputation == "median"
    assert rec.feature_scaling == "standard"
    assert rec.recommended_model_type == "hybrid"

def test_advisor_detects_invalid_smiles_and_drops_health_score():
    advisor = DatasetAdvisor()
    smiles = [
        "c1ccccc1",
        "NOT_A_VALID_SMILES",
        "",
        "INVALID_CHEMICAL",
        "CC(=O)O",
    ] * 20
    targets = {"val": [1.0] * 100}
    report = advisor.profile_dataset(smiles, targets, dataset_name="bad_data")
    chem = report.chemical_profile
    assert chem.invalid_smiles_count == 60
    assert chem.invalid_fraction == pytest.approx(0.60)
    assert len(chem.invalid_examples) > 0
    rec = report.recommendation
    assert rec.health_score < 70
    assert any(r.severity == "HIGH" and "Invalid" in r.title for r in rec.risks)

def test_advisor_formats_and_exports_reports(tmp_path: Path):
    advisor = DatasetAdvisor()
    smiles = ["c1ccccc1", "c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O", "CCO"] * 10
    targets = {"ic50": np.linspace(1.0, 10.0, len(smiles)).tolist()}
    features = {"tpsa": np.linspace(10.0, 90.0, len(smiles)).tolist()}
    report = advisor.profile_dataset(smiles, targets, tabular_features=features, dataset_name="export_test")
    console_str = advisor.format_console_report(report)
    assert "CHEMISTRY FRAMEWORK: DATASET HEALTH & ADVISORY REPORT" in console_str
    assert "export_test" in console_str
    assert "PRESCRIPTIVE TRAINING RECOMMENDATIONS" in console_str
    md_str = advisor.format_markdown_report(report)
    assert "# Chemical Dataset Analysis & Advisory Report: `export_test`" in md_str
    assert "## 1. Executive Summary & Health Check" in md_str
    assert "```yaml" in md_str
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"
    cfg_path = tmp_path / "recommended.yaml"
    assert report.save_json(json_path).is_file()
    assert report.save_markdown(md_path).is_file()
    assert report.save_config(cfg_path).is_file()
    saved_data = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved_data["dataset_name"] == "export_test"
    assert "chemical_profile" in saved_data
    assert "prescriptions" in saved_data

def test_advisor_profile_csv_and_cli_inspect(tmp_path: Path):
    csv_file = tmp_path / "sample_dataset.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", "activity", "mw_feature", "dummy_const"])
        writer.writerow(["c1ccccc1", "1", "78.11", "1.0"])
        writer.writerow(["c1ccccc1O", "0", "94.11", "1.0"])
        writer.writerow(["c1ccccc1N", "1", "93.13", "1.0"])
        writer.writerow(["CCO", "0", "46.07", "1.0"])
        writer.writerow(["CC(=O)O", "1", "60.05", "1.0"])
    advisor = DatasetAdvisor()
    report = advisor.profile_csv(csv_file, smiles_column="smiles", target_columns=["activity"])
    assert report.chemical_profile.valid_smiles_count == 5
    assert report.tabular_profile is not None
    assert "dummy_const" in report.tabular_profile.constant_columns
    out_dir = tmp_path / "analysis_out"
    rec_cfg = tmp_path / "rec.yaml"
    ret = main([
        "inspect",
        str(csv_file),
        "--output-dir",
        str(out_dir),
        "--generate-config",
        str(rec_cfg),
    ])
    assert ret == 0
    assert (out_dir / "sample_dataset-health-report.json").is_file()
    assert (out_dir / "sample_dataset-advisory-report.md").is_file()
    assert rec_cfg.is_file()

@pytest.mark.parametrize("values", [
    [0.1 * i for i in range(120)],
    [i % 2 for i in range(120)],
])

def test_advisor_generated_config_passes_schema(values):
    smiles = ["CCO", "CCCO", "c1ccccc1", "c1ccccc1O", "CC(=O)O", "CCN"] * 20
    report = DatasetAdvisor().profile_dataset(smiles, {"y": values}, dataset_name="schema_check")
    config = ExperimentConfig.parse(report.recommendation.config_dict)
    assert config.tasks["y"]["kind"] in {"scalar", "binary_classification"}
