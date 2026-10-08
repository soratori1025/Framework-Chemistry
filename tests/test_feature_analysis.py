from __future__ import annotations

import json
import struct
import zlib

import numpy as np
import pytest

from chemistry_framework.feature_analysis import (
    ModelContribution,
    PriorCoefficientEvidence,
    analyze_feature_evidence,
    leave_group_out_importance,
    permutation_group_importance,
    summarize_prior_coefficients,
)
from chemistry_framework.features import FeatureMatrix, FeatureRegistry, FeatureSpec
from chemistry_framework.splits import DatasetSplit


def _feature_fixture():
    registry = FeatureRegistry()
    registry.register(
        FeatureSpec(
            "composition.carbon",
            lambda _: 0.0,
            "composition",
            metadata={"element": "C"},
            group="composition",
            source="elements",
            definition="Count of carbon atoms",
            interpretation="Molecular carbon content",
        )
    )
    registry.register(
        FeatureSpec(
            "functional.ester",
            lambda _: 0.0,
            "functional_group",
            group="functional",
            source="smarts",
            definition="SMARTS count: C(=O)O",
        )
    )
    matrix = FeatureMatrix(
        ("composition.carbon", "functional.ester"),
        np.asarray([[0, 0], [1, 0], [2, 0], [3, 0], [4, 1], [5, 2]], dtype=float),
    )
    split = DatasetSplit(
        np.asarray([0, 1, 2]),
        np.asarray([3]),
        np.asarray([4, 5]),
    )
    targets = {"property": np.asarray([0, 1, 2, 3, 4, 5], dtype=float)}
    return registry, matrix, split, targets


def test_feature_evidence_reports_provenance_association_and_ood(tmp_path) -> None:
    registry, matrix, split, targets = _feature_fixture()
    report = analyze_feature_evidence(
        matrix,
        split,
        targets,
        registry,
        feature_groups={
            "composition": ("composition.carbon",),
            "functional": ("functional.ester",),
        },
        support_threshold=2,
    )
    by_name = {item.name: item for item in report.features}
    carbon = by_name["composition.carbon"]
    assert carbon.group == "composition"
    assert carbon.source == "elements"
    assert carbon.definition == "Count of carbon atoms"
    assert carbon.interpretation == "Molecular carbon content"
    assert carbon.target_associations["property"] == pytest.approx(1.0)
    assert by_name["functional.ester"].train_support == 0
    assert by_name["functional.ester"].evidence_class == "ood_unsupported"

    paths = report.write_artifacts(tmp_path, "fixture")
    assert all(path.is_file() for path in paths.values())
    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert "_train_matrix" not in data
    assert data["features"][0]["provenance"]["metadata"] == {"element": "C"}
    html_report = paths["html"].read_text(encoding="utf-8")
    assert paths["heatmap"].name in html_report
    png = paths["heatmap"].read_bytes()
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    offset = 8
    image_data = bytearray()
    while offset < len(png):
        chunk_size = struct.unpack(">I", png[offset : offset + 4])[0]
        chunk_type = png[offset + 4 : offset + 8]
        chunk_data = png[offset + 8 : offset + 8 + chunk_size]
        expected_crc = struct.unpack(
            ">I", png[offset + 8 + chunk_size : offset + 12 + chunk_size]
        )[0]
        assert zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF == expected_crc
        if chunk_type == b"IDAT":
            image_data.extend(chunk_data)
        offset += chunk_size + 12
        if chunk_type == b"IEND":
            break
    assert len(zlib.decompress(image_data)) > 0


def test_fixed_features_are_not_described_as_learned_evidence() -> None:
    registry, matrix, split, targets = _feature_fixture()
    report = analyze_feature_evidence(
        matrix,
        split,
        targets,
        registry,
        fixed_coefficients={"composition.carbon": 2.5},
    )
    carbon = next(item for item in report.features if item.name == "composition.carbon")
    assert carbon.evidence_class == "fixed_analytical"
    assert carbon.prior_evidence is not None
    assert carbon.prior_evidence["mean"] == 2.5
    assert carbon.prior_evidence["count"] == 0


def test_supported_label_requires_prior_and_model_evidence() -> None:
    registry = FeatureRegistry()
    registry.register(
        FeatureSpec("f", lambda _: 0.0, "custom", group="group")
    )
    matrix = FeatureMatrix(
        ("f",),
        np.asarray([[0], [1], [2], [1], [2], [1]], dtype=float),
    )
    split = DatasetSplit(
        np.asarray([0, 1, 2]),
        np.asarray([3]),
        np.asarray([4, 5]),
    )
    coefficient = PriorCoefficientEvidence(
        feature="f",
        count=5,
        mean=1.0,
        standard_deviation=0.1,
        minimum=0.9,
        maximum=1.1,
        sign_consistency=1.0,
        bootstrap_95_ci=(0.9, 1.1),
    )
    contribution = ModelContribution(
        group="group",
        method="leave_group_out",
        metric="auroc",
        baseline_score=0.9,
        comparison_scores=(0.8, 0.81),
        contribution_deltas=(0.1, 0.09),
        mean_delta=0.095,
        standard_deviation=0.007,
        positive_fraction=1.0,
    )
    report = analyze_feature_evidence(
        matrix,
        split,
        {"target": [0, 1, 2, 1, 2, 1]},
        registry,
        prior_evidence={"f": coefficient},
        model_contributions=(contribution,),
    )
    assert report.features[0].evidence_class == "supported"
    assert "not causal" in report.interpretation_note


def test_prior_coefficient_stability_and_fixed_values() -> None:
    evidence = summarize_prior_coefficients(
        [
            {"stable": 1.0, "unstable": 1.0},
            {"stable": 1.1, "unstable": -1.0},
            {"stable": 0.9, "unstable": 0.1},
            {"stable": 1.0, "unstable": -0.5},
        ],
        fixed_coefficients={"fixed": 3.0},
        bootstrap_samples=500,
    )
    assert evidence["stable"].sign_consistency == 1.0
    assert evidence["unstable"].sign_consistency == 0.5
    assert evidence["stable"].bootstrap_95_ci[0] <= evidence["stable"].mean
    assert evidence["fixed"].count == 0
    assert evidence["fixed"].bootstrap_95_ci == (3.0, 3.0)


def test_group_permutation_importance_preserves_joint_group_rows() -> None:
    features = np.asarray([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=float)
    targets = features[:, 0]

    def predictor(values):
        return values[:, 0]

    result = permutation_group_importance(
        predictor,
        features,
        targets,
        {"both": (0, 1)},
        lambda actual, predicted: -float(np.mean((actual - predicted) ** 2)),
        metric_name="negative_mse",
        repeats=4,
        seed=4,
    )
    assert result["both"].method == "group_permutation"
    assert result["both"].mean_delta > 0
    assert len(result["both"].contribution_deltas) == 4


def test_leave_group_out_retrains_for_each_feature_subset() -> None:
    targets = np.asarray([1.0, 2.0])
    calls: list[tuple[int, ...]] = []

    def fit_predict(columns):
        calls.append(columns)
        if columns == (0,):
            return np.asarray([0.0, 0.0])
        return targets

    result = leave_group_out_importance(
        fit_predict,
        ("composition", "fragment"),
        {"fragment": ("fragment",)},
        targets,
        lambda actual, predicted: -float(np.mean((actual - predicted) ** 2)),
        metric_name="negative_mse",
    )
    assert calls == [(0, 1), (0,)]
    assert result["fragment"].method == "leave_group_out"
    assert result["fragment"].mean_delta == pytest.approx(2.5)
