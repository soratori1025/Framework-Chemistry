"""Model-independent chemical feature evidence and report artifacts."""
from __future__ import annotations

import csv
import html
import json
import struct
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..diagnostics import diagnose_features
from ..features import FeatureMatrix, FeatureRegistry
from ..splits import DatasetSplit
from .model_evidence import ModelContribution, PriorCoefficientEvidence


@dataclass(frozen=True)
class FeatureEvidence:
    name: str
    group: str
    domain: str
    source: str
    definition: str
    description: str | None
    interpretation: str | None
    provenance: dict[str, object]
    train_support: int
    train_support_fraction: float
    missing_count: int
    train_mean: float
    train_std: float
    train_min: float
    train_max: float
    test_min: float
    test_max: float
    out_of_range_count: int
    out_of_range_fraction: float
    constant_in_train: bool
    target_associations: dict[str, float | None]
    collinear_features: tuple[str, ...]
    prior_evidence: dict[str, object] | None
    model_evidence: tuple[dict[str, object], ...]
    evidence_class: str


@dataclass(frozen=True)
class FeatureEvidenceReport:
    dataset_rows: int
    train_rows: int
    test_rows: int
    correlation_threshold: float
    support_threshold: int
    features: tuple[FeatureEvidence, ...]
    model_contributions: tuple[ModelContribution, ...]
    _train_matrix: np.ndarray | None = field(default=None, repr=False, compare=False)
    interpretation_note: str = (
        "Associations and model contributions are empirical evidence under this "
        "dataset, split, and protocol; they are not causal effects."
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_rows": self.dataset_rows,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "correlation_threshold": self.correlation_threshold,
            "support_threshold": self.support_threshold,
            "features": [asdict(item) for item in self.features],
            "model_contributions": [
                asdict(item) for item in self.model_contributions
            ],
            "interpretation_note": self.interpretation_note,
        }

    def write_artifacts(self, output_dir: str | Path, stem: str) -> dict[str, Path]:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "json": directory / f"{stem}-feature-evidence.json",
            "csv": directory / f"{stem}-feature-evidence.csv",
            "html": directory / f"{stem}-feature-report.html",
            "heatmap": directory / f"{stem}-correlation-heatmap.png",
        }
        paths["json"].write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        self._write_csv(paths["csv"])
        paths["html"].write_text(
            self._to_html(paths["heatmap"].name), encoding="utf-8"
        )
        paths["heatmap"].write_bytes(self._to_heatmap_png())
        return paths

    def _write_csv(self, path: Path) -> None:
        columns = [
            "name",
            "group",
            "domain",
            "source",
            "definition",
            "description",
            "interpretation",
            "provenance",
            "train_support",
            "train_support_fraction",
            "missing_count",
            "train_mean",
            "train_std",
            "train_min",
            "train_max",
            "test_min",
            "test_max",
            "out_of_range_count",
            "out_of_range_fraction",
            "constant_in_train",
            "target_associations",
            "collinear_features",
            "prior_evidence",
            "model_evidence",
            "evidence_class",
        ]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for item in self.features:
                row = asdict(item)
                for key in (
                    "provenance",
                    "target_associations",
                    "collinear_features",
                    "prior_evidence",
                    "model_evidence",
                ):
                    row[key] = json.dumps(row[key], sort_keys=True, allow_nan=False)
                writer.writerow(row)

    def _to_html(self, heatmap_name: str) -> str:
        headers = (
            "Feature",
            "Group",
            "Definition",
            "Train support",
            "Train range",
            "Test range",
            "Out of range",
            "Target association",
            "Redundant with",
            "Evidence class",
        )
        rows = []
        for feature in self.features:
            association = "; ".join(
                f"{name}: {value:.3g}" if value is not None else f"{name}: n/a"
                for name, value in feature.target_associations.items()
            )
            row = (
                feature.name,
                feature.group,
                feature.definition,
                f"{feature.train_support} ({feature.train_support_fraction:.1%})",
                f"[{feature.train_min:.5g}, {feature.train_max:.5g}]",
                f"[{feature.test_min:.5g}, {feature.test_max:.5g}]",
                f"{feature.out_of_range_count} ({feature.out_of_range_fraction:.1%})",
                association,
                ", ".join(feature.collinear_features),
                feature.evidence_class,
            )
            rows.append(
                "<tr>"
                + "".join(f"<td>{html.escape(str(value))}</td>" for value in row)
                + "</tr>"
            )
        header_html = "".join(f"<th>{html.escape(value)}</th>" for value in headers)
        contributions = [
            (
                f"<li>{html.escape(item.group)} — {html.escape(item.method)} "
                f"({html.escape(item.metric)}): Δ={item.mean_delta:.5g} "
                f"± {item.standard_deviation:.5g}; positive fraction "
                f"{item.positive_fraction:.1%}</li>"
            )
            for item in self.model_contributions
        ]
        contribution_html = (
            "<ul>" + "".join(contributions) + "</ul>"
            if contributions
            else "<p>Model-dependent evidence has not been supplied.</p>"
        )
        note = html.escape(self.interpretation_note)
        return (
            "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
            "<title>Chemical feature evidence</title>"
            "<style>body{font:14px sans-serif;margin:2rem}table{border-collapse:collapse}"
            "th,td{border:1px solid #bbb;padding:.35rem;text-align:left}"
            "th{position:sticky;top:0;background:#eee}td{max-width:24rem;overflow-wrap:anywhere}"
            "</style><body><h1>Chemical feature evidence</h1>"
            f"<p>{self.train_rows} training and {self.test_rows} test rows; "
            f"{len(self.features)} features.</p>"
            f"<p>{note}</p><h2>Feature scorecard</h2><table><thead><tr>{header_html}"
            "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table><h2>Model-specific group contributions</h2>"
            + contribution_html
            + '<p><img src="'
            + html.escape(heatmap_name)
            + '" alt="Feature correlation heatmap; red is positive, blue is negative"></p>'
            + "<p>Heatmap colors encode train-only Pearson correlations "
            "(red: positive, blue: negative); row and column order follows the CSV.</p>"
            "</body></html>\n"
        )

    def _to_heatmap_png(self) -> bytes:
        n_features = len(self.features)
        train = self._train_matrix
        if train is None:
            raise RuntimeError("correlation heatmap requires the training feature matrix")
        standard_deviation = train.std(axis=0)
        active = standard_deviation > 1e-12
        standardized = np.zeros_like(train)
        standardized[:, active] = (
            train[:, active] - train[:, active].mean(axis=0)
        ) / standard_deviation[active]
        correlation = standardized.T @ standardized / len(train)
        correlation[np.diag_indices(n_features)] = active.astype(float)
        colors = np.empty((n_features, n_features, 3), dtype=np.uint8)
        positive = correlation >= 0
        positive_intensity = 1.0 - np.clip(correlation, 0.0, 1.0)
        negative_intensity = 1.0 + np.clip(correlation, -1.0, 0.0)
        colors[:, :, 0] = 255
        colors[:, :, 1] = np.where(
            positive, 255 * positive_intensity, 255 * negative_intensity
        ).astype(np.uint8)
        colors[:, :, 2] = colors[:, :, 1]
        colors[:, :, 2] = np.where(positive, colors[:, :, 2], 255).astype(np.uint8)
        colors[:, :, 0] = np.where(positive, 255, 255 * negative_intensity).astype(
            np.uint8
        )
        colors = np.repeat(np.repeat(colors, 3, axis=0), 3, axis=1)
        height, width, _ = colors.shape
        scanlines = b"".join(b"\x00" + row.tobytes() for row in colors)

        def chunk(kind: bytes, data: bytes) -> bytes:
            checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)

        return (
            b"\x89PNG\r\n\x1a\n"
            + chunk(
                b"IHDR",
                struct.pack(">2I5B", width, height, 8, 2, 0, 0, 0),
            )
            + chunk(b"IDAT", zlib.compress(scanlines, level=9))
            + chunk(b"IEND", b"")
        )

def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    finite = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(finite) < 2:
        return None
    left_ranks = _average_ranks(left[finite])
    right_ranks = _average_ranks(right[finite])
    if left_ranks.std() <= 0 or right_ranks.std() <= 0:
        return None
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def _target_associations(
    feature_values: np.ndarray,
    targets: Mapping[str, Sequence[float] | Sequence[Sequence[float]]],
    train_indices: np.ndarray,
    row_count: int,
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for target_name, raw_values in targets.items():
        values = np.asarray(raw_values, dtype=np.float64)
        if values.ndim not in {1, 2} or values.shape[0] != row_count:
            raise ValueError(
                f"target {target_name!r} must have one row per feature-matrix row"
            )
        if values.ndim == 1:
            result[target_name] = _spearman(
                feature_values[train_indices], values[train_indices]
            )
        else:
            for column in range(values.shape[1]):
                result[f"{target_name}[{column}]"] = _spearman(
                    feature_values[train_indices],
                    values[train_indices, column],
                )
    return result


def analyze_feature_evidence(
    matrix: FeatureMatrix,
    split: DatasetSplit,
    targets: Mapping[str, Sequence[float] | Sequence[Sequence[float]]],
    registry: FeatureRegistry,
    *,
    feature_groups: Mapping[str, Sequence[str]] | None = None,
    support_threshold: int = 1,
    correlation_threshold: float = 0.95,
    fixed_coefficients: Mapping[str, float] | None = None,
    prior_evidence: Mapping[str, PriorCoefficientEvidence] | None = None,
    model_contributions: Sequence[ModelContribution] = (),
) -> FeatureEvidenceReport:
    """Build model-independent evidence and attach optional prior/model evidence."""
    if support_threshold < 1:
        raise ValueError("support_threshold must be positive")
    values = matrix.values
    indices = np.concatenate((split.train, split.validation, split.test))
    if (
        len(indices) != values.shape[0]
        or set(indices.tolist()) != set(range(values.shape[0]))
    ):
        raise ValueError("split must partition every row in the feature matrix exactly once")
    train = values[split.train]
    test = values[split.test]
    diagnostics = diagnose_features(
        train,
        test,
        matrix.names,
        correlation_threshold=correlation_threshold,
    )
    feature_to_group = {
        feature: group
        for group, members in (feature_groups or {}).items()
        for feature in members
    }
    unknown_group_features = set(feature_to_group) - set(matrix.names)
    if unknown_group_features:
        raise ValueError(
            f"feature groups reference unknown columns: {sorted(unknown_group_features)}"
        )
    if len(feature_to_group) != sum(
        len(members) for members in (feature_groups or {}).values()
    ):
        raise ValueError("a feature must belong to no more than one analysis group")
    collinear_with: dict[str, set[str]] = {name: set() for name in matrix.names}
    for left, right, _ in diagnostics.collinear_pairs:
        collinear_with[left].add(right)
        collinear_with[right].add(left)
    contributions_by_group: dict[str, list[ModelContribution]] = {}
    for contribution in model_contributions:
        contributions_by_group.setdefault(contribution.group, []).append(contribution)

    fixed = fixed_coefficients or {}
    prior_runs = prior_evidence or {}
    feature_items: list[FeatureEvidence] = []
    for column, diagnostic in enumerate(diagnostics.features):
        name = diagnostic.name
        spec = registry.get(name)
        group = (
            spec.group
            or feature_to_group.get(name)
            or (name.split(".", 1)[0] if "." in name else spec.domain)
        )
        support = int(np.count_nonzero(np.abs(train[:, column]) > 1e-12))
        prior = prior_runs.get(name)
        if prior is None and name in fixed:
            prior = PriorCoefficientEvidence(
                feature=name,
                count=0,
                mean=float(fixed[name]),
                standard_deviation=0.0,
                minimum=float(fixed[name]),
                maximum=float(fixed[name]),
                sign_consistency=1.0,
                bootstrap_95_ci=(float(fixed[name]), float(fixed[name])),
            )
        group_contributions = tuple(contributions_by_group.get(group, ()))
        if name != group:
            group_contributions += tuple(contributions_by_group.get(name, ()))
        if name in fixed:
            evidence_class = "fixed_analytical"
        elif support == 0 or diagnostic.out_of_range_count > 0:
            evidence_class = "ood_unsupported"
        elif support < support_threshold:
            evidence_class = "weak_uncertain"
        elif collinear_with[name]:
            evidence_class = "redundant"
        elif (
            prior is not None
            and prior.count > 1
            and abs(prior.mean) > 1e-12
            and prior.sign_consistency >= 0.8
            and (
                prior.bootstrap_95_ci[0] > 0
                or prior.bootstrap_95_ci[1] < 0
            )
            and any(
                contribution.mean_delta > 0
                and contribution.positive_fraction >= 0.8
                for contribution in group_contributions
            )
        ):
            evidence_class = "supported"
        elif group_contributions:
            evidence_class = "model_dependent"
        else:
            evidence_class = "weak_uncertain"

        provenance = {
            "domain": spec.domain,
            "source": spec.source or "unspecified",
            "applies_to": list(spec.applies_to),
            "metadata": dict(spec.metadata),
        }
        feature_items.append(
            FeatureEvidence(
                name=name,
                group=group,
                domain=spec.domain,
                source=spec.source or "unspecified",
                definition=spec.definition or json.dumps(
                    dict(spec.metadata), sort_keys=True, default=str
                ),
                description=spec.description,
                interpretation=spec.interpretation,
                provenance=provenance,
                train_support=support,
                train_support_fraction=float(support / len(train)),
                missing_count=0,
                train_mean=diagnostic.train_mean,
                train_std=diagnostic.train_std,
                train_min=diagnostic.train_min,
                train_max=diagnostic.train_max,
                test_min=diagnostic.test_min,
                test_max=diagnostic.test_max,
                out_of_range_count=diagnostic.out_of_range_count,
                out_of_range_fraction=float(
                    diagnostic.out_of_range_count / len(test)
                ),
                constant_in_train=diagnostic.constant_in_train,
                target_associations=_target_associations(
                    values[:, column], targets, split.train, values.shape[0]
                ),
                collinear_features=tuple(sorted(collinear_with[name])),
                prior_evidence=asdict(prior) if prior is not None else None,
                model_evidence=tuple(
                    asdict(item) for item in group_contributions
                ),
                evidence_class=evidence_class,
            )
        )
    return FeatureEvidenceReport(
        dataset_rows=values.shape[0],
        train_rows=len(split.train),
        test_rows=len(split.test),
        correlation_threshold=correlation_threshold,
        support_threshold=support_threshold,
        features=tuple(feature_items),
        model_contributions=tuple(model_contributions),
        _train_matrix=train,
    )
