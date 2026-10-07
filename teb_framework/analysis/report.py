"""Feature and target distribution reports for a prepared experiment."""
from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, TypedDict
import numpy as np
from numpy.typing import ArrayLike
from ..diagnostics import diagnose_features
from ..features import FeatureMatrix
from ..splits import DatasetSplit


class DistributionSummary(TypedDict):
    count: int
    missing: int
    mean: float | None
    std: float | None
    min: float | None
    max: float | None
    histogram_edges: list[float]
    histogram_counts: list[int]


@dataclass(frozen=True)
class AnalysisReport:
    dataset_rows: int
    split_sizes: dict[str, int]
    features: tuple[dict[str, object], ...]
    targets: dict[str, dict[str, DistributionSummary]]
    collinear_pairs: tuple[tuple[str, str, float], ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return destination

def _distribution(values: np.ndarray, bins: int) -> DistributionSummary:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0, "missing": int(values.size), "mean": None, "std": None,
                "min": None, "max": None, "histogram_edges": [], "histogram_counts": []}
    counts, edges = np.histogram(finite, bins=bins)
    return {
        "count": int(finite.size),
        "missing": int(values.size - finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "histogram_edges": edges.astype(float).tolist(),
        "histogram_counts": counts.astype(int).tolist(),
    }

def analyze_features_and_targets(
    matrix: FeatureMatrix,
    split: DatasetSplit,
    targets: Mapping[str, ArrayLike],
    *,
    bins: int = 20,
    support_threshold: int = 1, correlation_threshold: float = 0.95,) -> AnalysisReport:
    """Summarize train-only feature support/distribution and test extrapolation."""
    if bins < 1 or support_threshold < 1:
        raise ValueError("bins and support_threshold must be positive")
    indices = np.concatenate((split.train, split.validation, split.test))
    if len(indices) != matrix.values.shape[0] or set(indices.tolist()) != set(
        range(matrix.values.shape[0])
    ):
        raise ValueError("split must partition every row in the feature matrix exactly once")
    feature_train = matrix.values[split.train]
    feature_test = matrix.values[split.test]
    diagnostics = diagnose_features(
        feature_train,
        feature_test,
        matrix.names,
        correlation_threshold=correlation_threshold,
    )
    feature_items: list[dict[str, object]] = []
    warnings: list[str] = []
    for column, diagnostic in enumerate(diagnostics.features):
        values = feature_train[:, column]
        support = int(np.count_nonzero(np.abs(values) > 1e-12))
        feature_items.append(
            {
                **asdict(diagnostic),
                "train_support": support,
                "train_support_fraction": float(support / len(values)),
                "train_histogram": _distribution(values, bins),
                "below_support_threshold": support < support_threshold,
            }
        )
        if support < support_threshold:
            warnings.append(
                f"feature {diagnostic.name!r} train support {support} "
                f"is below threshold {support_threshold}"
            )
        if diagnostic.out_of_range_count:
            warnings.append(
                f"feature {diagnostic.name!r} has {diagnostic.out_of_range_count} "
                "test values outside the train range"
            )
        if diagnostic.constant_in_train:
            warnings.append(f"feature {diagnostic.name!r} is constant in train")

    target_items: dict[str, dict[str, DistributionSummary]] = {}
    for name, raw_values in targets.items():
        values = np.asarray(raw_values, dtype=np.float64)
        if values.ndim not in {1, 2} or values.shape[0] != matrix.values.shape[0]:
            raise ValueError(f"target {name!r} must have one row per dataset molecule")
        train_distribution = _distribution(values[split.train], bins)
        target_items[name] = {
            "train": train_distribution,
            "validation": _distribution(values[split.validation], bins),
            "test": _distribution(values[split.test], bins),
        }
        if train_distribution["count"] == 0:
            warnings.append(f"target {name!r} has no finite training values")

    return AnalysisReport(
        dataset_rows=matrix.values.shape[0],
        split_sizes={
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
        },
        features=tuple(feature_items),
        targets=target_items,
        collinear_pairs=diagnostics.collinear_pairs,
        warnings=tuple(warnings),
    )
