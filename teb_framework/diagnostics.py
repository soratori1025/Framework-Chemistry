"""Train/test feature-range and collinearity diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class FeatureDiagnostics:
    name: str
    train_mean: float
    train_std: float
    train_min: float
    train_max: float
    test_min: float
    test_max: float
    out_of_range_count: int
    constant_in_train: bool


@dataclass(frozen=True)
class FeatureReport:
    features: tuple[FeatureDiagnostics, ...]
    collinear_pairs: tuple[tuple[str, str, float], ...]
    correlation_threshold: float

    def format_report(self) -> str:
        lines = ["Feature diagnostics"]
        for item in self.features:
            lines.append(
                f"{item.name}: train mean={item.train_mean:.6g}, "
                f"std={item.train_std:.6g}, range=[{item.train_min:.6g}, "
                f"{item.train_max:.6g}], test range=[{item.test_min:.6g}, "
                f"{item.test_max:.6g}], out-of-range={item.out_of_range_count}"
            )
            if item.constant_in_train:
                lines.append(f"WARNING: {item.name} is constant in training data")
            if item.out_of_range_count:
                lines.append(
                    f"WARNING: {item.name} has {item.out_of_range_count} test values "
                    "outside the training range"
                )
        for left, right, correlation in self.collinear_pairs:
            lines.append(
                f"WARNING: {left} and {right} are strongly correlated "
                f"(r={correlation:.4f})"
            )
        return "\n".join(lines)


def diagnose_features(
    train: np.ndarray,
    test: np.ndarray,
    feature_names: Sequence[str],
    *,
    correlation_threshold: float = 0.95,
    constant_tolerance: float = 1e-12,
) -> FeatureReport:
    """Report test extrapolation and strong pairwise train-set correlations."""
    train = np.asarray(train, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    names = tuple(feature_names)
    if train.ndim != 2 or test.ndim != 2:
        raise ValueError("train and test feature data must be 2-D matrices")
    if train.shape[1] != len(names) or test.shape[1] != len(names):
        raise ValueError("feature-name count must match train and test column counts")
    if train.shape[0] < 2 or test.shape[0] < 1:
        raise ValueError("diagnostics require at least 2 train rows and 1 test row")
    if not names or len(set(names)) != len(names):
        raise ValueError("feature_names must be non-empty and unique")
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError("diagnostic inputs must contain only finite values")
    if not 0.0 < correlation_threshold <= 1.0:
        raise ValueError("correlation_threshold must be in (0, 1]")
    if constant_tolerance < 0:
        raise ValueError("constant_tolerance must be non-negative")

    means = train.mean(axis=0)
    stds = train.std(axis=0)
    minima = train.min(axis=0)
    maxima = train.max(axis=0)
    test_minima = test.min(axis=0)
    test_maxima = test.max(axis=0)
    diagnostics: list[FeatureDiagnostics] = []
    for column, name in enumerate(names):
        outside = (test[:, column] < minima[column]) | (test[:, column] > maxima[column])
        diagnostics.append(
            FeatureDiagnostics(
                name=name,
                train_mean=float(means[column]),
                train_std=float(stds[column]),
                train_min=float(minima[column]),
                train_max=float(maxima[column]),
                test_min=float(test_minima[column]),
                test_max=float(test_maxima[column]),
                out_of_range_count=int(outside.sum()),
                constant_in_train=bool(stds[column] <= constant_tolerance),
            )
        )

    correlation = np.corrcoef(train, rowvar=False) if len(names) > 1 else np.empty((1, 1))
    pairs: list[tuple[str, str, float]] = []
    for left in range(len(names)):
        if stds[left] <= constant_tolerance:
            continue
        for right in range(left + 1, len(names)):
            if stds[right] <= constant_tolerance:
                continue
            value = float(correlation[left, right])
            if abs(value) >= correlation_threshold:
                pairs.append((names[left], names[right], value))
    return FeatureReport(tuple(diagnostics), tuple(pairs), correlation_threshold)
