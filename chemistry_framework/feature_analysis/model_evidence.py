"""Prior stability and model-specific feature-family evidence utilities."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PriorCoefficientEvidence:
    feature: str
    count: int
    mean: float
    standard_deviation: float
    minimum: float
    maximum: float
    sign_consistency: float
    bootstrap_95_ci: tuple[float, float]


@dataclass(frozen=True)
class ModelContribution:
    """Metric change when a feature group is perturbed or removed.

    Positive deltas always mean that the complete, unperturbed model performed
    better. These are model-specific predictive measurements, not causal
    effects.
    """

    group: str
    method: str
    metric: str
    baseline_score: float
    comparison_scores: tuple[float, ...]
    contribution_deltas: tuple[float, ...]
    mean_delta: float
    standard_deviation: float
    positive_fraction: float


def summarize_prior_coefficients(
    runs: Sequence[Mapping[str, float]],
    *,
    fixed_coefficients: Mapping[str, float] | None = None,
    bootstrap_samples: int = 5000,
    seed: int = 0,
) -> dict[str, PriorCoefficientEvidence]:
    """Summarize fitted coefficients across runs with exploratory bootstrap CIs."""
    if not runs:
        raise ValueError("at least one coefficient mapping is required")
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    features = sorted({feature for run in runs for feature in run})
    if not features:
        raise ValueError("coefficient mappings contain no features")

    rng = np.random.default_rng(seed)
    output: dict[str, PriorCoefficientEvidence] = {}
    for feature in features:
        values = np.asarray(
            [float(run[feature]) for run in runs if feature in run],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError(f"coefficients for feature {feature!r} must be finite")
        signs = np.sign(values)
        nonzero_signs = signs[signs != 0]
        if nonzero_signs.size:
            sign_consistency = float(
                max(np.count_nonzero(nonzero_signs > 0), np.count_nonzero(nonzero_signs < 0))
                / values.size
            )
        else:
            sign_consistency = 0.0
        if values.size == 1:
            confidence_interval = (float(values[0]), float(values[0]))
        else:
            draws = rng.choice(values, size=(bootstrap_samples, values.size), replace=True)
            low, high = np.quantile(draws.mean(axis=1), [0.025, 0.975])
            confidence_interval = (float(low), float(high))
        output[feature] = PriorCoefficientEvidence(
            feature=feature,
            count=int(values.size),
            mean=float(values.mean()),
            standard_deviation=float(values.std(ddof=1)) if values.size > 1 else 0.0,
            minimum=float(values.min()),
            maximum=float(values.max()),
            sign_consistency=sign_consistency,
            bootstrap_95_ci=confidence_interval,
        )

    for feature, coefficient in (fixed_coefficients or {}).items():
        value = float(coefficient)
        if not np.isfinite(value):
            raise ValueError(f"fixed coefficient for {feature!r} must be finite")
        output[feature] = PriorCoefficientEvidence(
            feature=feature,
            count=0,
            mean=value,
            standard_deviation=0.0,
            minimum=value,
            maximum=value,
            sign_consistency=1.0,
            bootstrap_95_ci=(value, value),
        )
    return output


def _contribution(
    group: str,
    method: str,
    metric: str,
    baseline_score: float,
    comparison_scores: Sequence[float],
    *,
    higher_is_better: bool,
) -> ModelContribution:
    comparisons = np.asarray(comparison_scores, dtype=np.float64).reshape(-1)
    if comparisons.size == 0 or not np.isfinite(comparisons).all():
        raise ValueError("comparison scores must contain finite values")
    baseline = float(baseline_score)
    if not np.isfinite(baseline):
        raise ValueError("baseline score must be finite")
    deltas = baseline - comparisons if higher_is_better else comparisons - baseline
    return ModelContribution(
        group=group,
        method=method,
        metric=metric,
        baseline_score=baseline,
        comparison_scores=tuple(float(value) for value in comparisons),
        contribution_deltas=tuple(float(value) for value in deltas),
        mean_delta=float(deltas.mean()),
        standard_deviation=float(deltas.std(ddof=1)) if deltas.size > 1 else 0.0,
        positive_fraction=float(np.mean(deltas > 0)),
    )


def permutation_group_importance(
    predictor: Callable[[np.ndarray], Any],
    test_features: np.ndarray,
    test_targets: np.ndarray,
    groups: Mapping[str, Sequence[int]],
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    metric_name: str,
    higher_is_better: bool = True,
    repeats: int = 20,
    seed: int = 0,
) -> dict[str, ModelContribution]:
    """Measure test-metric change after joint row permutation of feature groups.

    Columns in a group share each row permutation, preserving dependencies
    within that group. Correlations across groups can still distort this
    model-specific estimate.
    """
    features = np.asarray(test_features, dtype=np.float64)
    targets = np.asarray(test_targets)
    if features.ndim != 2 or targets.shape[0] != features.shape[0]:
        raise ValueError("test features and targets must have matching row counts")
    if not np.isfinite(features).all():
        raise ValueError("test_features must contain only finite values")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    normalized: dict[str, tuple[int, ...]] = {}
    assigned_columns: set[int] = set()
    for name, raw_indices in groups.items():
        indices = tuple(int(index) for index in raw_indices)
        if not indices or len(set(indices)) != len(indices):
            raise ValueError(f"group {name!r} must contain unique feature indices")
        if min(indices) < 0 or max(indices) >= features.shape[1]:
            raise ValueError(f"group {name!r} contains an out-of-range feature index")
        if assigned_columns.intersection(indices):
            raise ValueError("feature groups must not overlap")
        assigned_columns.update(indices)
        normalized[name] = indices
    if not normalized:
        raise ValueError("at least one feature group is required")

    baseline_predictions = np.asarray(predictor(features))
    baseline_score = float(metric(targets, baseline_predictions))
    if not np.isfinite(baseline_score):
        raise ValueError("metric returned a non-finite baseline score")
    rng = np.random.default_rng(seed)
    results: dict[str, ModelContribution] = {}
    for name, columns in normalized.items():
        scores: list[float] = []
        for _ in range(repeats):
            permuted = features.copy()
            order = rng.permutation(features.shape[0])
            permuted[:, columns] = features[order][:, columns]
            predictions = np.asarray(predictor(permuted))
            score = float(metric(targets, predictions))
            if not np.isfinite(score):
                raise ValueError(f"metric returned a non-finite score for group {name!r}")
            scores.append(score)
        results[name] = _contribution(
            name,
            "group_permutation",
            metric_name,
            baseline_score,
            scores,
            higher_is_better=higher_is_better,
        )
    return results


def leave_group_out_importance(
    fit_predict: Callable[[tuple[int, ...]], Any],
    feature_names: Sequence[str],
    groups: Mapping[str, Sequence[str]],
    test_targets: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    metric_name: str,
    higher_is_better: bool = True,
) -> dict[str, ModelContribution]:
    """Compare a full model with models retrained after removing each group.

    ``fit_predict`` receives the retained column indices and is responsible
    for fitting with the same training/validation protocol on every call.
    """
    names = tuple(feature_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("feature_names must be non-empty and unique")
    if not groups:
        raise ValueError("at least one feature group is required")
    index_by_name = {name: index for index, name in enumerate(names)}
    normalized: dict[str, tuple[int, ...]] = {}
    assigned_columns: set[int] = set()
    for group, members in groups.items():
        unknown = set(members) - set(index_by_name)
        if unknown:
            raise ValueError(
                f"group {group!r} references unknown feature(s): {sorted(unknown)}"
            )
        columns = tuple(index_by_name[name] for name in members)
        if not columns:
            raise ValueError(f"group {group!r} must not be empty")
        if assigned_columns.intersection(columns):
            raise ValueError("feature groups must not overlap")
        assigned_columns.update(columns)
        normalized[group] = columns
    all_columns = tuple(range(len(names)))
    baseline_predictions = np.asarray(fit_predict(all_columns))
    baseline_score = float(metric(np.asarray(test_targets), baseline_predictions))
    if not np.isfinite(baseline_score):
        raise ValueError("metric returned a non-finite full-model score")

    results: dict[str, ModelContribution] = {}
    for group, removed_columns in normalized.items():
        retained = tuple(index for index in all_columns if index not in removed_columns)
        if not retained:
            raise ValueError(f"removing group {group!r} would remove all features")
        predictions = np.asarray(fit_predict(retained))
        score = float(metric(np.asarray(test_targets), predictions))
        results[group] = _contribution(
            group,
            "leave_group_out",
            metric_name,
            baseline_score,
            (score,),
            higher_is_better=higher_is_better,
        )
    return results


def model_contribution_dict(value: ModelContribution) -> dict[str, object]:
    """Return a JSON-ready representation of a model contribution."""
    return asdict(value)
