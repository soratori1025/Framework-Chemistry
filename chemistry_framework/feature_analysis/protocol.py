"""Leakage-safe evaluation protocol for evidence selection.

Comparing feature-selection methods is only meaningful if the comparison itself
respects the leakage boundary. Three failure modes are common in the published
literature and are explicitly handled here:

1. **Selection leakage.** Features are chosen on the full dataset and then
   evaluated on a held-out split. Because the selection already saw the held-out
   rows, scores are optimistically biased. :func:`nested_selection_evaluation`
   re-runs the *entire* selection inside every training fold.
2. **No uncertainty quantification.** Reporting a single score difference
   without an interval invites over-reading noise. :func:`paired_bootstrap_test`
   resamples the held-out evaluation units to produce a confidence interval and
   a sign-flip p-value for the *paired* difference between two methods.
3. **No audit trail.** :class:`SelectionAuditRecord` captures fold-level choices
   so a reader can verify how often each method nominates each feature.

All routines here are deterministic given a seed and never touch the evaluation
fold during selection.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .recommend import (
    DEFAULT_DUPLICATE_THRESHOLD,
    DEFAULT_MIN_SUPPORT,
    DEFAULT_SEED_MARGINAL_THRESHOLD,
    DEFAULT_SEED_PAIRS,
    DEFAULT_SEED_SYNERGY_THRESHOLD,
    RecommendationReport,
    recommend_features,
    scorecard_ranking,
)

# A selector maps training arrays to a :class:`RecommendationReport`.
Selector = Callable[[np.ndarray, np.ndarray, int], RecommendationReport]


@dataclass(frozen=True)
class BootstrapInterval:
    """Paired-bootstrap summary for a method-difference statistic."""

    mean_difference: float
    ci_low: float
    ci_high: float
    p_value: float
    n_resamples: int
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FoldAudit:
    """Per-fold record of what a selector chose on the training split."""

    fold: int
    n_train: int
    n_test: int
    method: str
    selected_names: tuple[str, ...]
    selected_indices: tuple[int, ...]
    test_score: float
    train_score: float


@dataclass(frozen=True)
class SelectionAuditRecord:
    """Aggregated audit trail across all folds of a nested evaluation."""

    folds: tuple[FoldAudit, ...]
    selection_frequency: dict[str, float]
    mean_test_score: float
    std_test_score: float
    mean_generalisation_gap: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "folds": [asdict(item) for item in self.folds],
            "selection_frequency": dict(self.selection_frequency),
            "mean_test_score": self.mean_test_score,
            "std_test_score": self.std_test_score,
            "mean_generalisation_gap": self.mean_generalisation_gap,
        }



def _kfold_indices(n_samples: int, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic shuffled k-fold splits as (train, test) index pairs."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if n_samples < n_splits:
        raise ValueError("n_samples must be at least n_splits")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_samples)
    folds = np.array_split(order, n_splits)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in folds:
        test = np.sort(fold)
        mask = np.ones(n_samples, dtype=bool)
        mask[test] = False
        splits.append((np.flatnonzero(mask), test))
    return splits


def _default_scorer(
    X_fit: np.ndarray, y_fit: np.ndarray, X_eval: np.ndarray, y_eval: np.ndarray
) -> float:
    """Out-of-sample Spearman association of the selected subspace.

    The selected columns are combined by an ordinary least-squares fit on the
    training rows, and the score is the Spearman correlation between that
    prediction and the held-out target. Spearman is used because it matches the
    rank-based framing of the selectors and is scale-free, which makes the
    train and test scores directly comparable across methods.
    """
    from scipy import stats as _stats

    if X_eval.shape[1] == 0:
        return 0.0
    if X_fit.size == 0 or X_fit.shape[1] == 0:
        # No features selected: fall back to the best single-column baseline so
        # the protocol still yields a finite, comparable number.
        best = 0.0
        for column in range(X_eval.shape[1]):
            try:
                rho, _ = _stats.spearmanr(X_eval[:, column], y_eval)
            except Exception:
                continue
            if np.isfinite(rho) and abs(rho) > abs(best):
                best = float(rho)
        return best

    design = np.column_stack([X_fit, np.ones(X_fit.shape[0])])
    try:
        coefficients, *_ = np.linalg.lstsq(design, y_fit, rcond=None)
    except Exception:
        return 0.0
    prediction = np.column_stack([X_eval, np.ones(X_eval.shape[0])]) @ coefficients
    try:
        rho, _ = _stats.spearmanr(prediction, y_eval)
    except Exception:
        return 0.0
    return float(rho) if np.isfinite(rho) else 0.0



def nested_selection_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    *,
    selector: Selector,
    feature_names: Sequence[str] | None = None,
    n_splits: int = 5,
    scorer: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float] | None = None,
    seed: int = 0,
) -> tuple[SelectionAuditRecord, list[float]]:
    """Run a selector under nested (leakage-safe) cross-validation.

    The selector is invoked **only** on the training rows of each fold and is
    then scored by projecting its choice onto the held-out rows. Because the
    selector never observes the held-out rows, the resulting test scores are
    free of selection leakage -- the bias that inflates naive feature-selection
    comparisons.

    Args:
        X: Feature matrix ``[n_samples, n_features]``.
        y: Target vector of length ``n_samples``.
        selector: Callable ``selector(X_train, y_train, fold_index)`` returning a
            :class:`RecommendationReport`.
        feature_names: Optional names aligned with the columns of ``X``.
        n_splits: Number of cross-validation folds.
        scorer: Optional override for :func:`_default_scorer`.
        seed: Seed for fold construction.

    Returns:
        ``(audit, fold_test_scores)``. The per-fold scores are returned so that
        callers can feed them into :func:`paired_bootstrap_test` for a
        method-to-method comparison on *identical* folds.
    """
    features = np.asarray(X, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    if features.ndim != 2:
        raise ValueError("X must be a 2D array")
    if features.shape[0] != target.size:
        raise ValueError("X and y must have matching row counts")

    names = list(feature_names) if feature_names is not None else [
        f"column_{i}" for i in range(features.shape[1])
    ]
    score_fn = scorer if scorer is not None else _default_scorer

    fold_records: list[FoldAudit] = []
    test_scores: list[float] = []
    counts: dict[str, int] = {name: 0 for name in names}

    for fold, (train_idx, test_idx) in enumerate(
        _kfold_indices(features.shape[0], n_splits, seed)
    ):
        report = selector(features[train_idx], target[train_idx], fold)
        chosen = report.selected_indices()
        if chosen:
            X_train = features[np.ix_(train_idx, chosen)]
            X_test = features[np.ix_(test_idx, chosen)]
        else:
            X_train = np.empty((train_idx.size, 0))
            X_test = np.empty((test_idx.size, 0))

        test_score = float(score_fn(X_train, target[train_idx], X_test, target[test_idx]))
        train_score = float(score_fn(X_train, target[train_idx], X_train, target[train_idx]))
        test_scores.append(test_score)

        for index in chosen:
            counts[names[index]] = counts.get(names[index], 0) + 1
        fold_records.append(
            FoldAudit(
                fold=fold,
                n_train=int(train_idx.size),
                n_test=int(test_idx.size),
                method=report.method,
                selected_names=tuple(report.selected_names()),
                selected_indices=tuple(report.selected_indices()),
                test_score=test_score,
                train_score=train_score,
            )
        )

    frequency = {name: counts.get(name, 0) / float(n_splits) for name in names}
    scores = np.asarray(test_scores, dtype=np.float64)
    gaps = np.asarray(
        [item.train_score - item.test_score for item in fold_records], dtype=np.float64
    )
    audit = SelectionAuditRecord(
        folds=tuple(fold_records),
        selection_frequency=frequency,
        mean_test_score=float(scores.mean()) if scores.size else 0.0,
        std_test_score=float(scores.std(ddof=0)) if scores.size else 0.0,
        mean_generalisation_gap=float(gaps.mean()) if gaps.size else 0.0,
    )
    return audit, test_scores



def paired_bootstrap_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    n_resamples: int = 5000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapInterval:
    """Paired bootstrap on the fold-wise difference ``scores_a - scores_b``.

    Both score vectors must come from the *same* folds so the comparison is
    paired -- pairing removes between-fold difficulty, which would otherwise
    swamp the method effect. The p-value is the two-sided bootstrap probability
    that the difference crosses zero.

    Args:
        scores_a: Per-fold scores for method A.
        scores_b: Per-fold scores for method B, aligned fold-by-fold with ``A``.
        n_resamples: Number of bootstrap resamples.
        confidence: Confidence level for the reported interval.
        seed: Seed for the resampler.

    Returns:
        A :class:`BootstrapInterval` describing the paired difference.

    Raises:
        ValueError: If lengths differ or either vector is empty.
    """
    a = np.asarray(scores_a, dtype=np.float64).reshape(-1)
    b = np.asarray(scores_b, dtype=np.float64).reshape(-1)
    if a.size != b.size:
        raise ValueError("paired_bootstrap_test requires equal-length score vectors")
    if a.size == 0:
        raise ValueError("paired_bootstrap_test requires at least one fold")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")

    differences = a - b
    observed = float(differences.mean())
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, differences.size, size=(n_resamples, differences.size))
    resampled = differences[draws].mean(axis=1)

    alpha = (1.0 - confidence) / 2.0
    ci_low = float(np.quantile(resampled, alpha))
    ci_high = float(np.quantile(resampled, 1.0 - alpha))

    # Two-sided bootstrap p-value: how often the resampled mean lands on the
    # opposite side of zero from the observed effect, doubled and clipped.
    if observed == 0.0:
        p_value = 1.0
    else:
        opposite = resampled <= 0.0 if observed > 0.0 else resampled >= 0.0
        p_value = float(min(1.0, 2.0 * opposite.mean()))

    return BootstrapInterval(
        mean_difference=observed,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        n_resamples=int(n_resamples),
        confidence=float(confidence),
    )


def siren_selector(
    *,
    k: int,
    weights: Mapping[str, float] | None = None,
    bins: int = 12,
    stability_resamples: int = 16,
    min_support: int = DEFAULT_MIN_SUPPORT,
    duplicate_threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
    pairwise_seed: bool = True,
    seed_pairs: int = DEFAULT_SEED_PAIRS,
    seed_marginal_threshold: float = DEFAULT_SEED_MARGINAL_THRESHOLD,
    seed_synergy_threshold: float = DEFAULT_SEED_SYNERGY_THRESHOLD,
    seed: int = 0,
) -> Selector:
    """Build a fold-aware SIREN selector for :func:`nested_selection_evaluation`."""

    def _select(X_train: np.ndarray, y_train: np.ndarray, fold: int) -> RecommendationReport:
        return recommend_features(
            X_train,
            y_train,
            k=k,
            weights=weights,
            bins=bins,
            stability_resamples=stability_resamples,
            min_support=min_support,
            duplicate_threshold=duplicate_threshold,
            pairwise_seed=pairwise_seed,
            seed_pairs=seed_pairs,
            seed_marginal_threshold=seed_marginal_threshold,
            seed_synergy_threshold=seed_synergy_threshold,
            seed=seed + fold,
        )

    return _select


def scorecard_selector(
    *,
    k: int,
    redundancy_threshold: float = 0.8,
) -> Selector:
    """Build a fold-aware legacy-scorecard selector for baseline comparison."""

    def _select(X_train: np.ndarray, y_train: np.ndarray, fold: int) -> RecommendationReport:
        return scorecard_ranking(
            X_train, y_train, max_features=k, redundancy_threshold=redundancy_threshold
        )

    return _select
