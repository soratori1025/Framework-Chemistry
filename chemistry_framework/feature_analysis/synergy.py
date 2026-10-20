"""Multivariate evidence beyond univariate association.

The legacy scorecard only measured each feature's marginal association with the
target (Spearman ``|r|``) and then subtracted redundancy with a hard ``|r|``
cut-off. That treats relevance as an additive, pairwise property, which fails in
two important regimes:

1. **Conditional relevance / suppressor features.** A descriptor may be nearly
   uncorrelated with the target marginally yet highly informative once a
   correlated partner is held fixed (classic suppression).
2. **Synergy (interaction) effects.** Two descriptors may be jointly predictive
   while neither is predictive alone, e.g. a functional-group count that only
   matters conditional on a size/lipophilicity term.

This module supplies estimators for both quantities using information-theoretic
and partial-correlation proxies. Every routine is deterministic given a seed and
operates strictly on the arrays it is handed, so callers can enforce a
train-only fit policy.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

# Default number of quantile bins used when discretising a continuous target for
# mutual-information estimation. Kraskov-style estimators are not available in
# scikit-learn, so a well-behaved rank-based discretisation is used instead.
DEFAULT_BINS = 12


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation that returns ``0.0`` when the value is undefined.

    Real descriptor matrices contain columns that are constant within a fold, and
    sampled subsamples can be degenerate as well. ``scipy`` emits a
    ``ConstantInputWarning`` and returns ``NaN`` in those cases; mapping them to
    ``0.0`` keeps callers free of ``NaN`` propagation and warning spam. ``0.0``
    is the honest neutral value because no linear association is defined.
    """
    if a.size < 2 or b.size < 2:
        return 0.0
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return 0.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            value, _ = stats.pearsonr(a, b)
    except Exception:
        return 0.0
    return float(value) if np.isfinite(value) else 0.0


@dataclass(frozen=True)
class ConditionalDependence:
    """Redundancy-adjusted relevance of a single candidate column.

    Attributes:
        index: Column index of the candidate inside ``X``.
        marginal_mi: Mutual information of the candidate with the target,
            ignoring every other feature.
        conditional_mi: Mutual information of the candidate with the target
            after residualising both against the already-selected context.
        partial_correlation: Partial Pearson correlation between the candidate
            and the target given the context (signed).
        suppression_gain: ``conditional_mi - marginal_mi``. Positive values
            flag suppressor features whose signal is masked marginally.
    """

    index: int
    marginal_mi: float
    conditional_mi: float
    partial_correlation: float
    suppression_gain: float


def _as_2d(matrix: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    return values


def _bin_series(values: np.ndarray, bins: int) -> np.ndarray:
    """Rank-based quantile discretisation that is stable under ties."""
    ranks = stats.rankdata(values, method="average")
    if ranks.size == 0:
        return ranks.astype(np.int64)
    # Map ranks to [0, bins - 1] so that every bin is non-empty by construction.
    scaled = np.floor((ranks - 1.0) / max(ranks.size, 1) * bins)
    return np.clip(scaled, 0.0, bins - 1.0).astype(np.int64)


def _is_classification(y: np.ndarray) -> bool:
    unique = np.unique(y)
    return unique.size <= 2 and np.all(np.isin(unique, (0.0, 1.0)))


def _mutual_information(X: np.ndarray, y: np.ndarray, *, bins: int, seed: int) -> np.ndarray:
    """Mutual information between every column of ``X`` and ``y``."""
    if X.size == 0:
        return np.zeros(X.shape[1], dtype=np.float64)
    discrete_y = _is_classification(y)
    estimator = mutual_info_classif if discrete_y else mutual_info_regression
    X_disc = X if discrete_y else np.column_stack(
        [_bin_series(X[:, j], bins) for j in range(X.shape[1])]
    )
    y_in = y if discrete_y else _bin_series(y, bins)
    try:
        values = estimator(X_disc, y_in, random_state=seed)
    except Exception:
        values = np.zeros(X.shape[1], dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    values[~np.isfinite(values)] = 0.0
    return values


def _interaction_term(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Standardised element-wise product used as an interaction proxy.

    Centring and scaling each component before multiplying makes the product an
    explicit interaction basis function. This lets a univariate MI estimator
    detect XOR-style ("both must be present") structure that is invisible to any
    bivariate binning of the raw pair.
    """
    a_std = np.std(a)
    b_std = np.std(b)
    if a_std < 1e-12 or b_std < 1e-12:
        return np.zeros_like(a)
    return ((a - a.mean()) / a_std) * ((b - b.mean()) / b_std)



def _residualise(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    """Regress ``values`` on ``design`` (plus intercept) and return residuals."""
    if design.size == 0:
        return values - values.mean()
    augmented = np.column_stack([np.ones(design.shape[0]), design])
    try:
        coefficients, *_ = np.linalg.lstsq(augmented, values, rcond=None)
    except np.linalg.LinAlgError:
        return values - values.mean()
    residuals = values - augmented @ coefficients
    residuals[~np.isfinite(residuals)] = 0.0
    return residuals


def _partial_correlation(x: np.ndarray, y: np.ndarray, context: np.ndarray) -> float:
    """Signed partial Pearson correlation of ``x`` and ``y`` given ``context``."""
    if x.size < 3:
        return 0.0
    x_res = _residualise(x, context)
    y_res = _residualise(y, context)
    if np.std(x_res) < 1e-12 or np.std(y_res) < 1e-12:
        return 0.0
    try:
        r, _ = stats.pearsonr(x_res, y_res)
    except Exception:
        return 0.0
    return float(r) if np.isfinite(r) else 0.0




def compute_conditional_dependence(
    X: np.ndarray,
    y: np.ndarray,
    candidates: Sequence[int],
    context: Sequence[int] = (),
    *,
    bins: int = DEFAULT_BINS,
    seed: int = 0,
) -> dict[int, ConditionalDependence]:
    """Conditional (redundancy-adjusted) relevance for each candidate column.

    For every ``j`` in ``candidates`` the routine computes the marginal mutual
    information ``I(X_j; y)`` and a conditional mutual information estimate
    ``I(X_j; y | X_context)`` obtained by residualising both ``X_j`` and ``y``
    against the context columns and re-measuring dependence on the residuals.
    Partial correlation is reported alongside as a signed, cheap companion.

    Args:
        X: Feature matrix ``[n_samples, n_features]``.
        y: Target vector ``[n_samples]``.
        candidates: Column indices to score.
        context: Column indices already accepted into the selected set.
        bins: Quantile bins for the discretising MI estimator.
        seed: Seed for the MI estimator's tie-breaking.

    Returns:
        Mapping ``index -> ConditionalDependence``.
    """
    features = _as_2d(X, name="X")
    target = np.asarray(y, dtype=np.float64).ravel()
    if features.shape[0] != target.shape[0]:
        raise ValueError("X and y must have matching row counts")
    if bins < 2:
        raise ValueError("bins must be at least 2")
    candidate_indices = [int(c) for c in candidates]
    context_indices = [int(c) for c in context]
    for index in candidate_indices + context_indices:
        if index < 0 or index >= features.shape[1]:
            raise ValueError(f"feature index {index} is out of range")
    if not candidate_indices:
        return {}

    marginal = _mutual_information(features[:, candidate_indices], target, bins=bins, seed=seed)
    context_matrix = (
        features[:, context_indices]
        if context_indices
        else np.empty((features.shape[0], 0))
    )

    output: dict[int, ConditionalDependence] = {}
    for position, index in enumerate(candidate_indices):
        column = features[:, index]
        if np.std(column) < 1e-12:
            output[index] = ConditionalDependence(
                index=index,
                marginal_mi=0.0,
                conditional_mi=0.0,
                partial_correlation=0.0,
                suppression_gain=0.0,
            )
            continue
        if context_indices:
            residual_feature = _residualise(column, context_matrix)
            residual_target = _residualise(target, context_matrix)
            conditional_mi = float(
                _mutual_information(
                    residual_feature.reshape(-1, 1), residual_target, bins=bins, seed=seed
                )[0]
            )
        else:
            conditional_mi = float(marginal[position])
        output[index] = ConditionalDependence(
            index=index,
            marginal_mi=float(marginal[position]),
            conditional_mi=conditional_mi,
            partial_correlation=_partial_correlation(column, target, context_matrix),
            suppression_gain=float(conditional_mi - marginal[position]),
        )
    return output


@dataclass(frozen=True)
class PairwiseSynergy:
    """Interaction lift between two candidate columns. Positive means synergy."""

    first: int
    second: int
    joint_mi: float
    first_mi: float
    second_mi: float
    synergy: float
    correlation: float
    interaction_mi: float = 0.0


def compute_pairwise_synergy(
    X: np.ndarray,
    y: np.ndarray,
    first: int,
    second: int,
    *,
    bins: int = DEFAULT_BINS,
    seed: int = 0,
) -> PairwiseSynergy:
    """Synergy of a feature pair measured as interaction-information lift.

    ``synergy = I({X_i, X_j}; y) - I(X_i; y) - I(X_j; y)``. This is the standard
    interaction-information definition: positive values indicate that the pair
    carries information unavailable to either member alone, while negative
    values indicate redundancy. Pairwise Pearson correlation is reported for
    context and is what the legacy scorecard pruned on.
    """
    features = _as_2d(X, name="X")
    target = np.asarray(y, dtype=np.float64).ravel()
    if features.shape[0] != target.shape[0]:
        raise ValueError("X and y must have matching row counts")
    for index in (first, second):
        if index < 0 or index >= features.shape[1]:
            raise ValueError(f"feature index {index} is out of range")
    if int(first) == int(second):
        raise ValueError("synergy requires two distinct feature indices")

    pair = features[:, [int(first), int(second)]]
    discrete_y = _is_classification(target)
    pair_disc = (
        pair
        if discrete_y
        else np.column_stack([_bin_series(pair[:, 0], bins), _bin_series(pair[:, 1], bins)])
    )
    target_in = target if discrete_y else _bin_series(target, bins)
    estimator = mutual_info_classif if discrete_y else mutual_info_regression

    # True joint information I({Xi, Xj}; y). The pair is encoded as a single
    # composite categorical variable (one bin index per cell of the quantile
    # grid) before measuring MI, because summing per-column MI estimates would
    # simply recover the marginals and never expose interaction structure.
    pair_codes = pair_disc[:, 0] * bins + pair_disc[:, 1]
    try:
        joint = float(_mutual_information(pair_codes.reshape(-1, 1), target, bins=bins, seed=seed)[0])
    except Exception:
        joint = 0.0
    if not np.isfinite(joint):
        joint = 0.0

    singles = _mutual_information(pair, target, bins=bins, seed=seed)


    # Explicit interaction basis. Two subtleties matter here:
    #   1. A plain bivariate binning of the raw pair cannot represent XOR-style
    #      structure, so an element-wise product term is added.
    #   2. The raw product of two near-collinear columns is dominated by the
    #      squared marginal signal (e.g. x^2 for a duplicate), which would be
    #      misread as synergy. Residualising the interaction against both raw
    #      columns removes that marginal leakage, leaving only the genuinely
    #      "beyond-marginal" component.
    interaction = _interaction_term(pair[:, 0], pair[:, 1])
    interaction = _residualise(interaction, pair)
    if not discrete_y:
        interaction = _bin_series(interaction, bins)
    try:
        interaction_mi = float(
            estimator(interaction.reshape(-1, 1), target_in, random_state=seed)[0]
        )
    except Exception:
        interaction_mi = 0.0
    if not np.isfinite(interaction_mi):
        interaction_mi = 0.0

    with warnings.catch_warnings():
        # Constant or near-constant pairs make the correlation undefined; the
        # collinearity gate below handles that case explicitly, so the warning
        # is pure log noise.
        warnings.simplefilter("ignore")
        correlation = _safe_pearson(pair[:, 0], pair[:, 1])

    # Collinearity gate. When two columns are near-duplicates the product term
    # is a deterministic function of the pair (e.g. x0 * x0), so its MI only
    # re-measures already-counted marginal signal and would masquerade as
    # synergy. Identical columns are *maximally redundant* by definition, so the
    # score is forced to the pure redundancy value -I(X; y) in that limit.
    if abs(correlation) >= 0.999 or np.unique(pair, axis=0).shape[0] <= 1:
        joint = singles[0]
        interaction_mi = 0.0


    # Relative interaction lift.
    #
    # A pair is synergistic when the joint information (bivariate + orthogonal
    # interaction) exceeds the sum of its marginal contributions. The raw lift
    # ``joint + interaction - first - second`` is in nats, but dividing it by the
    # marginal scale -- the obvious way to make pairs comparable -- explodes when
    # both marginals are near zero, which is exactly the interaction-only regime
    # the term exists to detect. On the real descriptor matrices that produced
    # objective values in the thousands and selected pure noise.
    #
    # The lift is therefore normalised by the *total* information the pair
    # carries. The result is bounded in ``[-1, 1]`` and has a direct reading:
    # the fraction of the pair's joint information that the marginals do not
    # explain. ``+1`` is a pure interaction (e.g. XOR with no marginal signal);
    # ``-1`` is a pure duplicate; ``0`` is a redundant-but-informative pair.
    total_information = max(
        float(joint + interaction_mi), float(singles[0] + singles[1]), 1e-6
    )
    synergy = float((joint + interaction_mi - singles[0] - singles[1]) / total_information)


    return PairwiseSynergy(
        first=int(first),
        second=int(second),
        joint_mi=joint,
        first_mi=float(singles[0]),
        second_mi=float(singles[1]),
        synergy=synergy,
        correlation=float(correlation),
        interaction_mi=float(interaction_mi),
    )




def rank_synergy_partners(
    X: np.ndarray,
    y: np.ndarray,
    anchor: int,
    candidates: Sequence[int],
    *,
    bins: int = DEFAULT_BINS,
    seed: int = 0,
) -> list[PairwiseSynergy]:
    """Rank every candidate by its synergy with ``anchor`` (descending)."""
    anchor_index = int(anchor)
    results = [
        compute_pairwise_synergy(X, y, anchor_index, int(candidate), bins=bins, seed=seed)
        for candidate in candidates
        if int(candidate) != anchor_index
    ]
    results.sort(key=lambda item: item.synergy, reverse=True)
    return results

    return output
