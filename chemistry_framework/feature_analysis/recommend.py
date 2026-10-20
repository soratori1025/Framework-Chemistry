"""SIREN: Stability-Informed Redundancy- and Synergy-aware Evidence Nomination.

This module contains the framework's *recommendation* contribution. The legacy
recommender (``selection.select_evidence_recommendation``) ranks features by
univariate Spearman ``|r|`` and then greedily prunes pairwise ``|r|`` above a
fixed threshold. That objective is additive and marginal, so it systematically:

* discards suppressor features (signal masked by a correlated partner);
* discards synergy partners (informative only in combination);
* ignores estimator variance (an unselected but stable feature loses to a
  marginally-stronger but unstable one);
* has no notion of evidence strength, so ``k`` must be supplied by the user
  rather than read off the data.

SIREN replaces the objective with a greedy forward selection that scores each
candidate *conditional on the already-selected context*:

.. code-block:: text

    J(j | S) = alpha * relevance(j | S)      # conditional MI, suppressor-safe
             + beta  * synergy(j, S)         # interaction-information lift
             + gamma * stability(j)          # resample sign/rank persistence
             - delta * redundancy(j, S)      # conditional redundancy mass

Every term is computed strictly on the training arrays handed to the selector,
so callers retain full control of the leakage boundary. The result is a fully
auditable :class:`RecommendationReport` that explains *which term* justified
each nomination.
"""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from .synergy import (
    DEFAULT_BINS,
    PairwiseSynergy,
    _bin_series,
    _is_classification,
    _mutual_information,
    _partial_correlation,
    _residualise,
    compute_pairwise_synergy,
)

# Default trade-off weights. The relevance term dominates (matching the prior of
# the legacy scorecard), while synergy and stability act as disciplined bonuses.
#
# ``redundancy`` is **disabled by default (0.0)**. It was introduced as a soft
# penalty in place of the legacy hard ``|r|`` cut-off, but the E-REC isolation
# study found the penalty strictly harmful: on every dataset tested the test
# score increases monotonically as the weight moves +1 -> 0 -> -1, and the
# selection is driven *away* from the target rather than towards a
# non-redundant subset. Measured with nested CV (see
# ``benchmarks/results/experiment_e_rec_recommendation_ablation.json``):
#
#     dataset          delta=+1   delta=0   delta=-1
#     ESOL              +0.8542    +0.8565   +0.8634
#     FreeSolv          +0.7988    +0.8008   +0.8182
#     Lipophilicity     +0.4440    +0.5600   (not run)
#
# The cause is that ``redundancy`` mixes two incommensurable quantities. The
# ``marginal - conditional`` term is built from *target* MI and so already
# penalises a candidate for the target variance the context explains -- double
# counting the relevance term that was just added. The ``correlation_mass`` term
# is an *unconditional*, non-partial correlation, so it penalises a feature that
# is marginally correlated with a selected one even when it carries independent
# information given the target. Weighting either at 1.0 removes the very
# suppressor features that the conditional-MI relevance term exists to find.
#
# The term and its weight remain available for research; pass
# ``weights={"redundancy": 1.0}`` to reproduce the pre-2026 behaviour.
DEFAULT_WEIGHTS: Mapping[str, float] = {
    "relevance": 1.0,
    "synergy": 0.5,
    "stability": 0.5,
    "redundancy": 0.0,
}

# Minimum number of distinct values a column must take before it is eligible
# for nomination. See :func:`_support_mask` for the measurement that motivates
# this: the scorecard's best features are monotone transforms of the target,
# while conditional MI over discretised rare fragment counters is estimated from
# a handful of rows, so without a screen the greedy objective is dominated by
# estimator variance on features that cannot support an association estimate.
DEFAULT_MIN_SUPPORT: int = 5

# Near-duplicate screen. Exact and near-exact duplicate descriptors (for
# example a raw count and its monotone transform, or the same fragment counted
# by two implementations) carry no additional information, so spending one of
# the ``k`` slots on the copy is strictly wasteful. The threshold is deliberately
# far above the correlation that defines a suppressor: suppressors are
# *correlated* partners (typically |r| <= 0.95) that each carry independent
# signal, whereas |r| >= 0.999 means the two columns are the same variable.
DEFAULT_DUPLICATE_THRESHOLD: float = 0.999

# Synergy seeding. Greedy forward selection evaluates synergy only against the
# features already selected, so at the first step -- with an empty context --
# a pure interaction pair (neither member marginally informative) is invisible
# and the first slot goes to whichever candidate has the largest marginal
# relevance or stability. When no candidate carries marginal information,
# :func:`recommend_features` therefore screens candidate pairs for the
# strongest interaction and seeds the selection with that pair.
DEFAULT_SEED_MARGINAL_THRESHOLD: float = 0.05
DEFAULT_SEED_SYNERGY_THRESHOLD: float = 0.10
DEFAULT_SEED_PAIRS: int = 500


@dataclass(frozen=True)
class FeatureNomination:
    """Auditable justification for a single nominated feature.

    Attributes:
        name: Feature name (or ``"column_<i>"`` when unnamed).
        index: Column index inside the original matrix.
        rank: 1-based order in which the feature was nominated.
        objective: Final value of ``J(j | S)`` at nomination time.
        relevance: Conditional mutual information given the prior context.
        synergy: Mean interaction-information lift against the context.
        stability: Resample persistence in ``[0, 1]``.
        redundancy: Conditional redundancy mass against the context.
        marginal_mi: Unconditional MI, for contrast with the conditional value.
        suppression_gain: ``conditional_mi - marginal_mi``; positive flags a
            feature that the univariate scorecard would have missed.
        dominant_term: Name of the largest positive contributor to ``objective``.
    """

    name: str
    index: int
    rank: int
    objective: float
    relevance: float
    synergy: float
    stability: float
    redundancy: float
    marginal_mi: float
    suppression_gain: float
    dominant_term: str


@dataclass(frozen=True)
class RecommendationReport:
    """Complete, serialisable output of a SIREN run."""

    method: str
    k: int
    n_candidates: int
    n_selected: int
    weights: dict[str, float]
    bins: int
    stability_resamples: int
    nominations: tuple[FeatureNomination, ...]
    objective_curve: tuple[float, ...]
    stopping_rule: str
    screened_features: tuple[str, ...] = ()
    interpretation_note: str = (
        "Nominations are conditional evidence under this dataset, split and "
        "protocol. Objective terms are association- and interaction-based "
        "statistics, not causal effects, and stability is measured by "
        "resample persistence rather than external validation."
    )

    def selected_indices(self) -> list[int]:
        return [item.index for item in self.nominations]

    def selected_names(self) -> list[str]:
        return [item.name for item in self.nominations]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "k": self.k,
            "n_candidates": self.n_candidates,
            "n_selected": self.n_selected,
            "weights": dict(self.weights),
            "bins": self.bins,
            "stability_resamples": self.stability_resamples,
            "nominations": [asdict(item) for item in self.nominations],
            "objective_curve": list(self.objective_curve),
            "stopping_rule": self.stopping_rule,
            "screened_features": list(self.screened_features),
            "interpretation_note": self.interpretation_note,
        }



def _stability(
    column: np.ndarray,
    target: np.ndarray,
    rng: np.random.Generator,
    resamples: int,
    subsample_fraction: float,
) -> float:
    """Resample persistence of the candidate-target association in ``[0, 1]``.

    The candidate is repeatedly scored on random subsamples; stability is the
    fraction of resamples that agree in *sign* with the full-sample association.
    A feature whose apparent signal flips sign under resampling is unreliable
    even when its point estimate is large, which is exactly the failure mode the
    legacy marginal scorecard cannot see.
    """
    if resamples < 1 or column.size < 4:
        return 0.0
    full = _safe_spearman(column, target)
    if abs(full) < 1e-12:
        return 0.0
    size = max(3, int(round(subsample_fraction * column.size)))
    size = min(size, column.size)
    agreements = 0
    total = 0
    for _ in range(resamples):
        indices = rng.choice(column.size, size=size, replace=False)
        sampled = _safe_spearman(column[indices], target[indices])
        if sampled == 0.0 and abs(full) > 0.0:
            # A degenerate subsample carries no sign evidence; skip it rather
            # than counting it as disagreement.
            if np.ptp(column[indices]) == 0:
                continue
        total += 1
        if np.sign(sampled) == np.sign(full):
            agreements += 1
    if total == 0:
        return 0.0
    return float(agreements / total)



def _candidate_components(
    X: np.ndarray,
    y: np.ndarray,
    index: int,
    selected: Sequence[int],
    *,
    bins: int,
    seed: int,
    stability_cache: Mapping[int, float],
    discrete_y: bool = False,
) -> dict[str, float]:
    """Compute each objective term for one candidate given the context ``S``."""
    column = X[:, index]
    context = np.asarray(selected, dtype=int)

    marginal = float(_mutual_information(column.reshape(-1, 1), y, bins=bins, seed=seed)[0])
    if context.size:
        residual_feature = _residualise(column, X[:, context])
        # For a categorical target ``residualise(y, ...)`` is not meaningful:
        # the residual is continuous and silently switches the estimator to
        # ``mutual_info_regression``, producing values that are not bounded by
        # ``H(y)`` and therefore cannot be compared with the marginal value or
        # with the nats-scale synergy term. Condition on the descriptor only and
        # leave the label discrete.
        residual_target = (
            y if discrete_y else _residualise(y, X[:, context])
        )
        conditional = float(
            _mutual_information(
                residual_feature.reshape(-1, 1), residual_target, bins=bins, seed=seed
            )[0]
        )
    else:
        conditional = marginal

    # Synergy is the *relative* interaction lift of the candidate/partner pair:
    # the share of their joint information that the two marginals do not
    # explain. ``PairwiseSynergy.synergy`` already returns that bounded ratio,
    # so no further normalisation is applied here. (Dividing by the marginal
    # scale instead explodes when both marginals are near zero -- the
    # interaction-only regime this term exists to detect -- and on the real
    # descriptor matrices produced objective values in the thousands.)
    synergy_values: list[float] = []
    for partner in selected:
        try:
            pairwise: PairwiseSynergy = compute_pairwise_synergy(
                X, y, index, int(partner), bins=bins, seed=seed
            )
        except Exception:
            continue
        synergy_values.append(float(pairwise.interaction_mi))
    synergy = float(np.max(synergy_values)) if synergy_values else 0.0

    # Redundancy is the shared information with the context, expressed through
    # the conditional MI *drop* relative to the marginal plus the absolute
    # partial-correlation mass. It is what the legacy hard |r| cut approximated.
    if context.size:
        correlations = np.abs(
            np.asarray(
                [_partial_correlation(column, X[:, int(j)], context) for j in selected],
                dtype=np.float64,
            )
        )
        correlations = correlations[np.isfinite(correlations)]
        correlation_mass = float(correlations.max()) if correlations.size else 0.0
    else:
        correlation_mass = 0.0
    redundancy = max(0.0, marginal - conditional) + correlation_mass


    stability = float(stability_cache.get(index, 0.0))
    # Stability is nominally centred on ``0.5`` so it can act as a *bonus* for
    # persistently signed features and a *penalty* for unstable ones. In practice
    # it saturates: ``_stability`` pins to ``1.0`` for 80-96% of real descriptors
    # (mean 0.96 on ESOL, 0.99 on FreeSolv), so this term is a large *common*
    # offset rather than a discriminator, and ``siren_full`` and
    # ``siren_no_stability`` select the same features. Reported here for audit;
    # see ``tests/test_recommend.py::test_stability_term_saturates_*``.
    return {
        "relevance": conditional,
        "synergy": synergy,
        "stability": 2.0 * (stability - 0.5),
        "redundancy": redundancy,
        "marginal_mi": marginal,
        "suppression_gain": conditional - marginal,
        "raw_stability": stability,
    }




def _support_mask(features: np.ndarray, min_support: int) -> np.ndarray:
    """Flag columns with enough distinct values to support an MI estimate.

    A column whose number of *distinct* values is at most ``min_support`` cannot
    carry a reliable association estimate: the discretiser collapses it to one
    non-empty bin per level and ``mutual_info_*`` then scores each level from
    only a handful of rows. On chemistry descriptors this is not a corner case.

    Measured on the ChEMBL subsets, every marginally-strong descriptor loses the
    comparison to a rare fragment counter. ``fr_azo`` is selected in 60% of
    folds on bbbp and ``fr_tetrazole`` in 38% on bace, while the scorecard's
    ``TPSA``/``BertzCT``/``MaxAbsEStateIndex`` -- which are monotone transforms
    of the target and score +1.0 correlation -- are dropped. With a few hundred
    molecules, ``fr_tetrazole`` is 1 for a handful of rows and 0 for the rest,
    so its MI estimate is driven by those few rows. The greedy objective is
    maximised by whichever rare feature happened to look associated *in that
    training split*, which is why SIREN's selection frequency for such features
    swings between 0.16 and 0.76 across seeds while the scorecard's picks sit at
    exactly 1.0. Screening them out is what makes the conditional information
    terms compete on features that can actually support an estimate.

    The screen is deliberately a *precondition*, not another weighted penalty:
    it removes candidates from consideration rather than trading off against
    relevance, so a screened feature can never be admitted on a spurious boost.

    Args:
        features: ``[n_samples, n_features]`` finite matrix.
        min_support: Minimum number of distinct values a column must take.
            ``0`` or ``1`` disables the screen.

    Returns:
        Boolean mask of length ``features.shape[1]``; ``True`` means usable.
    """
    n_features = features.shape[1]
    if min_support <= 1:
        return np.ones(n_features, dtype=bool)
    mask = np.empty(n_features, dtype=bool)
    for index in range(n_features):
        mask[index] = np.unique(features[:, index]).size >= min_support
    # Never screen everything out: an all-constant matrix is the caller's
    # problem, but returning an empty candidate set would silently produce a
    # report with no nominations and no explanation.
    if not mask.any():
        return np.ones(n_features, dtype=bool)
    return mask


def _stability_batch(
    features: np.ndarray,
    target: np.ndarray,
    rng: np.random.Generator,
    resamples: int,
    subsample_fraction: float,
) -> np.ndarray:
    """Resample persistence for every column at once.

    Spearman correlation is Pearson correlation on ranks, so the ranks are
    computed once per column and every resample is then a single vectorised
    dot product over all columns. The original loop issued one
    ``scipy.stats.spearmanr`` call per (column, resample) pair -- 3056 calls on
    a 191-descriptor matrix -- and dominated the remaining runtime after the MI
    terms were batched.

    The resampling sequence is deliberately identical to the original
    per-column routine (columns in order, then resamples, all drawn from one
    generator), so stability values -- and therefore every downstream
    selection made with them -- remain reproducible against earlier runs.

    Returns:
        Array of length ``n_features`` holding the fraction of resamples whose
        association has the same sign as the full-sample association, in
        ``[0, 1]``.
    """
    n_samples, n_features = features.shape
    if resamples < 1 or n_samples < 4:
        return np.zeros(n_features, dtype=np.float64)

    ranks_x = stats.rankdata(features, axis=0)
    ranks_y = stats.rankdata(target)
    centred_y = ranks_y - ranks_y.mean()
    denominator_y = float((centred_y**2).sum())
    size = max(3, int(round(subsample_fraction * n_samples)))
    size = min(size, n_samples)

    stability = np.zeros(n_features, dtype=np.float64)
    for index in range(n_features):
        ranks = ranks_x[:, index]
        centred = ranks - ranks.mean()
        denominator_x = float((centred**2).sum())
        if denominator_y <= 1e-12 or denominator_x <= 1e-12:
            continue
        full = float((centred * centred_y).sum() / np.sqrt(denominator_x * denominator_y))
        if abs(full) < 1e-12:
            continue
        agreements = 0
        total = 0
        for _ in range(resamples):
            indices = rng.choice(n_samples, size=size, replace=False)
            sampled_ranks = ranks[indices]
            sampled_y = ranks_y[indices]
            centred_sample = sampled_ranks - sampled_ranks.mean()
            centred_sample_y = sampled_y - sampled_y.mean()
            denominator = np.sqrt(
                float((centred_sample**2).sum())
                * float((centred_sample_y**2).sum())
            )
            sampled = (
                float((centred_sample * centred_sample_y).sum() / denominator)
                if denominator > 1e-12
                else 0.0
            )
            if sampled == 0.0 and np.ptp(features[indices, index]) == 0:
                continue
            total += 1
            if np.sign(sampled) == np.sign(full):
                agreements += 1
        if total:
            stability[index] = agreements / total
    return stability


def _duplicate_mask(features: np.ndarray, threshold: float) -> np.ndarray:
    """Flag columns that are near-duplicates of an earlier eligible column.

    The first column of each near-duplicate group is kept so the selection can
    still use that variable; every later copy is screened out. A standardised
    Gram matrix gives all pairwise Pearson correlations in one product, which is
    cheap relative to the MI terms that follow.

    Args:
        features: ``[n_samples, n_features]`` finite matrix.
        threshold: Absolute correlation at or above which two columns are
            treated as the same variable. Values ``>= 1`` disable the screen.

    Returns:
        Boolean mask of length ``n_features``; ``False`` marks a screened copy.
    """
    n_samples, n_features = features.shape
    if threshold >= 1.0 or n_features < 2:
        return np.ones(n_features, dtype=bool)

    centred = features - features.mean(axis=0, keepdims=True)
    scale = features.std(axis=0, keepdims=True)
    scale = np.where(scale < 1e-12, np.inf, scale)
    standardised = centred / scale
    correlation = standardised.T @ standardised / n_samples

    keep = np.ones(n_features, dtype=bool)
    for index in range(n_features - 1):
        if not keep[index]:
            continue
        duplicates = np.flatnonzero(
            np.abs(correlation[index, index + 1 :]) >= threshold
        )
        keep[index + 1 + duplicates] = False
    return keep


def _synergy_seed_pair(
    features: np.ndarray,
    target: np.ndarray,
    eligible: Sequence[int],
    *,
    bins: int,
    seed: int,
    n_pairs: int,
    threshold: float,
) -> tuple[float, int, int] | None:
    """Screen candidate pairs for the strongest interaction lift.

    All pairs are enumerated when the eligible set is small; otherwise
    ``n_pairs`` pairs are sampled without replacement. A pair is only returned
    when its raw interaction lift clears ``threshold`` *and* stands out from the
    sampled distribution (``median + 3 * MAD``), which stops the screen from
    seeding on the largest of many noise-level fluctuations.

    Returns:
        ``(synergy, first, second)`` or ``None`` when no pair qualifies.
    """
    pool = np.asarray(eligible, dtype=np.int64)
    if pool.size < 2:
        return None
    rng = np.random.default_rng(seed + 7919)
    if pool.size <= 40:
        pairs = [
            (int(pool[first]), int(pool[second]))
            for first in range(pool.size)
            for second in range(first + 1, pool.size)
        ]
    else:
        pairs = []
        for _ in range(max(1, n_pairs)):
            first, second = rng.choice(pool, size=2, replace=False)
            pairs.append((int(first), int(second)))

    scored: list[tuple[float, int, int]] = []
    for first, second in pairs:
        try:
            pair = compute_pairwise_synergy(
                features, target, first, second, bins=bins, seed=seed
            )
        except Exception:
            continue
        # Only pairs that actually carry joint information are candidates. The
        # relative lift is a ratio, so a pair whose joint MI is pure estimator
        # noise can still reach a large ratio; requiring a minimum joint MI in
        # nats removes exactly that failure mode.
        if np.isfinite(pair.interaction_mi) and pair.interaction_mi >= 0.15:
            scored.append((float(pair.interaction_mi), first, second))
    if not scored:
        return None

    values = np.asarray([item[0] for item in scored], dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    best = max(scored, key=lambda item: item[0])
    if best[0] < threshold or best[0] < median + 3.0 * mad:
        return None
    return best


def _first_occurrence(features: np.ndarray) -> np.ndarray:
    """Map every column to the index of the first column with identical values.

    Identical columns are indistinguishable evidence, so they must receive
    identical scores. The batched estimators break exact ties with a random
    jitter that is independent per column; without this normalisation a
    duplicated descriptor could outrank its own source purely because of that
    jitter. Mapping to the first occurrence restores the original per-column
    behaviour, where the same input produced the same estimate.
    """
    _, first, inverse = np.unique(
        features, axis=1, return_index=True, return_inverse=True
    )
    return np.asarray(first, dtype=np.int64)[np.asarray(inverse, dtype=np.int64)]


def _share_duplicate_values(
    values: np.ndarray,
    indices: np.ndarray,
    representative: np.ndarray,
) -> np.ndarray:
    """Copy each value from the first column identical to it.

    ``values`` is aligned with ``indices``; the returned array keeps that
    alignment but replaces every duplicate's estimate with its source's.
    """
    lookup = {int(index): position for position, index in enumerate(indices)}
    shared = np.empty(len(indices), dtype=np.float64)
    for position, index in enumerate(indices):
        source = int(representative[index])
        # Once the source column has already been selected it is no longer a
        # candidate. The duplicate's own estimate is then the only available
        # value -- and it is no longer competing with its source anyway.
        shared[position] = (
            values[lookup[source]] if source in lookup else values[position]
        )
    return shared


def _standardise_columns(matrix: np.ndarray) -> np.ndarray:
    """Centre and scale every column, mapping degenerate columns to zeros.

    The partial-correlation mass in the redundancy term is undefined for a
    column that is constant inside a fold. Scaling by ``+inf`` is the cheapest
    exact way to map such a column to an all-zero vector, which contributes
    zero correlation rather than ``NaN``.
    """
    centred = matrix - matrix.mean(axis=0, keepdims=True)
    scale = matrix.std(axis=0, keepdims=True)
    scale = np.where(scale < 1e-12, np.inf, scale)
    return centred / scale


def _batch_synergy_against_partner(
    features: np.ndarray,
    target: np.ndarray,
    candidates: np.ndarray,
    partner: int,
    *,
    bins: int,
    seed: int,
    discrete_y: bool,
    target_in: np.ndarray,
) -> np.ndarray:
    """Interaction information of every candidate against one fixed partner.

    This is the vectorised equivalent of calling
    :func:`~chemistry_framework.feature_analysis.synergy.compute_pairwise_synergy`
    once per candidate in ``candidates``. The orthogonalised product basis is
    stacked over all candidates and measured with a single batched estimator
    call instead of ``len(candidates)`` single-column calls.

    * the interaction term stacks the orthogonalised product basis of every
      candidate against the partner and measures it in one call.

    The collinearity gate mirrors ``compute_pairwise_synergy``: near-duplicate
    pairs have no beyond-marginal signal, so their interaction term is zeroed.

    Returns:
        Array aligned with ``candidates`` holding the interaction information in
        nats.
    """
    candidate_index = np.asarray(candidates, dtype=np.int64)
    if candidate_index.size == 0:
        return np.zeros(0, dtype=np.float64)

    partner_column = features[:, partner]
    candidate_columns = features[:, candidate_index]

    # Orthogonalised interaction basis: centre/scale the product, then remove
    # the component explained by the two raw margins. A plain product of two
    # near-collinear columns is dominated by the squared marginal signal, which
    # would masquerade as synergy.
    interaction = np.empty_like(candidate_columns)
    partner_centred = partner_column - partner_column.mean()
    partner_std = float(partner_column.std())
    for position, index in enumerate(candidate_index):
        column = candidate_columns[:, position]
        column_std = float(column.std())
        if column_std < 1e-12 or partner_std < 1e-12:
            interaction[:, position] = 0.0
            continue
        product = ((column - column.mean()) / column_std) * (
            partner_centred / partner_std
        )
        interaction[:, position] = _residualise(
            product, np.column_stack([column, partner_column])
        )
    if not discrete_y:
        interaction = np.column_stack(
            [
                _bin_series(interaction[:, position], bins)
                for position in range(interaction.shape[1])
            ]
        )
    estimator = mutual_info_classif if discrete_y else mutual_info_regression
    try:
        interaction_mi = np.asarray(
            estimator(interaction, target_in, random_state=seed), dtype=np.float64
        )
    except Exception:
        interaction_mi = np.zeros(interaction.shape[1], dtype=np.float64)
    interaction_mi[~np.isfinite(interaction_mi)] = 0.0

    # Vectorised Pearson correlation for the collinearity gate.
    denominator = candidate_columns.std(axis=0) * partner_std
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.where(
            denominator > 1e-12,
            (
                (candidate_columns - candidate_columns.mean(axis=0))
                * partner_centred[:, None]
            ).mean(axis=0)
            / denominator,
            0.0,
        )
    correlation[~np.isfinite(correlation)] = 0.0
    gated = np.abs(correlation) >= 0.999
    if gated.any():
        interaction_mi = np.where(gated, 0.0, interaction_mi)

    # The objective consumes the absolute interaction information in nats, which
    # is on the same scale as the conditional-MI relevance term and cannot
    # explode. The relative lift ``PairwiseSynergy.synergy`` remains available
    # for reporting and for the pairwise tests.
    return interaction_mi


def recommend_features(
    X: np.ndarray,
    y: np.ndarray,
    *,
    feature_names: Sequence[str] | None = None,
    k: int | None = None,
    weights: Mapping[str, float] | None = None,
    bins: int = DEFAULT_BINS,
    stability_resamples: int = 16,
    subsample_fraction: float = 0.8,
    min_objective: float | None = None,
    min_support: int = DEFAULT_MIN_SUPPORT,
    duplicate_threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
    pairwise_seed: bool = True,
    seed_pairs: int = DEFAULT_SEED_PAIRS,
    seed_marginal_threshold: float = DEFAULT_SEED_MARGINAL_THRESHOLD,
    seed_synergy_threshold: float = DEFAULT_SEED_SYNERGY_THRESHOLD,
    seed: int = 0,
) -> RecommendationReport:
    """Run SIREN forward selection and return an auditable report.

    Selection is greedy and forward: at each step every remaining candidate is
    scored with :func:`_candidate_components` given the context selected so far,
    and the candidate maximising ``J`` is nominated. This makes the procedure
    explicitly multivariate -- a feature is judged by the information it adds
    *beyond* what has already been chosen, which is what allows suppressor and
    synergy features to enter the set that a marginal scorecard would drop.

    Args:
        X: Feature matrix ``[n_samples, n_features]``; must be finite.
        y: Target vector of length ``n_samples`` (continuous or binary).
        feature_names: Optional names aligned with the columns of ``X``.
        k: Maximum number of features to nominate. ``None`` means "use the
            data-driven stopping rule" (stop when the best ``J`` falls to or
            below ``min_objective``, i.e. when no candidate adds net evidence).
        weights: Overrides for ``DEFAULT_WEIGHTS`` keys.
        bins: Quantile bins used for MI discretisation.
        stability_resamples: Number of subsamples used for the stability term.
            ``0`` disables the term (it then contributes nothing).
        subsample_fraction: Fraction of rows drawn per stability resample.
        min_objective: Optional objective floor. When provided, selection stops
            as soon as the best remaining ``J`` falls to or below this value.
            When ``None`` (default) *every* candidate is ranked, so the report
            orders the full feature list by evidence; truncate downstream with
            ``k`` or by inspecting the objective curve.
        seed: Seed for all stochastic components (stability resampling and the
            MI estimator's tie-breaking).
        min_support: Minimum number of *distinct* values a column must take to
            be eligible. Rare fragment counters are the pathological case: with a
            few hundred molecules an ``fr_*`` indicator is 1 for a handful of
            rows, so its discretised MI is estimated from those few rows and the
            greedy objective chases estimator noise rather than signal. Features
            skipped this way are listed in ``screened_features`` for audit.
            ``0`` or ``1`` disables the screen and reproduces the pre-screen
            behaviour (see ``siren_no_support_screen`` in the E-REC arms).

    Returns:
        A :class:`RecommendationReport` with per-feature justifications.

    Raises:
        ValueError: If inputs are malformed or ``k`` is negative.
    """
    features = np.asarray(X, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    if features.ndim != 2:
        raise ValueError("X must be a 2D array")
    if features.shape[0] != target.size:
        raise ValueError("X and y must have matching row counts")
    if not np.isfinite(features).all():
        raise ValueError("X must contain only finite values")
    if not np.isfinite(target).all():
        raise ValueError("y must contain only finite values")
    if k is not None and k < 0:
        raise ValueError("k must be non-negative")

    names = list(feature_names) if feature_names is not None else [
        f"column_{i}" for i in range(features.shape[1])
    ]
    if len(names) != features.shape[1]:
        raise ValueError("feature_names must have one entry per column of X")

    resolved = dict(DEFAULT_WEIGHTS)
    if weights:
        unknown = set(weights) - set(DEFAULT_WEIGHTS)
        if unknown:
            raise ValueError(f"unknown weight keys: {sorted(unknown)}")
        resolved.update({key: float(value) for key, value in weights.items()})

    rng = np.random.default_rng(seed)
    # Stability is context-independent by design: it measures how robust a
    # feature's *own* association is, so it is computed once for all columns.
    if stability_resamples > 0:
        raw_stability = _stability_batch(
            features, target, rng, stability_resamples, subsample_fraction
        )
    else:
        raw_stability = np.zeros(features.shape[1], dtype=np.float64)
    stability_cache = {
        index: float(raw_stability[index]) for index in range(features.shape[1])
    }

    # ------------------------------------------------------------------
    # Context-independent pre-computation.
    #
    # The original implementation called the mutual-information estimator once
    # per (candidate, greedy step), plus four more times per (candidate,
    # selected partner) for the synergy term. On the 191-descriptor ChEMBL
    # matrices that is roughly two thousand estimator calls per fold, and a
    # single nested-CV arm took over ten minutes. Every quantity below depends
    # only on the *data*, not on the selection context, so it is hoisted out of
    # the loop and computed once:
    #
    #   * the discretised feature matrix and target (one rank transform each),
    #   * the marginal MI of every column (the ``marginal`` vector),
    #   * the stability cache above.
    #
    # Conditional MI and synergy still depend on the context, but both are now
    # evaluated *for all remaining candidates at once*: one least-squares solve
    # for the residual matrix and one batched MI call per term. This keeps the
    # estimator and the mathematics identical while removing the per-candidate
    # interpreter overhead.
    # ------------------------------------------------------------------
    discrete_y = _is_classification(target)
    binned = (
        None
        if discrete_y
        else np.column_stack(
            [_bin_series(features[:, index], bins) for index in range(features.shape[1])]
        )
    )
    target_in = target if discrete_y else _bin_series(target, bins)
    marginal = _mutual_information(features, target, bins=bins, seed=seed)
    representative = _first_occurrence(features)
    marginal = marginal[representative]

    selected: list[int] = []
    usable = _support_mask(features, min_support) & _duplicate_mask(
        features, duplicate_threshold
    )
    eligible = [index for index in range(features.shape[1]) if usable[index]]
    remaining = list(eligible)
    nominations: list[FeatureNomination] = []
    objective_curve: list[float] = []
    limit = features.shape[1] if k is None else min(k, features.shape[1])

    # ---- synergy seeding for interaction-only signals ----
    #
    # Greedy forward selection can only reward a synergy term once one member of
    # the pair is already in the context. With an empty context, a pure
    # interaction pair (neither member marginally informative) is therefore
    # invisible and the first slot goes to whichever column has the largest
    # marginal relevance -- which for such data is estimator noise. When no
    # eligible candidate carries marginal information, screen candidate pairs and
    # seed the selection with the strongest interaction instead.
    def _marginal_noise_floor() -> float:
        """Permutation floor for "no candidate carries marginal information".

        The raw MI of a noise column is not zero on a finite sample, and the
        maximum over ``p`` columns grows with ``p``. Comparing the observed
        maximum against an absolute cut-off therefore misfires on wide
        interaction-only designs. Permuting the target once, re-estimating every
        marginal MI, and taking the largest value gives the sampling floor that
        the observed maximum must exceed before the data is considered to carry
        a marginal signal.
        """
        rng_floor = np.random.default_rng(seed + 104729)
        shuffled = target[rng_floor.permutation(target.size)]
        null = _mutual_information(features, shuffled, bins=bins, seed=seed)
        return float(np.max(null[eligible])) * 1.5

    best_marginal = float(np.max(marginal[eligible])) if eligible else 0.0
    # ``0.15`` nats is a cheap upper bound on "no credible marginal signal".
    # Measured on the molecular benchmarks even the weakest endpoint (BBBP,
    # 0.21 nats) clears it, so real datasets skip the permutation floor and the
    # pair screen entirely; only genuinely interaction-dominated designs pay for
    # them.
    if (
        pairwise_seed
        and resolved["synergy"] > 0.0
        and len(eligible) >= 2
        and (k is None or k >= 2)
        and best_marginal < 0.15
        and best_marginal < max(seed_marginal_threshold, _marginal_noise_floor())
    ):
        seed_pair = _synergy_seed_pair(
            features,
            target,
            eligible,
            bins=bins,
            seed=seed,
            n_pairs=seed_pairs,
            threshold=seed_synergy_threshold,
        )
        if seed_pair is not None:
            _, first_index, second_index = seed_pair
            for index in (first_index, second_index):
                parts = _candidate_components(
                    features,
                    target,
                    index,
                    selected,
                    bins=bins,
                    seed=seed,
                    stability_cache=stability_cache,
                    discrete_y=discrete_y,
                )
                score = (
                    resolved["relevance"] * parts["relevance"]
                    + resolved["synergy"] * parts["synergy"]
                    + resolved["stability"] * parts["stability"]
                    - resolved["redundancy"] * parts["redundancy"]
                )
                nominations.append(
                    FeatureNomination(
                        name=names[index],
                        index=int(index),
                        rank=len(selected) + 1,
                        objective=float(score),
                        relevance=float(parts["relevance"]),
                        synergy=float(parts["synergy"]),
                        stability=float(parts["raw_stability"]),
                        redundancy=float(parts["redundancy"]),
                        marginal_mi=float(parts["marginal_mi"]),
                        suppression_gain=float(parts["suppression_gain"]),
                        dominant_term="synergy_seed",
                    )
                )
                objective_curve.append(float(score))
                selected.append(int(index))
                remaining.remove(int(index))

    while remaining and len(selected) < limit:
        context = np.asarray(selected, dtype=np.int64)
        candidate_index = np.asarray(remaining, dtype=np.int64)

        # ---- conditional relevance: one solve, one batched MI call ----
        if context.size:
            design = np.column_stack(
                [np.ones(features.shape[0]), features[:, context]]
            )
            coefficients, *_ = np.linalg.lstsq(design, features, rcond=None)
            residuals = features - design @ coefficients
            # For a categorical target the residual of ``y`` is not defined, so
            # only the descriptor is residualised. Keeping ``y`` discrete also
            # keeps the estimator at ``mutual_info_classif``, which bounds the
            # conditional value by ``H(y)`` exactly like the marginal value.
            residual_target = (
                target if discrete_y else _residualise(target, features[:, context])
            )
            conditional = _mutual_information(
                residuals[:, candidate_index],
                residual_target,
                bins=bins,
                seed=seed,
            )
        else:
            conditional = marginal[candidate_index]
        conditional = _share_duplicate_values(
            conditional, candidate_index, representative
        )

        # ---- synergy: one batched pair of MI calls per selected partner ----
        if context.size and resolved["synergy"] != 0.0:
            synergy = np.stack(
                [
                    _batch_synergy_against_partner(
                        features,
                        target,
                        candidate_index,
                        int(partner),
                        bins=bins,
                        seed=seed,
                        discrete_y=discrete_y,
                        target_in=target_in,
                    )
                    for partner in selected
                ]
            ).max(axis=0)
            synergy = _share_duplicate_values(
                synergy, candidate_index, representative
            )
        else:
            synergy = np.zeros(candidate_index.size, dtype=np.float64)

        # ---- redundancy: partial-correlation mass against the context ----
        #
        # This term is disabled by default, so it is evaluated with the original
        # per-candidate routine rather than the batched path: only the research
        # arms (``siren_legacy``, explicit ``weights={"redundancy": 1.0}``) pay
        # for it, and keeping the exact historical computation means their
        # archived ablation numbers remain reproducible.
        if context.size and resolved["redundancy"] != 0.0:
            correlation_mass = np.asarray(
                [
                    max(
                        (
                            abs(_partial_correlation(column, features[:, int(partner)], context))
                            for partner in selected
                        ),
                        default=0.0,
                    )
                    for column in (features[:, index] for index in candidate_index)
                ],
                dtype=np.float64,
            )
            correlation_mass[~np.isfinite(correlation_mass)] = 0.0
            redundancy = (
                np.maximum(0.0, marginal[candidate_index] - conditional)
                + correlation_mass
            )
        else:
            redundancy = np.zeros(candidate_index.size, dtype=np.float64)

        stability_component = np.asarray(
            [2.0 * (stability_cache[index] - 0.5) for index in candidate_index],
            dtype=np.float64,
        )

        score = (
            resolved["relevance"] * conditional
            + resolved["synergy"] * synergy
            + resolved["stability"] * stability_component
            - resolved["redundancy"] * redundancy
        )
        best_position = int(np.argmax(score))
        best_index = int(candidate_index[best_position])
        best_score = float(score[best_position])
        best_parts: dict[str, float] = {
            "relevance": float(conditional[best_position]),
            "synergy": float(synergy[best_position]),
            "stability": float(stability_component[best_position]),
            "redundancy": float(redundancy[best_position]),
            "marginal_mi": float(marginal[best_index]),
            "suppression_gain": float(
                conditional[best_position] - marginal[best_index]
            ),
            "raw_stability": float(stability_cache[best_index]),
        }

        # The redundancy term is disabled by default and therefore excluded
        # from the score. It is still reported for the *nominated* feature so
        # the audit trail keeps explaining how much a pick duplicates the
        # context -- the original implementation computed it for every
        # candidate, which is wasted work when its weight is zero.
        if resolved["redundancy"] == 0.0:
            base = max(
                0.0,
                float(marginal[best_index]) - float(conditional[best_position]),
            )
            if context.size:
                mass = max(
                    (
                        abs(
                            _partial_correlation(
                                features[:, best_index],
                                features[:, int(partner)],
                                context,
                            )
                        )
                        for partner in selected
                    ),
                    default=0.0,
                )
                if not np.isfinite(mass):
                    mass = 0.0
                base += mass
            best_parts["redundancy"] = float(base)

        if min_objective is not None and best_score <= min_objective:
            break

        contributions = {
            "relevance": resolved["relevance"] * best_parts["relevance"],
            "synergy": resolved["synergy"] * best_parts["synergy"],
            "stability": resolved["stability"] * best_parts["stability"],
        }
        dominant = max(contributions, key=contributions.get)
        if contributions[dominant] <= 0:
            dominant = "redundancy_penalised"

        nominations.append(
            FeatureNomination(
                name=names[best_index],
                index=int(best_index),
                rank=len(selected) + 1,
                objective=float(best_score),
                relevance=float(best_parts["relevance"]),
                synergy=float(best_parts["synergy"]),
                stability=float(best_parts["raw_stability"]),
                redundancy=float(best_parts["redundancy"]),
                marginal_mi=float(best_parts["marginal_mi"]),
                suppression_gain=float(best_parts["suppression_gain"]),
                dominant_term=dominant,
            )
        )
        objective_curve.append(float(best_score))
        selected.append(int(best_index))
        remaining.remove(int(best_index))

    return RecommendationReport(
        method="siren",
        k=int(k) if k is not None else len(selected),
        n_candidates=int(features.shape[1]),
        n_selected=len(nominations),
        weights={key: float(value) for key, value in resolved.items()},
        bins=int(bins),
        stability_resamples=int(stability_resamples),
        nominations=tuple(nominations),
        objective_curve=tuple(objective_curve),
        screened_features=tuple(
            names[index] for index in range(features.shape[1]) if not usable[index]
        ),
        stopping_rule=(
            f"fixed k={k}"
            if k is not None
            else (
                f"objective_floor={min_objective}"
                if min_objective is not None
                else "full_ranking"
            )
        ),
    )



def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation that returns ``0.0`` for degenerate input.

    Real descriptor matrices routinely contain columns that are constant within
    a fold (a rare substructure flag that only fires outside the training rows).
    ``scipy`` raises ``ConstantInputWarning`` and returns ``NaN`` for those, which
    would otherwise leak warnings into benchmark logs and silently poison any
    downstream ``max``/``mean``. The correlation is genuinely undefined here, so
    ``0.0`` (no linear association) is the honest neutral value.
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


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation that returns ``0.0`` for degenerate input.

    See :func:`_safe_pearson` for why the degenerate case is mapped to zero
    rather than propagated as ``NaN``.
    """
    if a.size < 2 or b.size < 2:
        return 0.0
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return 0.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            value, _ = stats.spearmanr(a, b)
    except Exception:
        return 0.0
    return float(value) if np.isfinite(value) else 0.0


def scorecard_ranking(
    X: np.ndarray,
    y: np.ndarray,
    *,
    feature_names: Sequence[str] | None = None,
    redundancy_threshold: float = 0.8,
    max_features: int | None = None,
) -> RecommendationReport:
    """Legacy marginal scorecard, re-expressed as a SIREN report.

    Reproduces the historical ``select_evidence_recommendation`` behaviour --
    univariate Spearman ranking followed by greedy ``|r|`` redundancy pruning --
    but emits the same :class:`RecommendationReport` type as
    :func:`recommend_features`. This keeps the old results bit-comparable while
    letting the benchmark harness treat both methods uniformly.
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
    scores = []
    for index in range(features.shape[1]):
        rho = _safe_spearman(features[:, index], target)
        scores.append(abs(rho))
    order = np.argsort(-np.asarray(scores))

    selected: list[int] = []
    for index in order:
        index = int(index)
        if max_features is not None and len(selected) >= max_features:
            break
        if selected:
            correlations = np.abs(
                [_safe_pearson(features[:, index], features[:, other]) for other in selected]
            )
            if correlations.size and float(correlations.max()) > redundancy_threshold:
                continue
        selected.append(index)

    # Re-prune against the survivor set, mirroring the legacy greedy pass.
    kept: list[int] = []
    for index in selected:
        if kept:
            correlations = np.abs(
                [_safe_pearson(features[:, index], features[:, other]) for other in kept]
            )
            if correlations.size and float(correlations.max()) > redundancy_threshold:
                continue
        kept.append(index)

    nominations = tuple(
        FeatureNomination(
            name=names[index],
            index=int(index),
            rank=position + 1,
            objective=float(scores[index]),
            relevance=float(scores[index]),
            synergy=0.0,
            stability=0.0,
            redundancy=0.0,
            marginal_mi=float(scores[index]),
            suppression_gain=0.0,
            dominant_term="relevance",
        )
        for position, index in enumerate(kept)
    )
    return RecommendationReport(
        method="scorecard",
        k=len(nominations),
        n_candidates=int(features.shape[1]),
        n_selected=len(nominations),
        weights=dict(DEFAULT_WEIGHTS),
        bins=DEFAULT_BINS,
        stability_resamples=0,
        nominations=nominations,
        objective_curve=tuple(item.objective for item in nominations),
        stopping_rule=f"redundancy_threshold={redundancy_threshold}",
        interpretation_note=(
            "Legacy marginal Spearman scorecard retained for backward "
            "comparability; it is univariate and applies a hard |r| redundancy "
            "cut-off, so it cannot represent suppression or synergy."
        ),
    )
