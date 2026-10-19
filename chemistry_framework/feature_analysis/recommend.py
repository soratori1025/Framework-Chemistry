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
) -> dict[str, float]:
    """Compute each objective term for one candidate given the context ``S``."""
    column = X[:, index]
    context = np.asarray(selected, dtype=int)

    marginal = float(_mutual_information(column.reshape(-1, 1), y, bins=bins, seed=seed)[0])
    if context.size:
        residual_feature = _residualise(column, X[:, context])
        residual_target = _residualise(y, X[:, context])
        conditional = float(
            _mutual_information(
                residual_feature.reshape(-1, 1), residual_target, bins=bins, seed=seed
            )[0]
        )
    else:
        conditional = marginal

    # Synergy is the *incremental* interaction gain of adding the candidate to
    # the context: I(candidate, context; y) - I(context; y) - I(candidate; y),
    # normalised by the candidate's own marginal MI. The raw
    # `joint - singles` form goes negative whenever the context already explains
    # the target, which would wrongly punish genuine synergy partners; the
    # incremental form isolates the interaction lift itself.
    synergy_values: list[float] = []
    for partner in selected:
        try:
            pairwise: PairwiseSynergy = compute_pairwise_synergy(
                X, y, index, int(partner), bins=bins, seed=seed
            )
        except Exception:
            continue
        scale = max(
            pairwise.first_mi,
            pairwise.second_mi,
            np.mean([pairwise.first_mi, pairwise.second_mi]),
            1e-6,
        )
        synergy_values.append(pairwise.synergy / scale)
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
    # feature's *own* association is, so it is computed once and cached.
    stability_cache = {
        index: (
            _stability(
                features[:, index], target, rng, stability_resamples, subsample_fraction
            )
            if stability_resamples > 0
            else 0.0
        )
        for index in range(features.shape[1])
    }

    selected: list[int] = []
    usable = _support_mask(features, min_support)
    eligible = [index for index in range(features.shape[1]) if usable[index]]
    remaining = list(eligible)
    nominations: list[FeatureNomination] = []
    objective_curve: list[float] = []
    limit = features.shape[1] if k is None else min(k, features.shape[1])

    while remaining and len(selected) < limit:
        best_index = None
        best_score = -np.inf
        best_parts: dict[str, float] | None = None
        for index in remaining:
            parts = _candidate_components(
                features,
                target,
                index,
                selected,
                bins=bins,
                seed=seed,
                stability_cache=stability_cache,
            )
            score = (
                resolved["relevance"] * parts["relevance"]
                + resolved["synergy"] * parts["synergy"]
                + resolved["stability"] * parts["stability"]
                - resolved["redundancy"] * parts["redundancy"]
            )
            if score > best_score:
                best_score = score
                best_index = index
                best_parts = parts

        if best_index is None or best_parts is None:
            break
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
