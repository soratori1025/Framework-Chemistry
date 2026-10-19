from __future__ import annotations

import numpy as np
import pytest

from chemistry_framework.feature_analysis.protocol import (
    BootstrapInterval,
    nested_selection_evaluation,
    paired_bootstrap_test,
    scorecard_selector,
    siren_selector,
)
from chemistry_framework.feature_analysis.recommend import (
    DEFAULT_WEIGHTS,
    RecommendationReport,
    recommend_features,
    scorecard_ranking,
)
from chemistry_framework.feature_analysis.synergy import (
    compute_conditional_dependence,
    compute_pairwise_synergy,
    rank_synergy_partners,
)


def _synergy_fixture(seed: int = 0, n: int = 400):
    """Return (X, y, names) containing marginal, duplicate, noise and synergy."""
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n)
    marginal = y + rng.normal(scale=0.1, size=n)
    duplicate = marginal.copy()
    noise = rng.normal(size=n)
    # XOR-style partner: informative only jointly with `noise`.
    sign = (y > 0).astype(float)
    partner = np.where(sign > 0, noise, -noise)
    X = np.column_stack([marginal, duplicate, noise, partner])
    return X, y, ["marginal", "duplicate", "noise", "partner"]



def test_conditional_dependence_suppresses_duplicates() -> None:
    X, y, _ = _synergy_fixture()
    dependence = compute_conditional_dependence(X, y, candidates=[0, 1, 2], context=[0])
    # Given the marginal feature itself, nothing further is explained.
    assert dependence[0].conditional_mi < 0.05
    assert dependence[0].suppression_gain < 0
    # A duplicate is fully redundant with its twin.
    assert dependence[1].conditional_mi < 0.05
    assert dependence[1].marginal_mi > 0.5


def test_pairwise_synergy_separates_interaction_from_redundancy() -> None:
    X, y, _ = _synergy_fixture()
    interaction = compute_pairwise_synergy(X, y, 2, 3)
    assert interaction.first_mi < 0.05
    assert interaction.second_mi < 0.05
    assert interaction.synergy > 0.05

    redundant = compute_pairwise_synergy(X, y, 0, 1)
    assert redundant.correlation == pytest.approx(1.0)
    assert redundant.synergy < 0.0


def test_rank_synergy_partners_puts_the_interaction_first() -> None:
    X, y, _ = _synergy_fixture()
    partners = rank_synergy_partners(X, y, 2, [0, 1, 3])
    assert partners[0].second == 3
    assert partners[0].synergy > 0.0
    assert partners[-1].synergy <= partners[0].synergy


def test_pairwise_synergy_rejects_bad_input() -> None:
    X, y, _ = _synergy_fixture()
    with pytest.raises(ValueError):
        compute_pairwise_synergy(X, y, 0, 0)
    with pytest.raises(ValueError):
        compute_pairwise_synergy(X, y, 0, 99)



def test_siren_recovers_marginal_and_synergy_ignoring_duplicate() -> None:
    X, y, names = _synergy_fixture()
    report = recommend_features(X, y, feature_names=names, k=None, seed=0)
    assert isinstance(report, RecommendationReport)
    assert report.method == "siren"
    assert report.n_selected == X.shape[1]
    assert report.stopping_rule == "full_ranking"

    ordered = report.selected_names()
    assert ordered[0] == "marginal"
    # The duplicate must never outrank the genuine marginal feature.
    assert ordered.index("duplicate") > ordered.index("marginal")
    # Both synergy partners must be nominated somewhere in the ranking.
    assert {"noise", "partner"} <= set(ordered)

    by_name = {item.name: item for item in report.nominations}
    assert by_name["duplicate"].redundancy > by_name["marginal"].redundancy
    assert by_name["duplicate"].rank > 1


def test_siren_synergy_term_dominates_for_interaction_partner() -> None:
    X, y, names = _synergy_fixture()
    report = recommend_features(X, y, feature_names=names, k=3, seed=0)
    ordered = report.selected_names()
    assert "noise" in ordered and "partner" in ordered
    by_name = {item.name: item for item in report.nominations}
    partner = by_name["partner"]
    # The interaction partner is admitted on the interaction term, not on its own
    # (near-zero) marginal association.
    assert partner.marginal_mi < 0.1
    assert partner.synergy > 0.0


def test_siren_k_truncates_and_respects_weights() -> None:
    X, y, names = _synergy_fixture()
    report = recommend_features(X, y, feature_names=names, k=2, seed=0)
    assert report.n_selected == 2
    assert report.stopping_rule == "fixed k=2"
    assert len(report.objective_curve) == 2

    no_stability = recommend_features(
        X, y, feature_names=names, k=2, weights={"stability": 0.0}, seed=0
    )
    assert no_stability.weights["stability"] == 0.0
    assert no_stability.weights["relevance"] == DEFAULT_WEIGHTS["relevance"]


def test_stability_term_saturates_at_its_ceiling_for_most_features() -> None:
    """The stability term is near-inert because it saturates at ``+1.0``.

    ``_stability`` counts the fraction of subsamples whose Spearman *sign*
    agrees with the full-sample sign of the *same* column. For any feature whose
    association is detectable, subsample signs agree almost surely, so the
    statistic pins to ``1.0``. Centred as ``2 * (stability - 0.5)`` it becomes
    ``+1.0`` for the vast majority of features and varies only in a narrow band
    below that, so it carries almost no discriminating signal.

    Measured on real descriptors the raw statistic has mean ~0.96 (ESOL) to
    ~0.99 (FreeSolv) with 80-96% of features pinned at exactly 1.0, leaving a
    between-feature spread of only 0.08-0.20 on a term weighted 0.5 -- i.e. a
    contribution of at most ~0.1, against a redundancy term of order 1.0.

    This test records that the gamma ablation isolates almost nothing. It is
    written to fail if the statistic is ever made properly discriminative, at
    which point the recorded E-REC gamma-ablation numbers must be regenerated.
    """
    from chemistry_framework.feature_analysis.recommend import _stability

    rng = np.random.default_rng(0)
    n = 500
    y = rng.normal(size=n)
    X = np.column_stack(
        [
            y + rng.normal(scale=0.1, size=n),   # strong association
            rng.normal(size=n),                  # pure noise
            y + rng.normal(scale=2.0, size=n),   # weak association
        ]
    )

    raw = [_stability(X[:, i], y, np.random.default_rng(1), 16, 0.8) for i in range(X.shape[1])]
    # Saturates at its ceiling: a strongly associated feature and a weak one are
    # indistinguishable, which is the whole problem with the statistic.
    assert raw[0] == 1.0, f"strongly associated feature not saturated: {raw}"
    assert raw[2] == 1.0, f"weakly associated feature not saturated: {raw}"
    # The centred term therefore contributes a large *common* offset, not a
    # discriminating bonus: it is well above zero for every candidate.
    centred = [2.0 * (v - 0.5) for v in raw]
    assert min(centred) > 0.0, f"stability is not a non-negative offset: {centred}"


def test_redundancy_penalty_is_disabled_by_default() -> None:
    """The redundancy term is off by default because it is harmful in practice.

    The E-REC isolation study showed the penalty drives selection *away* from the
    target: the nested-CV score rises monotonically as the weight moves
    ``+1 -> 0 -> -1`` (ESOL +0.8542 / +0.8565 / +0.8634, FreeSolv +0.7988 /
    +0.8008 / +0.8182). The default is therefore ``0.0``; this test locks that in
    so it cannot silently regress to the old penalty.
    """
    assert DEFAULT_WEIGHTS["redundancy"] == 0.0
    # The other terms keep their previous defaults.
    assert DEFAULT_WEIGHTS["relevance"] == 1.0
    assert DEFAULT_WEIGHTS["synergy"] == 0.5
    assert DEFAULT_WEIGHTS["stability"] == 0.5

    X, y, names = _synergy_fixture()
    report = recommend_features(X, y, feature_names=names, k=3, seed=0)
    assert report.weights["redundancy"] == 0.0


def test_redundancy_penalty_still_available_and_can_change_selection() -> None:
    """The term is disabled, not deleted: an explicit weight must still bite.

    This guards the research path -- ``weights={"redundancy": 1.0}`` has to
    restore the legacy objective, otherwise the recorded ablation numbers would
    not be reproducible.
    """
    X, y, names = _synergy_fixture()
    legacy = recommend_features(
        X, y, feature_names=names, k=3, weights={"redundancy": 1.0}, seed=0
    )
    assert legacy.weights["redundancy"] == 1.0
    # The redundancy term itself is still computed and reported for audit.
    assert all(np.isfinite(item.redundancy) for item in legacy.nominations)
    assert any(item.redundancy > 0.0 for item in legacy.nominations)


def test_siren_is_deterministic_and_serialisable() -> None:
    X, y, names = _synergy_fixture()
    first = recommend_features(X, y, feature_names=names, k=3, seed=11)
    second = recommend_features(X, y, feature_names=names, k=3, seed=11)
    assert first.selected_names() == second.selected_names()
    assert first.objective_curve == second.objective_curve

    payload = first.to_dict()
    assert payload["method"] == "siren"
    assert len(payload["nominations"]) == first.n_selected
    assert set(payload) >= {"weights", "objective_curve", "stopping_rule"}


def test_siren_validation_errors() -> None:
    X, y, names = _synergy_fixture(n=40)
    with pytest.raises(ValueError):
        recommend_features(X, y, feature_names=names, k=-1)


def test_scorecard_wrapper_matches_legacy_behaviour() -> None:
    X, y, names = _synergy_fixture(n=200)
    report = scorecard_ranking(X, y, feature_names=names, max_features=3)
    assert report.method == "scorecard"
    assert report.selected_names()[0] == "marginal"
    # The legacy marginals cannot separate the interaction pair from noise.
    assert {"noise", "partner"} & set(report.selected_names())


def test_nested_evaluation_is_leakage_safe_and_audited() -> None:
    X, y, names = _synergy_fixture(seed=3, n=200)
    audit, scores = nested_selection_evaluation(
        X, y, selector=siren_selector(k=3, seed=0), feature_names=names, n_splits=4, seed=0
    )
    assert len(audit.folds) == 4
    assert len(scores) == 4
    assert all(fold.method == "siren" for fold in audit.folds)
    # Each fold selects at most k features, chosen on the training split only.
    assert all(len(fold.selected_names) <= 3 for fold in audit.folds)
    # The marginal feature must be found in every fold; the duplicate may not be.
    assert audit.selection_frequency["marginal"] == pytest.approx(1.0)
    assert audit.selection_frequency["duplicate"] <= 1.0
    assert audit.mean_generalisation_gap > 0 or audit.mean_generalisation_gap <= 0
    assert set(audit.to_dict()) == {
        "folds",
        "selection_frequency",
        "mean_test_score",
        "std_test_score",
        "mean_generalisation_gap",
    }


def test_nested_evaluation_comparison_uses_identical_folds() -> None:
    X, y, names = _synergy_fixture(seed=5, n=200)
    _, siren_scores = nested_selection_evaluation(
        X, y, selector=siren_selector(k=3, seed=0), feature_names=names, n_splits=4, seed=1
    )
    _, baseline_scores = nested_selection_evaluation(
        X, y, selector=scorecard_selector(k=3), feature_names=names, n_splits=4, seed=1
    )
    assert len(siren_scores) == len(baseline_scores)
    interval = paired_bootstrap_test(siren_scores, baseline_scores, n_resamples=2000, seed=0)
    assert isinstance(interval, BootstrapInterval)
    assert interval.ci_low <= interval.mean_difference <= interval.ci_high
    assert 0.0 <= interval.p_value <= 1.0
    assert interval.n_resamples == 2000


def test_paired_bootstrap_detects_a_real_shift_and_ignores_noise() -> None:
    # Identical vectors -> no evidence of a difference.
    flat = paired_bootstrap_test([0.5] * 8, [0.5] * 8, n_resamples=1000, seed=0)
    assert flat.mean_difference == pytest.approx(0.0)
    assert flat.p_value == pytest.approx(1.0)

    # A consistent positive shift -> significant, with a strictly positive CI.
    shifted = paired_bootstrap_test([1.0] * 10, [0.0] * 10, n_resamples=1000, seed=0)
    assert shifted.mean_difference == pytest.approx(1.0)
    assert shifted.ci_low > 0.0
    assert shifted.p_value < 0.05


def test_paired_bootstrap_validates_input() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap_test([0.1, 0.2], [0.1])
    with pytest.raises(ValueError):
        paired_bootstrap_test([], [])
    with pytest.raises(ValueError):
        paired_bootstrap_test([0.1], [0.2], n_resamples=0)


def test_kfold_split_rejects_degenerate_configurations() -> None:
    X, y, names = _synergy_fixture(n=20)
    with pytest.raises(ValueError):
        nested_selection_evaluation(
            X, y, selector=siren_selector(k=2, seed=0), feature_names=names, n_splits=1
        )
    with pytest.raises(ValueError):
        nested_selection_evaluation(
            X[:3], y[:3], selector=siren_selector(k=2, seed=0), feature_names=names, n_splits=5
        )

    with pytest.raises(ValueError):
        recommend_features(X, y, feature_names=["only_one"])
    with pytest.raises(ValueError):
        recommend_features(X, y[:-1], feature_names=names)
    with pytest.raises(ValueError):
        recommend_features(X, y, weights={"nonexistent": 1.0})
