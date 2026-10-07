"""Compare external prediction molecules with training feature distributions."""
from __future__ import annotations
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any
from ..diagnostics import FeatureReport, diagnose_features
from ..features import FeatureCache, MoleculeFeatureBlock
if TYPE_CHECKING:
    from ..experiment import PreparedExperiment

def analyze_prediction_molecules(experiment: PreparedExperiment, molecules: Sequence[str | Any],) -> FeatureReport:
    """Report train-range extrapolation for molecules supplied at inference time."""
    if not molecules:
        raise ValueError("at least one prediction molecule is required")
    block = MoleculeFeatureBlock(experiment.feature_registry, experiment.features.names)
    if experiment.config.cache.enabled:
        matrix = FeatureCache(
            experiment.config.resolve_path(experiment.config.cache.directory),
            experiment.feature_fingerprint,
        ).transform(block, tuple(molecules))
    else:
        matrix = block.transform(molecules)
    return diagnose_features(
        experiment.features.values[experiment.split.train],
        matrix.values,
        matrix.names,
        correlation_threshold=experiment.config.analysis.correlation_threshold,
    )
