"""Prepare datasets, feature matrices, splits, caches, and analysis from config."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .analysis import AnalysisReport, analyze_features_and_targets
from .config import ExperimentConfig
from .data import MolecularDataset, load_configured_dataset
from .features import (
    FeatureCache,
    FeatureMatrix,
    FeatureRegistry,
    MoleculeFeatureBlock,
    build_feature_registry,
    configured_feature_fingerprint,
)
from .priors import PriorFit, fit_linear_prior
from .splits import DatasetSplit, split_indices


@dataclass(frozen=True)
class PreparedExperiment:
    config: ExperimentConfig
    dataset: MolecularDataset
    split: DatasetSplit
    features: FeatureMatrix
    feature_blocks: dict[str, tuple[str, ...]]
    prior_features: dict[str, tuple[str, ...]]
    route_features: dict[str, tuple[str, ...]]
    analysis: AnalysisReport | None
    feature_registry: FeatureRegistry
    feature_fingerprint: str


def _resolve_blocks(
    selected_blocks: tuple[str, ...],
    resolved_names: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    names: list[str] = []
    for block_name in selected_blocks:
        names.extend(resolved_names[block_name])
    return tuple(names)


def prepare_experiment(config: ExperimentConfig) -> PreparedExperiment:
    """Load the user dataset and compute reproducible splits/features/reports."""
    dataset = load_configured_dataset(config)
    split = split_indices(dataset.smiles, config.split)
    registry, feature_blocks = build_feature_registry(
        config.features,
        dataset.smiles,
        split.train,
    )
    all_names = tuple(
        feature_name
        for block_names in feature_blocks.values()
        for feature_name in block_names
    )
    if not all_names:
        raise ValueError("feature configuration resolved to zero columns")
    feature_block = MoleculeFeatureBlock(registry, all_names)

    import hashlib
    import json

    schema = {
        "definitions_fingerprint": configured_feature_fingerprint(config.features),
        "resolved_names": all_names,
    }
    config_fingerprint = hashlib.sha256(
        json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if config.cache.enabled:
        cache_dir = config.resolve_path(config.cache.directory)
        matrix = FeatureCache(cache_dir, config_fingerprint).transform(
            feature_block, dataset.smiles
        )
    else:
        matrix = feature_block.transform(dataset.smiles)

    prior_features = {
        property_name: _resolve_blocks(prior.use, feature_blocks)
        for property_name, prior in config.priors.items()
    }
    route_features = {
        property_name: _resolve_blocks(selected, feature_blocks)
        for property_name, selected in config.routes.items()
    }
    report = None
    if config.analysis.enabled:
        report = analyze_features_and_targets(
            matrix,
            split,
            dataset.targets,
            bins=config.analysis.bins,
            support_threshold=config.analysis.support_threshold,
            correlation_threshold=config.analysis.correlation_threshold,
        )
        output_dir = config.resolve_path(config.analysis.output_dir)
        report.write_json(output_dir / f"{config.name}-analysis.json")
    return PreparedExperiment(
        config=config,
        dataset=dataset,
        split=split,
        features=matrix,
        feature_blocks=feature_blocks,
        prior_features=prior_features,
        route_features=route_features,
        analysis=report,
        feature_registry=registry,
        feature_fingerprint=config_fingerprint,
    )


def fit_experiment_prior(
    experiment: PreparedExperiment,
    property_name: str,
    *,
    target_name: str | None = None,
    ridge_alpha: float = 1e-3,
) -> PriorFit:
    """Fit and freeze the configured prior using training rows only."""
    if property_name not in experiment.config.priors:
        raise KeyError(f"no prior configured for property {property_name!r}")
    if target_name is None:
        task = experiment.config.tasks.get(property_name)
        if task is None:
            raise ValueError(
                f"target_name is required because task {property_name!r} is not configured"
            )
        target_name = task["target"]
    if target_name not in experiment.dataset.targets:
        raise KeyError(f"unknown dataset target {target_name!r}")
    target = experiment.dataset.targets[target_name]
    y = np.asarray(target, dtype=float)
    if y.ndim not in {1, 2}:
        raise ValueError("linear prior fitting requires a scalar or 2-D target")
    names = experiment.prior_features.get(property_name, ())
    if not names:
        raise ValueError(f"prior {property_name!r} has no resolved features")
    columns = [experiment.features.names.index(name) for name in names]
    x_train = experiment.features.values[experiment.split.train][:, columns]
    y_train = y[experiment.split.train]
    config = experiment.config.priors[property_name]
    fixed: dict[str, float] = {}
    for feature_name, coefficient in config.fixed.items():
        if feature_name in names:
            fixed[feature_name] = coefficient
            continue
        expanded = experiment.feature_blocks.get(feature_name, ())
        if len(expanded) == 1 and expanded[0] in names:
            fixed[expanded[0]] = coefficient
        else:
            raise ValueError(
                f"fixed coefficient key {feature_name!r} must be a resolved prior feature; "
                "a multi-column feature block cannot use one scalar fixed coefficient"
            )
    return fit_linear_prior(
        x_train,
        y_train,
        names,
        fixed=fixed,
        method=config.fit,
        ridge_alpha=ridge_alpha,
    )
