"""Composable building blocks for molecular property prediction."""

__version__ = "0.1.0"

from .diagnostics import FeatureDiagnostics, FeatureReport, diagnose_features
from .encoders import MLPEncoder, MLPResidual
from .experiment import PreparedExperiment, fit_experiment_prior, prepare_experiment
from .features import (
    FeatureMatrix,
    FeatureRegistry,
    FeatureSpec,
    MoleculeFeatureBlock,
    molecular_feature_registry,
)
from .heads import IdentityOutputHead, NASA7OutputBlock, ScalarOutputHead, VectorOutputHead
from .models import ResidualPropertyModel
from .priors import (
    BaselineBlock,
    CallablePhysicsPrior,
    CompositePrior,
    LinearAdditivePrior,
    ZeroPrior,
)
from .tasks import PropertyTask

__all__ = [
    "BaselineBlock",
    "CallablePhysicsPrior",
    "CompositePrior",
    "FeatureDiagnostics",
    "FeatureMatrix",
    "FeatureRegistry",
    "FeatureReport",
    "FeatureSpec",
    "IdentityOutputHead",
    "LinearAdditivePrior",
    "MLPEncoder",
    "MLPResidual",
    "MoleculeFeatureBlock",
    "NASA7OutputBlock",
    "PropertyTask",
    "PreparedExperiment",
    "ResidualPropertyModel",
    "ScalarOutputHead",
    "VectorOutputHead",
    "ZeroPrior",
    "diagnose_features",
    "molecular_feature_registry",
    "fit_experiment_prior",
    "prepare_experiment",
]
