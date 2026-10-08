"""Composable building blocks for molecular property prediction."""

__version__ = "0.1.0"

from .core import ArchitectureRecipe, FeatureRole, FusionStrategy, RoleAssignment
from .diagnostics import FeatureDiagnostics, FeatureReport, diagnose_features
from .encoders import MLPEncoder, MLPResidual
from .experiment import PreparedExperiment, fit_experiment_prior, prepare_experiment
from .feature_analysis import (
    FeatureEvidence,
    FeatureEvidenceReport,
    ModelContribution,
    PriorCoefficientEvidence,
    analyze_feature_evidence,
    leave_group_out_importance,
    permutation_group_importance,
    summarize_prior_coefficients,
)
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
    "ArchitectureRecipe",
    "BaselineBlock",
    "CallablePhysicsPrior",
    "CompositePrior",
    "FeatureDiagnostics",
    "FeatureEvidence",
    "FeatureEvidenceReport",
    "FeatureMatrix",
    "FeatureRegistry",
    "FeatureReport",
    "FeatureRole",
    "FeatureSpec",
    "FusionStrategy",
    "IdentityOutputHead",
    "LinearAdditivePrior",
    "MLPEncoder",
    "MLPResidual",
    "ModelContribution",
    "MoleculeFeatureBlock",
    "NASA7OutputBlock",
    "PropertyTask",
    "PriorCoefficientEvidence",
    "PreparedExperiment",
    "ResidualPropertyModel",
    "RoleAssignment",
    "ScalarOutputHead",
    "VectorOutputHead",
    "ZeroPrior",
    "diagnose_features",
    "analyze_feature_evidence",
    "leave_group_out_importance",
    "molecular_feature_registry",
    "permutation_group_importance",
    "summarize_prior_coefficients",
    "fit_experiment_prior",
    "prepare_experiment",
]
