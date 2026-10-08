"""Public recipe objects for role-based chemistry model construction."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class FeatureRole(str, Enum):
    """How a feature block participates in a predictive model."""

    PRIOR = "prior"
    RESIDUAL = "residual"
    AUXILIARY = "auxiliary"
    DIRECT = "direct"


class FusionStrategy(str, Enum):
    """How predictions from multiple low-level branches are combined."""

    ADDITIVE = "additive"
    WEIGHTED = "weighted"
    CONCATENATE = "concatenate"
    GATED = "gated"


@dataclass(frozen=True)
class RoleAssignment:
    """Describe how a feature family is assigned to a model role."""

    name: str
    role: FeatureRole | str
    features: tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.role, str):
            object.__setattr__(self, "role", FeatureRole(self.role.lower()))


@dataclass(frozen=True)
class ArchitectureRecipe:
    """High-level recipe for building a domain-specific chemistry model.

    The recipe keeps the public library explicit about the role of each feature
    family and how separate prior/residual components are fused. This lets users
    swap feature groups or model backbones without rewriting the baseline model
    skeleton.
    """

    name: str
    prior_features: tuple[str, ...] = ()
    residual_features: tuple[str, ...] = ()
    auxiliary_features: tuple[str, ...] = ()
    fusion_strategy: FusionStrategy | str = FusionStrategy.ADDITIVE
    prior_model: str = "linear"
    residual_model: str = "mlp"
    description: str = ""
    assignments: tuple[RoleAssignment, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if isinstance(self.fusion_strategy, str):
            object.__setattr__(
                self,
                "fusion_strategy",
                FusionStrategy(self.fusion_strategy.lower()),
            )

    def with_assignment(
        self,
        name: str,
        role: FeatureRole | str,
        features: Iterable[str] | str | None = None,
        description: str = "",
    ) -> "ArchitectureRecipe":
        """Return a new recipe with a role assignment appended."""
        normalized_features: tuple[str, ...]
        if features is None:
            normalized_features = ()
        elif isinstance(features, str):
            normalized_features = (features,)
        else:
            normalized_features = tuple(features)
        assignment = RoleAssignment(name=name, role=role, features=normalized_features, description=description)
        updated_assignments = self.assignments + (assignment,)
        updated_prior = self.prior_features
        updated_residual = self.residual_features
        updated_aux = self.auxiliary_features
        if assignment.role == FeatureRole.PRIOR:
            updated_prior = tuple(dict.fromkeys(self.prior_features + assignment.features))
        elif assignment.role == FeatureRole.RESIDUAL:
            updated_residual = tuple(dict.fromkeys(self.residual_features + assignment.features))
        elif assignment.role in {FeatureRole.AUXILIARY, FeatureRole.DIRECT}:
            updated_aux = tuple(dict.fromkeys(self.auxiliary_features + assignment.features))
        return ArchitectureRecipe(
            name=self.name,
            prior_features=updated_prior,
            residual_features=updated_residual,
            auxiliary_features=updated_aux,
            fusion_strategy=self.fusion_strategy,
            prior_model=self.prior_model,
            residual_model=self.residual_model,
            description=self.description,
            assignments=updated_assignments,
        )


__all__ = [
    "ArchitectureRecipe",
    "FeatureRole",
    "FusionStrategy",
    "RoleAssignment",
]
