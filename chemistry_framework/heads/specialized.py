from __future__ import annotations
from typing import Any
import torch
from torch import Tensor, nn
from torch.nn import functional as F

class GaussianNLLHead(nn.Module):
    def __init__(self, in_features: int, out_targets: int = 1) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_targets = out_targets
        self.mean_proj = nn.Linear(in_features, out_targets)
        self.logvar_proj = nn.Linear(in_features, out_targets)

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        mean = self.mean_proj(features)

        logvar = self.logvar_proj(features).clamp(-12.0, 12.0)
        var = torch.exp(logvar)
        std = torch.sqrt(var)
        return {
            "prediction": mean,
            "mean": mean,
            "variance": var,
            "std": std,
            "logvar": logvar,
        }

class EvidentialRegressionHead(nn.Module):
    def __init__(self, in_features: int, out_targets: int = 1, eps: float = 1e-6) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_targets = out_targets
        self.eps = eps
        self.proj = nn.Linear(in_features, out_targets * 4)

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        raw = self.proj(features)
        if self.out_targets == 1:
            gamma = raw[..., 0:1]
            nu = F.softplus(raw[..., 1:2]) + self.eps
            alpha = F.softplus(raw[..., 2:3]) + 1.0 + self.eps
            beta = F.softplus(raw[..., 3:4]) + self.eps
        else:
            reshaped = raw.view(*raw.shape[:-1], self.out_targets, 4)
            gamma = reshaped[..., 0]
            nu = F.softplus(reshaped[..., 1]) + self.eps
            alpha = F.softplus(reshaped[..., 2]) + 1.0 + self.eps
            beta = F.softplus(reshaped[..., 3]) + self.eps
        var = beta / (alpha - 1.0)
        epistemic = beta / (nu * (alpha - 1.0))
        total_var = var + epistemic
        return {
            "prediction": gamma,
            "gamma": gamma,
            "nu": nu,
            "alpha": alpha,
            "beta": beta,
            "aleatoric_variance": var,
            "epistemic_variance": epistemic,
            "variance": total_var,
            "std": torch.sqrt(total_var),
        }

class MultiTaskHead(nn.Module):
    def __init__(self, in_features: int | None = None, task_specs: dict[str, tuple[str, int] | int] | None = None, *, in_dim: int | None = None, task_dims: dict[str, int] | None = None,) -> None:
        super().__init__()
        in_features = in_features if in_features is not None else in_dim
        if in_features is None:
            raise ValueError("in_features or in_dim must be provided")
        self.in_features = in_features
        raw_specs = task_specs or task_dims or {}
        self.task_specs: dict[str, tuple[str, int]] = {}
        self.projections = nn.ModuleDict()
        for name, spec in raw_specs.items():
            if isinstance(spec, int):
                kind, dim = "scalar" if spec == 1 else "vector", spec
            else:
                kind, dim = spec
            self.task_specs[name] = (kind, dim)
            self.projections[name] = nn.Linear(in_features, dim)

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        outputs: dict[str, Tensor] = {}
        for name, proj in self.projections.items():
            outputs[name] = proj(features)
        return outputs

class NodeLevelHead(nn.Module):
    def __init__(self, in_features: int, out_dim: int = 1) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, out_dim),
        )

    def forward(self, atom_features: Tensor) -> Tensor:
        return self.proj(atom_features)

__all__ = [
    "EvidentialRegressionHead",
    "GaussianNLLHead",
    "MultiTaskHead",
    "NodeLevelHead",
]
