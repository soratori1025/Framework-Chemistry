"""Model builder from ArchitectureConfig."""
from __future__ import annotations
from typing import Any
import torch
from torch import nn, Tensor
from ..config.schema import ArchitectureConfig, ComponentConfig
from ..priors import BaselineBlock
from .residual import ResidualPropertyModel
from ..heads import IdentityOutputHead

class GatedFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim)
        
    def forward(self, prior: Tensor, residual: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate(torch.cat([prior, residual], dim=-1)))
        return g * prior + (1 - g) * residual

class WeightedFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim) * 0.5)
        
    def forward(self, prior: Tensor, residual: Tensor) -> Tensor:
        a = torch.sigmoid(self.alpha)
        return a * prior + (1 - a) * residual

class ConcatFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim * 2, dim)
        
    def forward(self, prior: Tensor, residual: Tensor) -> Tensor:
        return self.proj(torch.cat([prior, residual], dim=-1))

class AdditiveFusion(nn.Module):
    def forward(self, prior: Tensor, residual: Tensor) -> Tensor:
        return prior + residual

class RolePredictor(nn.Module):
    """Composes prior and residual according to the configured fusion strategy."""
    def __init__(self, prior_module: nn.Module | None, residual_module: nn.Module | None, fusion_module: nn.Module, output_head: nn.Module | None = None):
        super().__init__()
        self.prior = prior_module
        self.residual = residual_module
        self.fusion = fusion_module
        self.output_head = output_head or IdentityOutputHead()
        
    def forward(self, prior_features: Tensor | None = None, residual_features: Tensor | None = None) -> dict[str, Tensor]:
        if self.prior is not None and prior_features is None:
            raise ValueError("prior_features required")
        if self.residual is not None and residual_features is None:
            raise ValueError("residual_features required")
            
        p_out = self.prior(prior_features) if self.prior is not None else None
        r_out = self.residual(residual_features) if self.residual is not None else None
        
        if p_out is not None and r_out is not None:
            fused = self.fusion(p_out, r_out)
        elif p_out is not None:
            fused = p_out
        elif r_out is not None:
            fused = r_out
        else:
            raise RuntimeError("RolePredictor requires at least one of prior or residual")
            
        prediction = self.output_head(fused)
        
        out = {"prediction": prediction}
        if p_out is not None:
            out["baseline"] = p_out
        if r_out is not None:
            out["residual"] = r_out
            
        return out

def build_component(config: ComponentConfig, in_features: int, out_features: int = 1) -> nn.Module:
    """Factory for standard components."""
    if config.model == "zero":
        class ZeroModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.zeros(x.size(0), out_features, device=x.device, dtype=x.dtype)
        return ZeroModel()
        
    if config.model in ("linear", "ridge"):
        return nn.Linear(in_features, out_features)
        
    if config.model == "mlp":
        hidden_dim = config.params.get("hidden_dim", 64)
        depth = config.params.get("depth", 2)
        layers = []
        in_dim = in_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_features))
        return nn.Sequential(*layers)
        
    raise ValueError(f"unsupported component model {config.model!r}")

def build_role_predictor(config: ArchitectureConfig, prior_in_features: int = 0, residual_in_features: int = 0, out_features: int = 1) -> RolePredictor:
    """Build a complete runnable model from ArchitectureConfig."""
    prior_mod = build_component(config.prior, prior_in_features, out_features) if config.prior else None
    residual_mod = build_component(config.residual, residual_in_features, out_features) if config.residual else None
    
    fusion_map = {
        "additive": AdditiveFusion,
        "weighted": WeightedFusion,
        "gated": GatedFusion,
        "concatenate": ConcatFusion,
    }
    
    if config.fusion not in fusion_map:
        raise ValueError(f"unknown fusion strategy {config.fusion!r}")
        
    fusion_mod = fusion_map[config.fusion](out_features) if (config.fusion != "additive") else AdditiveFusion()
    
    return RolePredictor(prior_mod, residual_mod, fusion_mod)
