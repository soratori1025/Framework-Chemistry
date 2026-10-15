from __future__ import annotations
import math
from dataclasses import asdict, dataclass, field
from typing import Any

STRUCTURE_ENCODERS = {"dmpnn", "gin", "egnn", "none"}
AGGREGATIONS = {"mean", "sum", "max", "norm", "attention"}
FUSIONS = {"concat", "gated", "sum"}
HEADS = {"auto", "regression", "gaussian", "gaussian_nll", "evidential", "classification", "scalar", "vector"}
PRIOR_MODES = {"auto", "none", "frozen", "trainable"}
MIXTURE_POOLING = {"weighted_sum", "weighted_sum_max"}
REACTION_MODES = {"diff", "concat", "diff_concat"}
PROTEIN_INTERACTIONS = {"concat", "interaction"}
_LEGACY_KEYS = {"composition", "descriptor_encoder", "hidden_dim", "depth", "dropout"}
__all__ = [
    "DescriptorSpec",
    "FFNSpec",
    "ModelSpec",
    "ProteinSpec",
    "StructureSpec",
]

def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)

def _only(data: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown {name} field(s): {', '.join(sorted(unknown))}")

def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value

def _dropout(value: Any, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 0.0 <= value < 1.0
    ):
        raise ValueError(f"{name} must be a number in [0, 1)")
    return float(value)

def _choice(value: Any, allowed: set[str], name: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(sorted(allowed))}")
    return value

@dataclass(frozen=True)
class StructureSpec:
    encoder: str = "none"
    hidden_dim: int = 300
    depth: int = 3
    dropout: float = 0.0
    aggregation: str = "mean"
    cutoff: float = 5.0
    conformer_seed: int = 0
    virtual_node: bool = False

    @classmethod
    def parse(cls, value: Any, defaults: dict[str, Any]) -> StructureSpec:
        if isinstance(value, str):
            value = {"encoder": value}
        data = {**defaults, **_mapping(value, "model.structure")}
        if "num_layers" in data and "depth" not in data:
            data["depth"] = data.pop("num_layers")
        _only(
            data,
            {"encoder", "hidden_dim", "depth", "num_layers", "dropout", "aggregation", "cutoff", "conformer_seed", "virtual_node"},
            "model.structure",
        )
        spec = cls(**data)
        _choice(spec.encoder, STRUCTURE_ENCODERS, "model.structure.encoder")
        _positive_int(spec.hidden_dim, "model.structure.hidden_dim")
        _positive_int(spec.depth, "model.structure.depth")
        _dropout(spec.dropout, "model.structure.dropout")
        _choice(spec.aggregation, AGGREGATIONS, "model.structure.aggregation")
        if not isinstance(spec.virtual_node, bool):
            raise TypeError("model.structure.virtual_node must be a boolean")
        if (
            not isinstance(spec.cutoff, (int, float))
            or isinstance(spec.cutoff, bool)
            or not math.isfinite(spec.cutoff)
            or spec.cutoff <= 0
        ):
            raise ValueError("model.structure.cutoff must be a positive number")
        if not isinstance(spec.conformer_seed, int) or isinstance(spec.conformer_seed, bool):
            raise TypeError("model.structure.conformer_seed must be an integer")
        return spec

    @property
    def enabled(self) -> bool:
        return self.encoder != "none"

    @property
    def needs_coordinates(self) -> bool:
        return self.encoder == "egnn"

@dataclass(frozen=True)
class DescriptorSpec:
    enabled: str | bool = "auto"
    hidden_dim: int = 128
    depth: int = 1
    dropout: float = 0.0
    standardize: bool = True

    @classmethod
    def parse(cls, value: Any, defaults: dict[str, Any]) -> DescriptorSpec:
        if isinstance(value, bool):
            value = {"enabled": value}
        data = {**defaults, **_mapping(value, "model.descriptors")}
        if "num_layers" in data and "depth" not in data:
            data["depth"] = data.pop("num_layers")
        data.pop("in_dim", None)
        _only(data, {"enabled", "hidden_dim", "depth", "num_layers", "in_dim", "dropout", "standardize"}, "model.descriptors")
        spec = cls(**data)
        if spec.enabled != "auto" and not isinstance(spec.enabled, bool):
            raise ValueError("model.descriptors.enabled must be true, false, or 'auto'")
        _positive_int(spec.hidden_dim, "model.descriptors.hidden_dim")
        _positive_int(spec.depth, "model.descriptors.depth")
        _dropout(spec.dropout, "model.descriptors.dropout")
        if not isinstance(spec.standardize, bool):
            raise TypeError("model.descriptors.standardize must be a boolean")
        return spec

@dataclass(frozen=True)
class FFNSpec:
    hidden_dim: int = 300
    depth: int = 2
    dropout: float = 0.0

    @classmethod
    def parse(cls, value: Any) -> FFNSpec:
        data = _mapping(value, "model.ffn")
        if "num_layers" in data and "depth" not in data:
            data["depth"] = data.pop("num_layers")
        _only(data, {"hidden_dim", "depth", "num_layers", "dropout"}, "model.ffn")
        spec = cls(**data)
        _positive_int(spec.hidden_dim, "model.ffn.hidden_dim")
        _positive_int(spec.depth, "model.ffn.depth")
        _dropout(spec.dropout, "model.ffn.dropout")
        return spec

@dataclass(frozen=True)
class ProteinSpec:
    embedding_dim: int = 64
    channels: int = 128
    kernel_size: int = 7
    max_length: int = 1000
    interaction: str = "concat"

    @classmethod
    def parse(cls, value: Any) -> ProteinSpec:
        data = _mapping(value, "model.protein")
        _only(
            data,
            {"embedding_dim", "channels", "kernel_size", "max_length", "interaction"},
            "model.protein",
        )
        spec = cls(**data)
        for name in ("embedding_dim", "channels", "kernel_size", "max_length"):
            _positive_int(getattr(spec, name), f"model.protein.{name}")
        _choice(spec.interaction, PROTEIN_INTERACTIONS, "model.protein.interaction")
        return spec

@dataclass(frozen=True)
class ModelSpec:
    structure: StructureSpec = field(default_factory=StructureSpec)
    descriptors: DescriptorSpec = field(default_factory=DescriptorSpec)
    ffn: FFNSpec = field(default_factory=FFNSpec)
    fusion: str = "concat"
    head: str = "auto"
    prior: str = "auto"
    target_scaling: bool = True
    mixture_pooling: str = "weighted_sum"
    reaction_mode: str = "diff_concat"
    protein: ProteinSpec = field(default_factory=ProteinSpec)

    def __post_init__(self) -> None:
        if isinstance(self.structure, dict):
            object.__setattr__(self, "structure", StructureSpec.parse(self.structure, {}))
        if isinstance(self.descriptors, dict):
            object.__setattr__(self, "descriptors", DescriptorSpec.parse(self.descriptors, {}))
        if isinstance(self.ffn, dict):
            object.__setattr__(self, "ffn", FFNSpec.parse(self.ffn))
        if isinstance(self.protein, dict):
            object.__setattr__(self, "protein", ProteinSpec.parse(self.protein))
        if isinstance(self.head, dict):
            h_val = self.head.get("kind", "auto")
            object.__setattr__(self, "head", h_val)
        if self.head == "gaussian_nll":
            object.__setattr__(self, "head", "gaussian")
        elif self.head in ("scalar", "vector"):
            object.__setattr__(self, "head", "regression")

    @classmethod
    def parse(cls, value: Any, graph: dict[str, Any] | None = None) -> ModelSpec:
        data = _mapping(value, "model")
        graph = graph or {}
        allowed = {
            "structure", "descriptors", "ffn", "fusion", "head", "prior", "target_scaling",
            "mixture_pooling", "reaction_mode", "protein",
        } | _LEGACY_KEYS
        _only(data, allowed, "model")
        descriptor_defaults: dict[str, Any] = {}
        for legacy in ("hidden_dim", "depth", "dropout"):
            if legacy in data:
                descriptor_defaults[legacy] = data[legacy]
        structure_defaults: dict[str, Any] = {}
        graph_encoder = graph.get("encoder")
        if isinstance(graph_encoder, str) and graph_encoder in STRUCTURE_ENCODERS:
            structure_defaults["encoder"] = graph_encoder
        for key in ("hidden_dim", "depth"):
            if key in graph:
                structure_defaults[key] = graph[key]
        if "aggregation" in graph and graph.get("encoder") is not None:
            structure_defaults["aggregation"] = graph["aggregation"]
        target_scaling = data.get("target_scaling", True)
        if not isinstance(target_scaling, bool):
            raise TypeError("model.target_scaling must be a boolean")
        return cls(
            structure=StructureSpec.parse(data.get("structure"), structure_defaults),
            descriptors=DescriptorSpec.parse(data.get("descriptors"), descriptor_defaults),
            ffn=FFNSpec.parse(data.get("ffn")),
            fusion=_choice(data.get("fusion", "concat"), FUSIONS, "model.fusion"),
            head=_choice(data.get("head", "auto"), HEADS, "model.head"),
            prior=_choice(data.get("prior", "auto"), PRIOR_MODES, "model.prior"),
            target_scaling=target_scaling,
            mixture_pooling=_choice(
                data.get("mixture_pooling", "weighted_sum"), MIXTURE_POOLING, "model.mixture_pooling"
            ),
            reaction_mode=_choice(
                data.get("reaction_mode", "diff_concat"), REACTION_MODES, "model.reaction_mode"
            ),
            protein=ProteinSpec.parse(data.get("protein")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
