from __future__ import annotations
from typing import Any
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from ..data.graph import BatchMolGraph, collate_mol_graphs
from ..encoders.gnn import DMPNNEncoder, EGNNEncoder, GINEncoder, ProteinEncoder
from ..encoders.mlp import MLPEncoder
from ..heads.basic import IdentityOutputHead, ScalarOutputHead, VectorOutputHead
from ..heads.specialized import EvidentialRegressionHead, GaussianNLLHead
from ..priors.base import BaselineBlock
from ..config.model_spec import ModelSpec

class HybridChemistryModel(nn.Module):
    def __init__(self, structure_encoder: nn.Module | None = None, descriptor_encoder: nn.Module | None = None, protein_encoder: nn.Module | None = None, prior: BaselineBlock | None = None, ffn: nn.Module | None = None, output_head: nn.Module | None = None, fusion_strategy: str = "concat", mixture_pooling: str = "weighted_sum", reaction_mode: str = "diff_concat",) -> None:
        super().__init__()
        self.structure_encoder = structure_encoder
        self.descriptor_encoder = descriptor_encoder
        self.protein_encoder = protein_encoder
        self.prior = prior
        self.ffn = ffn or nn.Identity()
        self.output_head = output_head or IdentityOutputHead()
        self.fusion_strategy = fusion_strategy
        self.mixture_pooling = mixture_pooling
        self.reaction_mode = reaction_mode

    def encode_structure(self, structure_input: Any) -> Tensor | None:
        if self.structure_encoder is None:
            return None
        if isinstance(structure_input, BatchMolGraph):
            return self.structure_encoder(structure_input)
        if isinstance(structure_input, dict):

            if "reactants" in structure_input and "products" in structure_input:
                h_react = self.structure_encoder(structure_input["reactants"])
                h_prod = self.structure_encoder(structure_input["products"])
                diff = h_prod - h_react
                if self.reaction_mode == "diff":
                    return diff
                if self.reaction_mode == "concat":
                    return torch.cat([h_react, h_prod], dim=-1)
                return torch.cat([diff, h_prod], dim=-1)

            if "mixture_graphs" in structure_input and "fractions" in structure_input:
                graphs_list = structure_input["mixture_graphs"]
                fractions = structure_input["fractions"]
                h_components = [self.structure_encoder(bg) for bg in graphs_list]
                stacked = torch.stack(h_components, dim=1)
                weights = fractions.unsqueeze(-1).to(stacked.device, stacked.dtype)
                return (stacked * weights).sum(dim=1)

            if "ligand" in structure_input:
                h_ligand = self.structure_encoder(structure_input["ligand"])
                if self.protein_encoder is not None and "protein_tokens" in structure_input:
                    h_prot = self.protein_encoder(structure_input["protein_tokens"])
                    return torch.cat([h_ligand, h_prot], dim=-1)
                return h_ligand
        if isinstance(structure_input, (list, tuple)):
            bg = collate_mol_graphs(structure_input)
            return self.structure_encoder(bg)
        raise TypeError(f"unsupported structure_input format: {type(structure_input).__name__}")

    def forward(self, inputs: Any = None, descriptor_features: Tensor | None = None, prior_features: Tensor | None = None, **kwargs: Any,) -> dict[str, Tensor]:
        structure_input = kwargs.get("batch_graph") or kwargs.get("graph") or kwargs.get("structure")
        desc_input = descriptor_features or kwargs.get("descriptors")
        pri_input = prior_features or kwargs.get("prior") or kwargs.get("prior_features")
        if isinstance(inputs, dict):
            if structure_input is None:
                structure_input = inputs.get("graph") or inputs.get("structure")
            if desc_input is None:
                desc_input = inputs.get("descriptors")
            if pri_input is None:
                pri_input = inputs.get("prior") or inputs.get("prior_features")
        elif isinstance(inputs, BatchMolGraph) or (isinstance(inputs, (list, tuple)) and inputs and isinstance(inputs[0], str)):
            if structure_input is None:
                structure_input = inputs
        elif isinstance(inputs, Tensor):

            if desc_input is None:
                desc_input = inputs

        h_struct = self.encode_structure(structure_input) if structure_input is not None else None

        h_desc = None
        if self.descriptor_encoder is not None and desc_input is not None:
            h_desc = self.descriptor_encoder(desc_input)

        if h_struct is not None and h_desc is not None:
            if self.fusion_strategy == "concat":
                h_fused = torch.cat([h_struct, h_desc], dim=-1)
            elif self.fusion_strategy == "sum":
                h_fused = h_struct + h_desc
            else:
                h_fused = torch.cat([h_struct, h_desc], dim=-1)
        elif h_struct is not None:
            h_fused = h_struct
        elif h_desc is not None:
            h_fused = h_desc
        else:
            raise RuntimeError("HybridChemistryModel requires at least one of structure or descriptors")

        latent = self.ffn(h_fused)

        head_out = self.output_head(latent)
        if isinstance(head_out, Tensor):
            pred = head_out
            aux_dict = {}
        elif isinstance(head_out, dict):
            pred = head_out.get("prediction", head_out.get("mean"))
            aux_dict = {k: v for k, v in head_out.items() if k != "prediction"}
        else:
            raise TypeError("output_head must return a Tensor or a mapping of Tensors")

        prior_out = None
        if self.prior is not None:
            if pri_input is None and desc_input is not None:
                pri_input = desc_input
            if pri_input is not None:
                prior_out = self.prior(pri_input)

                if prior_out.shape != pred.shape:
                    if prior_out.numel() == pred.numel():
                        prior_out = prior_out.reshape_as(pred)
                pred = pred + prior_out
        result = {
            "prediction": pred,
            "latent": latent,
            **aux_dict,
        }
        if prior_out is not None:
            result["prior"] = prior_out
        return result

def build_hybrid_model(model_spec: Any, task_kind: str = "scalar", out_dim: int = 1, descriptor_dim: int = 0,prior: BaselineBlock | None = None,) -> HybridChemistryModel:
    if not isinstance(model_spec, ModelSpec):
        model_spec = ModelSpec.parse(model_spec)

    struct_enc: nn.Module | None = None
    struct_out_dim = 0
    if model_spec.structure.enabled:
        enc_name = model_spec.structure.encoder
        h_dim = model_spec.structure.hidden_dim
        depth = model_spec.structure.depth
        dropout = model_spec.structure.dropout
        agg = model_spec.structure.aggregation
        vn = model_spec.structure.virtual_node
        if enc_name == "dmpnn":
            struct_enc = DMPNNEncoder(hidden_dim=h_dim, depth=depth, dropout=dropout, aggregation=agg, use_virtual_node=vn)
            struct_out_dim = h_dim
        elif enc_name == "gin":
            struct_enc = GINEncoder(hidden_dim=h_dim, depth=depth, dropout=dropout, aggregation=agg, use_virtual_node=vn)
            struct_out_dim = h_dim
        elif enc_name == "egnn":
            struct_enc = EGNNEncoder(hidden_dim=h_dim, depth=depth, cutoff=model_spec.structure.cutoff, aggregation=agg)
            struct_out_dim = h_dim
        else:
            raise ValueError(f"unknown structure encoder {enc_name!r}")

    desc_enc: nn.Module | None = None
    desc_out_dim = 0
    desc_enabled = model_spec.descriptors.enabled
    if desc_enabled is True or (desc_enabled == "auto" and descriptor_dim > 0):
        if descriptor_dim > 0:
            d_hdim = model_spec.descriptors.hidden_dim
            d_depth = model_spec.descriptors.depth
            d_drop = model_spec.descriptors.dropout
            desc_enc = MLPEncoder(
                input_dim=descriptor_dim,
                representation_dim=d_hdim,
                hidden_dim=d_hdim,
                depth=d_depth,
                dropout=d_drop,
            )
            desc_out_dim = d_hdim

    prot_enc: nn.Module | None = None
    prot_out_dim = 0
    if model_spec.protein.embedding_dim > 0:
        prot_enc = ProteinEncoder(
            embedding_dim=model_spec.protein.embedding_dim,
            channels=model_spec.protein.channels,
            kernel_size=model_spec.protein.kernel_size,
            max_length=model_spec.protein.max_length,
        )
        prot_out_dim = model_spec.protein.channels

    if struct_out_dim > 0 and desc_out_dim > 0:
        combined_dim = struct_out_dim + desc_out_dim if model_spec.fusion == "concat" else struct_out_dim
    elif struct_out_dim > 0:
        combined_dim = struct_out_dim
    elif desc_out_dim > 0:
        combined_dim = desc_out_dim
    else:

        combined_dim = 128
        desc_enc = MLPEncoder(input_dim=max(1, descriptor_dim), representation_dim=combined_dim)

    ffn_dim = model_spec.ffn.hidden_dim
    ffn_depth = model_spec.ffn.depth
    ffn_drop = model_spec.ffn.dropout
    ffn_layers: list[nn.Module] = []
    in_d = combined_dim
    for _ in range(ffn_depth - 1):
        ffn_layers.extend([nn.Linear(in_d, ffn_dim), nn.ReLU(), nn.Dropout(ffn_drop)])
        in_d = ffn_dim
    ffn_layers.append(nn.Linear(in_d, ffn_dim))
    ffn_layers.append(nn.ReLU())
    ffn = nn.Sequential(*ffn_layers)

    head_type = model_spec.head
    if head_type == "auto":
        head_type = "classification" if "classification" in task_kind else "regression"

    if head_type == "gaussian":
        head = GaussianNLLHead(in_features=ffn_dim, out_targets=out_dim)
    elif head_type == "evidential":
        head = EvidentialRegressionHead(in_features=ffn_dim, out_targets=out_dim)
    elif out_dim > 1:
        head = nn.Sequential(nn.Linear(ffn_dim, out_dim), VectorOutputHead(out_dim))
    else:
        head = nn.Sequential(nn.Linear(ffn_dim, 1), ScalarOutputHead())
    return HybridChemistryModel(
        structure_encoder=struct_enc,
        descriptor_encoder=desc_enc,
        protein_encoder=prot_enc,
        prior=prior,
        ffn=ffn,
        output_head=head,
        fusion_strategy=model_spec.fusion,
        mixture_pooling=model_spec.mixture_pooling,
        reaction_mode=model_spec.reaction_mode,
    )

__all__ = [
    "HybridChemistryModel",
    "build_hybrid_model",
]
