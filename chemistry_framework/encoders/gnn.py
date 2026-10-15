from __future__ import annotations
import math
from typing import Any
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from ..data.graph import ATOM_FEATURE_DIM, BOND_FEATURE_DIM, BatchMolGraph
import io
import matplotlib.pyplot as plt

class GraphAggregation(nn.Module):
    """Pooling layer mapping atom/node representations to a graph representation."""

    def __init__(self, mode: str = "mean", hidden_dim: int = 300) -> None:
        super().__init__()
        if mode not in {"mean", "sum", "max", "norm", "attention"}:
            raise ValueError(f"unsupported aggregation mode {mode!r}")
        self.mode = mode
        if mode == "attention":
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )

    def forward(self, atom_hiddens: Tensor, batch: Tensor, num_graphs: int) -> Tensor:
        device = atom_hiddens.device
        dtype = atom_hiddens.dtype
        hidden_dim = atom_hiddens.shape[-1]
        if num_graphs == 0 or atom_hiddens.shape[0] == 0:
            return torch.zeros((num_graphs, hidden_dim), device=device, dtype=dtype)
        if self.mode == "attention":
            logits = self.gate(atom_hiddens).squeeze(-1)

            max_logits = torch.full((num_graphs,), fill_value=-1e9, device=device, dtype=dtype)
            max_logits.scatter_reduce_(0, batch, logits, reduce="amax")
            exp_logits = torch.exp(logits - max_logits[batch])
            sum_exp = torch.zeros(num_graphs, device=device, dtype=dtype)
            sum_exp.scatter_add_(0, batch, exp_logits)
            weights = (exp_logits / (sum_exp[batch] + 1e-8)).unsqueeze(-1)
            weighted_atoms = atom_hiddens * weights
            out = torch.zeros((num_graphs, hidden_dim), device=device, dtype=dtype)
            out.scatter_add_(0, batch.unsqueeze(-1).expand_as(weighted_atoms), weighted_atoms)
            return out
        out = torch.zeros((num_graphs, hidden_dim), device=device, dtype=dtype)
        out.scatter_add_(0, batch.unsqueeze(-1).expand_as(atom_hiddens), atom_hiddens)

        if self.mode == "sum":
            return out
        counts = torch.zeros(num_graphs, device=device, dtype=dtype)
        counts.scatter_add_(0, batch, torch.ones_like(batch, dtype=dtype))
        counts = counts.clamp(min=1.0).unsqueeze(-1)
        if self.mode == "mean":
            return out / counts
        if self.mode == "norm":
            return out / torch.sqrt(counts)
        if self.mode == "max":
            max_out = torch.full((num_graphs, hidden_dim), fill_value=-1e9, device=device, dtype=dtype)
            max_out.scatter_reduce_(0, batch.unsqueeze(-1).expand_as(atom_hiddens), atom_hiddens, reduce="amax")
            max_out[max_out <= -1e8] = 0.0
            return max_out
        return out

class DMPNNEncoder(nn.Module):
    def __init__(self, atom_dim: int = ATOM_FEATURE_DIM, bond_dim: int = BOND_FEATURE_DIM, hidden_dim: int = 300, depth: int = 3, dropout: float = 0.0, aggregation: str = "mean", use_virtual_node: bool = False,) -> None:
        super().__init__()
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.dropout = dropout
        self.use_virtual_node = use_virtual_node

        self.w_i = nn.Linear(atom_dim + bond_dim, hidden_dim, bias=False)

        self.w_m = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.w_a = nn.Linear(atom_dim + hidden_dim, hidden_dim)
        if use_virtual_node:
            self.vn_embedding = nn.Parameter(torch.zeros(1, hidden_dim))
            self.vn_mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(max(1, depth - 1))
            ])
            self.vn_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim)
                for _ in range(max(1, depth - 1))
            ])
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.pool = GraphAggregation(aggregation, hidden_dim)

    def forward(self, batch_graph: BatchMolGraph) -> Tensor:
        num_atoms = batch_graph.num_atoms
        num_edges = batch_graph.num_edges
        num_graphs = batch_graph.num_graphs
        if num_graphs == 0 or num_atoms == 0:
            return torch.zeros((num_graphs, self.hidden_dim), device=batch_graph.atom_features.device)
        vn = self.vn_embedding.expand(num_graphs, -1) if self.use_virtual_node else None
        if num_edges == 0:

            zero_edges = torch.zeros((num_atoms, self.hidden_dim), device=batch_graph.atom_features.device)
            atom_input = torch.cat([batch_graph.atom_features, zero_edges], dim=-1)
            atom_hiddens = self.act(self.w_a(atom_input))
            if self.use_virtual_node and vn is not None:
                atom_hiddens = atom_hiddens + vn[batch_graph.batch]
            return self.pool(atom_hiddens, batch_graph.batch, num_graphs)

        src_nodes = batch_graph.edge_index[0]
        edge_inputs = torch.cat([batch_graph.atom_features[src_nodes], batch_graph.edge_features], dim=-1)
        h_0 = self.act(self.w_i(edge_inputs))
        h = h_0

        b2a = batch_graph.b2a
        b2rev = batch_graph.b2rev

        for step in range(self.depth - 1):

            atom_messages = torch.zeros((num_atoms, self.hidden_dim), device=h.device, dtype=h.dtype)
            atom_messages.scatter_add_(0, b2a.unsqueeze(-1).expand_as(h), h)
            if self.use_virtual_node and vn is not None:
                graph_agg = torch.zeros((num_graphs, self.hidden_dim), device=h.device, dtype=h.dtype)
                graph_agg.scatter_add_(0, batch_graph.batch.unsqueeze(-1).expand_as(atom_messages), atom_messages)
                vn_update = self.vn_mlps[step](self.vn_norms[step](vn + graph_agg))
                vn = vn + self.drop(vn_update)
                atom_messages = atom_messages + vn[batch_graph.batch]

            incoming_to_src = atom_messages[src_nodes]
            reverse_message = h[b2rev]
            m = incoming_to_src - reverse_message
            h = self.drop(self.act(h_0 + self.w_m(m)))

        final_atom_messages = torch.zeros((num_atoms, self.hidden_dim), device=h.device, dtype=h.dtype)
        final_atom_messages.scatter_add_(0, b2a.unsqueeze(-1).expand_as(h), h)
        if self.use_virtual_node and vn is not None:
            final_atom_messages = final_atom_messages + vn[batch_graph.batch]

        atom_repr_input = torch.cat([batch_graph.atom_features, final_atom_messages], dim=-1)
        atom_hiddens = self.drop(self.act(self.w_a(atom_repr_input)))

        graph_repr = self.pool(atom_hiddens, batch_graph.batch, num_graphs)
        return graph_repr

class GINEncoder(nn.Module):
    def __init__(self, atom_dim: int = ATOM_FEATURE_DIM, hidden_dim: int = 300, depth: int = 3, dropout: float = 0.0, aggregation: str = "mean", use_virtual_node: bool = False, ) -> None:
        super().__init__()
        self.atom_dim = atom_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.dropout = dropout
        self.use_virtual_node = use_virtual_node
        self.input_proj = nn.Linear(atom_dim, hidden_dim)
        self.layers = nn.ModuleList()
        self.eps = nn.ParameterList()
        if use_virtual_node:
            self.vn_embedding = nn.Parameter(torch.zeros(1, hidden_dim))
            self.vn_mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(depth)
            ])
            self.vn_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim)
                for _ in range(depth)
            ])
        for _ in range(depth):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
            )
            self.layers.append(mlp)
            self.eps.append(nn.Parameter(torch.zeros(1)))
        self.pool = GraphAggregation(aggregation, hidden_dim)

    def forward(self, batch_graph: BatchMolGraph) -> Tensor:
        h = F.relu(self.input_proj(batch_graph.atom_features))
        num_atoms = batch_graph.num_atoms
        edge_index = batch_graph.edge_index
        num_graphs = batch_graph.num_graphs
        vn = self.vn_embedding.expand(num_graphs, -1) if (self.use_virtual_node and num_graphs > 0) else None
        for step, (layer, eps) in enumerate(zip(self.layers, self.eps)):
            if self.use_virtual_node and vn is not None and num_atoms > 0:
                h = h + vn[batch_graph.batch]
            if edge_index.shape[1] > 0:
                neighbor_sum = torch.zeros_like(h)
                src, dst = edge_index[0], edge_index[1]
                neighbor_sum.scatter_add_(0, dst.unsqueeze(-1).expand_as(h[src]), h[src])
            else:
                neighbor_sum = torch.zeros_like(h)
            h = layer((1.0 + eps) * h + neighbor_sum)
            if self.use_virtual_node and vn is not None and num_atoms > 0:
                graph_agg = torch.zeros((num_graphs, self.hidden_dim), device=h.device, dtype=h.dtype)
                graph_agg.scatter_add_(0, batch_graph.batch.unsqueeze(-1).expand_as(h), h)
                vn_update = self.vn_mlps[step](self.vn_norms[step](vn + graph_agg))
                vn = vn + vn_update
        return self.pool(h, batch_graph.batch, batch_graph.num_graphs)

class EGNNLayer(nn.Module):
    def __init__(self, hidden_dim: int, cutoff: float = 5.0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cutoff = cutoff

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h: Tensor, x: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        if edge_index.shape[1] == 0:
            return h, x
        src, dst = edge_index[0], edge_index[1]
        diff = x[src] - x[dst]
        dist_sq = (diff ** 2).sum(dim=-1, keepdim=True)

        msg_input = torch.cat([h[src], h[dst], dist_sq], dim=-1)
        m_ij = self.edge_mlp(msg_input)

        coord_weights = self.coord_mlp(m_ij).clamp(-10.0, 10.0)
        coord_delta = diff * coord_weights

        m_i = torch.zeros_like(h)
        m_i.scatter_add_(0, dst.unsqueeze(-1).expand_as(m_ij), m_ij)
        x_delta = torch.zeros_like(x)
        x_delta.scatter_add_(0, dst.unsqueeze(-1).expand_as(coord_delta), coord_delta)
        x_new = x + x_delta
        h_new = h + self.node_mlp(torch.cat([h, m_i], dim=-1))
        return h_new, x_new

class EGNNEncoder(nn.Module):
    def __init__(self, atom_dim: int = ATOM_FEATURE_DIM, hidden_dim: int = 128, depth: int = 3, cutoff: float = 5.0, aggregation: str = "mean",) -> None:
        super().__init__()
        self.input_proj = nn.Linear(atom_dim, hidden_dim)
        self.layers = nn.ModuleList([EGNNLayer(hidden_dim, cutoff=cutoff) for _ in range(depth)])
        self.pool = GraphAggregation(aggregation, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, batch_graph: BatchMolGraph) -> Tensor:
        h = self.input_proj(batch_graph.atom_features)
        coords = batch_graph.coordinates
        if coords is None:

            coords = torch.zeros((batch_graph.num_atoms, 3), device=h.device, dtype=h.dtype)
        edge_index = batch_graph.edge_index
        for layer in self.layers:
            h, coords = layer(h, coords, edge_index)
        return self.pool(h, batch_graph.batch, batch_graph.num_graphs)

class ProteinEncoder(nn.Module):
    def __init__(self, vocab_size: int = 30, embedding_dim: int = 64, channels: int = 128, kernel_size: int = 7, max_length: int = 1000,) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embedding_dim, channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.fc = nn.Linear(channels, channels)
        self.channels = channels

    def forward(self, token_ids: Tensor) -> Tensor:
        x = self.embedding(token_ids).transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        pooled = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        return self.fc(pooled)

__all__ = [
    "DMPNNEncoder",
    "EGNNEncoder",
    "GINEncoder",
    "GraphAggregation",
    "ProteinEncoder",
]
