from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np
import torch
from torch import Tensor

_ATOMIC_NUMBERS = (1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 26, 29, 30, 34, 35, 53)
_DEGREES = (0, 1, 2, 3, 4, 5)
_FORMAL_CHARGES = (-2, -1, 0, 1, 2)
_HYBRIDIZATIONS = ("SP", "SP2", "SP3", "SP3D", "SP3D2")
_NUM_HYDROGENS = (0, 1, 2, 3, 4)
_BOND_TYPES = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
_STEREO_TAGS = (0, 1, 2, 3, 4)

def _one_hot(value: Any, choices: Sequence[Any]) -> list[float]:
    encoding = [1.0 if value == choice else 0.0 for choice in choices]
    encoding.append(1.0 if value not in choices else 0.0)
    return encoding

def atom_features(atom: Any) -> list[float]:
    hyb = str(atom.GetHybridization()).split(".")[-1]
    features = (
        _one_hot(atom.GetAtomicNum(), _ATOMIC_NUMBERS)
        + _one_hot(atom.GetTotalDegree(), _DEGREES)
        + _one_hot(atom.GetFormalCharge(), _FORMAL_CHARGES)
        + _one_hot(int(atom.GetChiralTag()), (0, 1, 2, 3))
        + _one_hot(atom.GetTotalNumHs(includeNeighbors=True), _NUM_HYDROGENS)
        + _one_hot(hyb, _HYBRIDIZATIONS)
        + [
            1.0 if atom.GetIsAromatic() else 0.0,
            float(atom.GetMass()) * 0.01,
        ]
    )
    return features

def bond_features(bond: Any) -> list[float]:
    b_type = str(bond.GetBondType()).split(".")[-1]
    features = (
        _one_hot(b_type, _BOND_TYPES)
        + [
            1.0 if bond.GetIsConjugated() else 0.0,
            1.0 if bond.IsInRing() else 0.0,
        ]
        + _one_hot(int(bond.GetStereo()), _STEREO_TAGS)
    )
    return features

ATOM_FEATURE_DIM = len(atom_features(_MockAtom(6))) if False else (
    (len(_ATOMIC_NUMBERS) + 1)
    + (len(_DEGREES) + 1)
    + (len(_FORMAL_CHARGES) + 1)
    + (4 + 1)
    + (len(_NUM_HYDROGENS) + 1)
    + (len(_HYBRIDIZATIONS) + 1)
    + 2
)
BOND_FEATURE_DIM = (len(_BOND_TYPES) + 1) + 2 + (len(_STEREO_TAGS) + 1)

@dataclass(frozen=True)
class MolGraph:
    smiles: str
    num_atoms: int
    num_bonds: int
    num_edges: int
    atom_features: Tensor
    edge_features: Tensor
    edge_index: Tensor
    a2b: list[list[int]]
    b2a: list[int]
    b2rev: list[int]
    coordinates: Tensor | None = None

def mol_to_graph(mol_or_smiles: Any, coordinates: np.ndarray | None = None) -> MolGraph:
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError("RDKit is required to construct molecular graphs") from exc
    if isinstance(mol_or_smiles, str):
        mol = Chem.MolFromSmiles(mol_or_smiles)
        smiles = mol_or_smiles
    else:
        mol = mol_or_smiles
        smiles = Chem.MolToSmiles(mol) if mol is not None else ""
    if mol is None:
        raise ValueError(f"invalid molecule: {mol_or_smiles!r}")
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        raise ValueError(f"molecule has zero atoms: {smiles!r}")

    atom_feat_list = [atom_features(atom) for atom in mol.GetAtoms()]
    atom_feat = torch.tensor(atom_feat_list, dtype=torch.float32)

    coords_tensor: Tensor | None = None
    if coordinates is not None:
        coords_arr = np.asarray(coordinates, dtype=np.float32)
        if coords_arr.shape == (num_atoms, 3):
            coords_tensor = torch.from_numpy(coords_arr)
    elif mol.GetNumConformers() > 0:
        pos = mol.GetConformer().GetPositions()
        coords_tensor = torch.tensor(pos, dtype=torch.float32)

    bonds = list(mol.GetBonds())
    num_bonds = len(bonds)
    num_edges = num_bonds * 2
    if num_edges == 0:

        edge_feat = torch.empty((0, BOND_FEATURE_DIM), dtype=torch.float32)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        a2b = [[] for _ in range(num_atoms)]
        b2a = []
        b2rev = []
    else:
        edge_feat_list: list[list[float]] = []
        src_nodes: list[int] = []
        dst_nodes: list[int] = []
        a2b = [[] for _ in range(num_atoms)]
        b2a = []
        b2rev = []
        for b_idx, bond in enumerate(bonds):
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            feat = bond_features(bond)

            fwd_idx = len(b2a)

            bwd_idx = fwd_idx + 1
            edge_feat_list.append(feat)
            src_nodes.append(u)
            dst_nodes.append(v)
            a2b[v].append(fwd_idx)
            b2a.append(v)
            b2rev.append(bwd_idx)
            edge_feat_list.append(feat)
            src_nodes.append(v)
            dst_nodes.append(u)
            a2b[u].append(bwd_idx)
            b2a.append(u)
            b2rev.append(fwd_idx)
        edge_feat = torch.tensor(edge_feat_list, dtype=torch.float32)
        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    return MolGraph(
        smiles=smiles,
        num_atoms=num_atoms,
        num_bonds=num_bonds,
        num_edges=num_edges,
        atom_features=atom_feat,
        edge_features=edge_feat,
        edge_index=edge_index,
        a2b=a2b,
        b2a=b2a,
        b2rev=b2rev,
        coordinates=coords_tensor,
    )

class BatchMolGraph:
    def __init__(self, graphs: Sequence[MolGraph]) -> None:
        if not graphs:
            raise ValueError("BatchMolGraph requires at least one graph")
        self.graphs = tuple(graphs)
        self.num_graphs = len(graphs)

        atom_counts = [g.num_atoms for g in graphs]
        edge_counts = [g.num_edges for g in graphs]
        self.num_atoms = sum(atom_counts)
        self.num_edges = sum(edge_counts)

        self.atom_features = torch.cat([g.atom_features for g in graphs], dim=0)

        if self.num_edges > 0:
            self.edge_features = torch.cat([g.edge_features for g in graphs], dim=0)
        else:
            self.edge_features = torch.empty((0, BOND_FEATURE_DIM), dtype=torch.float32)

        batch_ids: list[int] = []
        atom_scopes: list[tuple[int, int]] = []
        edge_scopes: list[tuple[int, int]] = []
        atom_offset = 0
        edge_offset = 0
        batched_edge_indices: list[Tensor] = []
        self.b2a_list: list[int] = []
        self.b2rev_list: list[int] = []
        self.a2b_list: list[list[int]] = []
        all_have_coords = all(g.coordinates is not None for g in graphs)
        coords_list: list[Tensor] = []
        for g_idx, g in enumerate(graphs):
            atom_scopes.append((atom_offset, g.num_atoms))
            edge_scopes.append((edge_offset, g.num_edges))
            batch_ids.extend([g_idx] * g.num_atoms)
            if g.num_edges > 0:
                batched_edge_indices.append(g.edge_index + atom_offset)
                self.b2a_list.extend([target + atom_offset for target in g.b2a])
                self.b2rev_list.extend([rev + edge_offset for rev in g.b2rev])
            for incoming in g.a2b:
                self.a2b_list.append([b + edge_offset for b in incoming])
            if all_have_coords:
                assert g.coordinates is not None
                coords_list.append(g.coordinates)
            atom_offset += g.num_atoms
            edge_offset += g.num_edges
        self.batch = torch.tensor(batch_ids, dtype=torch.long)
        self.atom_scopes = tuple(atom_scopes)
        self.edge_scopes = tuple(edge_scopes)
        if batched_edge_indices:
            self.edge_index = torch.cat(batched_edge_indices, dim=1)
        else:
            self.edge_index = torch.empty((2, 0), dtype=torch.long)
        self.coordinates = torch.cat(coords_list, dim=0) if all_have_coords else None

        self.b2a = torch.tensor(self.b2a_list, dtype=torch.long) if self.b2a_list else torch.empty(0, dtype=torch.long)
        self.b2rev = torch.tensor(self.b2rev_list, dtype=torch.long) if self.b2rev_list else torch.empty(0, dtype=torch.long)

        max_num_bonds = max((len(b_list) for b_list in self.a2b_list), default=0)
        self.max_degree = max_num_bonds
        padded_a2b = torch.full((self.num_atoms, max(1, max_num_bonds)), fill_value=-1, dtype=torch.long)
        degree = torch.zeros(self.num_atoms, dtype=torch.long)
        for a_idx, b_list in enumerate(self.a2b_list):
            degree[a_idx] = len(b_list)
            for pos, b_idx in enumerate(b_list):
                padded_a2b[a_idx, pos] = b_idx
        self.a2b = padded_a2b
        self.degree = degree

    def to(self, device: torch.device | str) -> BatchMolGraph:
        self.atom_features = self.atom_features.to(device)
        self.edge_features = self.edge_features.to(device)
        self.edge_index = self.edge_index.to(device)
        self.batch = self.batch.to(device)
        self.b2a = self.b2a.to(device)
        self.b2rev = self.b2rev.to(device)
        self.a2b = self.a2b.to(device)
        self.degree = self.degree.to(device)
        if self.coordinates is not None:
            self.coordinates = self.coordinates.to(device)
        return self

def collate_mol_graphs(items: Sequence[MolGraph | str | Any]) -> BatchMolGraph:
    graphs = [
        item if isinstance(item, MolGraph) else mol_to_graph(item)
        for item in items
    ]
    return BatchMolGraph(graphs)

__all__ = [
    "ATOM_FEATURE_DIM",
    "BOND_FEATURE_DIM",
    "BatchMolGraph",
    "MolGraph",
    "atom_features",
    "bond_features",
    "collate_mol_graphs",
    "mol_to_graph",
]
