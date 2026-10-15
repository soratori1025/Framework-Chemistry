from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from .graph import BatchMolGraph, collate_mol_graphs
from ..inputs.adapters import get_input_adapter
from ..inputs.objects import ComplexObject, MixtureObject, ReactionObject

@dataclass(frozen=True)
class ExperimentDataLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader

_AMINO_ACID_VOCAB = {char: idx + 1 for idx, char in enumerate("ACDEFGHIKLMNOPQRSTUVWYXBZJ")}

def _tokenize_protein(seq: str, max_len: int = 1000) -> Tensor:
    tokens = [_AMINO_ACID_VOCAB.get(char, 0) for char in seq[:max_len]]
    if len(tokens) < max_len:
        tokens.extend([0] * (max_len - len(tokens)))
    return torch.tensor(tokens, dtype=torch.long)

class ChemistryDataset(Dataset):
    def __init__(self, indices: np.ndarray) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> int:
        return int(self.indices[idx])

def make_experiment_dataloaders(experiment: Any, target_name: str, *, batch_size: int | None = None, num_workers: int | None = None, task_name: str | None = None, structure_enabled: bool | None = None,) -> ExperimentDataLoaders:
    dataset = experiment.dataset
    if target_name not in dataset.targets:
        raise KeyError(f"unknown dataset target {target_name!r}")
    batch_size = batch_size or experiment.config.training.batch_size
    num_workers = experiment.config.training.num_workers if num_workers is None else num_workers

    task = experiment.config.tasks.get(task_name or target_name)
    if task is None:
        matching = [t for t in experiment.config.tasks.values() if t.get("target") == target_name]
        task = matching[0] if len(matching) == 1 else None
    task_kind = task.get("kind", "scalar") if task is not None else "scalar"
    is_classification = "classification" in task_kind

    route_name = task.get("route") if task is not None else None
    if route_name and route_name in experiment.route_features:
        selected_names = experiment.route_features[route_name]
    else:
        selected_names = experiment.features.names
    selected_columns = [experiment.features.names.index(n) for n in selected_names] if selected_names else []
    all_features = experiment.features.values[:, selected_columns] if selected_columns else None

    prior_name = task.get("prior") if task is not None else None
    prior_names = experiment.prior_features.get(prior_name, ()) if prior_name else ()
    prior_columns = [experiment.features.names.index(n) for n in prior_names] if prior_names else []
    all_priors = experiment.features.values[:, prior_columns] if prior_columns else None
    raw_targets = np.asarray(dataset.targets[target_name], dtype=np.float32)

    if structure_enabled is None:
        model_cfg = experiment.config.model or {}
        struct_cfg = model_cfg.get("structure") or {}
        enc = struct_cfg.get("encoder") if isinstance(struct_cfg, dict) else struct_cfg
        structure_enabled = enc not in {None, "none"}
    input_format = dataset_config_format = getattr(experiment.config.dataset, "input_format", "smiles")
    adapter = get_input_adapter(input_format) if input_format != "smiles" else None

    parsed_records: list[Any] | None = None
    if structure_enabled:
        if adapter is not None:
            parsed_records = [adapter.parse(s, row=idx + 2) for idx, s in enumerate(dataset.smiles)]
        else:
            parsed_records = list(dataset.smiles)

    def collate_fn(batch_indices: list[int]) -> tuple[dict[str, Any], Tensor]:
        b_idx = np.asarray(batch_indices, dtype=np.int64)
        targets_tensor = torch.as_tensor(raw_targets[b_idx], dtype=torch.float32)
        inputs_dict: dict[str, Any] = {}
        if all_features is not None:
            inputs_dict["descriptors"] = torch.as_tensor(all_features[b_idx], dtype=torch.float32)
        if all_priors is not None:
            inputs_dict["prior"] = torch.as_tensor(all_priors[b_idx], dtype=torch.float32)
        if structure_enabled and parsed_records is not None:
            batch_items = [parsed_records[i] for i in b_idx]
            first = batch_items[0]
            if isinstance(first, ReactionObject):
                r_mols = [rec.combined("reactants") for rec in batch_items]
                p_mols = [rec.combined("products") for rec in batch_items]
                inputs_dict["structure"] = {
                    "reactants": collate_mol_graphs(r_mols),
                    "products": collate_mol_graphs(p_mols),
                }
            elif isinstance(first, MixtureObject):

                max_comp = max(len(m.components) for m in batch_items)
                comp_graphs = []
                fractions_mat = torch.zeros((len(batch_items), max_comp), dtype=torch.float32)
                for c_i in range(max_comp):
                    mols_c = [
                        m.components[c_i].mol if c_i < len(m.components) else m.components[0].mol
                        for m in batch_items
                    ]
                    comp_graphs.append(collate_mol_graphs(mols_c))
                for row_i, m in enumerate(batch_items):
                    for c_i, frac in enumerate(m.fractions):
                        fractions_mat[row_i, c_i] = frac
                inputs_dict["structure"] = {
                    "mixture_graphs": comp_graphs,
                    "fractions": fractions_mat,
                }
            elif isinstance(first, ComplexObject):
                ligand_graphs = collate_mol_graphs([c.ligand.mol for c in batch_items])
                prot_tokens = torch.stack([_tokenize_protein(c.protein_sequence) for c in batch_items])
                inputs_dict["structure"] = {
                    "ligand": ligand_graphs,
                    "protein_tokens": prot_tokens,
                }
            else:
                inputs_dict["graph"] = collate_mol_graphs(batch_items)
        return inputs_dict, targets_tensor

    def make_loader(split_indices: np.ndarray, shuffle: bool) -> DataLoader:
        row_targets = raw_targets[split_indices]
        finite = np.isfinite(row_targets)
        if is_classification:
            valid_rows = finite if finite.ndim == 1 else finite.any(axis=1)
        else:
            valid_rows = finite if finite.ndim == 1 else finite.all(axis=1)
        valid_indices = split_indices[valid_rows]
        if len(valid_indices) == 0:
            raise ValueError(f"split has no finite targets for {target_name!r}")
        return DataLoader(
            ChemistryDataset(valid_indices),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

    return ExperimentDataLoaders(
        train=make_loader(experiment.split.train, shuffle=True),
        validation=make_loader(experiment.split.validation, shuffle=False),
        test=make_loader(experiment.split.test, shuffle=False),
    )

__all__ = [
    "ChemistryDataset",
    "ExperimentDataLoaders",
    "make_experiment_dataloaders",
]
