"""Tensor data loaders for descriptor-based tasks."""
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
if TYPE_CHECKING:
    from ..experiment import PreparedExperiment

@dataclass(frozen=True)
class TensorDataLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader

def make_tensor_dataloaders(experiment: PreparedExperiment, target_name: str, *, batch_size: int | None = None, num_workers: int | None = None, task_name: str | None = None,) -> TensorDataLoaders:
    """Build loaders for a target, applying its configured task descriptor route."""
    if target_name not in experiment.dataset.targets:
        raise KeyError(f"unknown dataset target {target_name!r}")
    batch_size = batch_size or experiment.config.training.batch_size
    num_workers = (
        experiment.config.training.num_workers if num_workers is None else num_workers
    )
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    if task_name is not None:
        task = experiment.config.tasks.get(task_name)
        if task is None:
            raise KeyError(f"unknown task {task_name!r}")
        if task["target"] != target_name:
            raise ValueError(
                f"task {task_name!r} uses target {task['target']!r}, not {target_name!r}"
            )
    else:
        task = experiment.config.tasks.get(target_name)
    if task is None:
        matching_tasks = [
            configured_task
            for configured_task in experiment.config.tasks.values()
            if configured_task["target"] == target_name
        ]
        if len(matching_tasks) == 1:
            task = matching_tasks[0]
        elif len(matching_tasks) > 1:
            raise ValueError(
                f"multiple tasks use target {target_name!r}; select a task-specific "
                "route with the task_name argument"
            )
    selected_names = (
        experiment.route_features[task["route"]]
        if task is not None and task.get("route") is not None
        else experiment.features.names
    )
    if not selected_names:
        raise ValueError(f"no descriptor features are configured for target {target_name!r}")
    selected_columns = [experiment.features.names.index(name) for name in selected_names]
    features = torch.as_tensor(
        experiment.features.values[:, selected_columns],
        dtype=torch.float32,
    )
    targets = np.asarray(experiment.dataset.targets[target_name], dtype=np.float32)

    def loader(indices: np.ndarray, shuffle: bool) -> DataLoader:
        row_targets = targets[indices]
        finite = np.isfinite(row_targets)
        valid_rows = finite if finite.ndim == 1 else finite.all(axis=1)
        valid_indices = indices[valid_rows]
        if len(valid_indices) == 0:
            raise ValueError(
                f"split has no finite labels for target {target_name!r}"
            )
        dataset = TensorDataset(
            features[valid_indices],
            torch.as_tensor(targets[valid_indices], dtype=torch.float32),
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    return TensorDataLoaders(
        train=loader(experiment.split.train, shuffle=True),
        validation=loader(experiment.split.validation, shuffle=False),
        test=loader(experiment.split.test, shuffle=False),
    )
