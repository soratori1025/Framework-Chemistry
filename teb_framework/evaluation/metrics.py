"""Regression metrics with explicit missing-target handling."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    selected = torch.device(device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("evaluation requested CUDA but CUDA is unavailable")
    if (
        selected.type == "mps"
        and (not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available())
    ):
        raise RuntimeError("evaluation requested MPS but MPS is unavailable")
    return selected


def regression_metrics(
    actual: Sequence[float],
    predicted: Sequence[float],
    metrics: Sequence[str] = ("mae", "rmse", "r2"),
) -> dict[str, float | int]:
    y_true = np.asarray(actual, dtype=np.float64)
    y_pred = np.asarray(predicted, dtype=np.float64)
    if y_true.shape != y_pred.shape or y_true.ndim not in {1, 2}:
        raise ValueError("actual and predicted values must be equal-shaped 1-D or 2-D arrays")
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]
    if not len(y_true):
        raise ValueError("no finite actual/predicted pairs available for evaluation")
    error = y_pred - y_true
    result: dict[str, float | int] = {"count": int(len(y_true))}
    for metric in metrics:
        if metric == "mae":
            result[metric] = float(np.mean(np.abs(error)))
        elif metric == "rmse":
            result[metric] = float(np.sqrt(np.mean(np.square(error))))
        elif metric == "r2":
            denominator = float(np.sum(np.square(y_true - y_true.mean())))
            result[metric] = (
                float(1.0 - np.sum(np.square(error)) / denominator)
                if denominator > 0
                else float("nan")
            )
        elif metric == "mape":
            nonzero = np.abs(y_true) > np.finfo(np.float64).eps
            if not nonzero.any():
                raise ValueError("MAPE is undefined when all actual values are zero")
            result[metric] = float(
                np.mean(np.abs(error[nonzero] / y_true[nonzero])) * 100.0
            )
        else:
            raise ValueError(f"unsupported metric {metric!r}")
    return result


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    metrics: Sequence[str] = ("mae", "rmse", "r2"),
    *,
    device: str | torch.device = "cpu",
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
    """Evaluate a tensor-output model and return metrics, targets, predictions."""
    selected_device = _resolve_device(device)
    model = model.to(selected_device)
    model.eval()
    actual_batches: list[np.ndarray] = []
    predicted_batches: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                raise TypeError("evaluation batches must be (inputs, targets) pairs")
            inputs, targets = batch
            if isinstance(inputs, Tensor):
                inputs = inputs.to(selected_device)
            elif isinstance(inputs, dict):
                inputs = {
                    key: value.to(selected_device) if isinstance(value, Tensor) else value
                    for key, value in inputs.items()
                }
            output = model(inputs)
            if isinstance(output, dict):
                output = output.get("prediction")
            if not isinstance(output, Tensor):
                raise TypeError("model must return a tensor or mapping with 'prediction'")
            if output.shape != targets.shape:
                if output.numel() != targets.numel():
                    raise ValueError(
                        f"prediction shape {tuple(output.shape)} does not match "
                        f"target shape {tuple(targets.shape)}"
                    )
                output = output.reshape_as(targets)
            actual_batches.append(targets.detach().cpu().numpy().reshape(-1))
            predicted_batches.append(output.detach().cpu().numpy().reshape(-1))
    if not actual_batches:
        raise ValueError("evaluation loader yielded no batches")
    actual = np.concatenate(actual_batches)
    predicted = np.concatenate(predicted_batches)
    return regression_metrics(actual, predicted, metrics), actual, predicted


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Load a framework checkpoint or plain state dict into a model."""
    selected_device = _resolve_device(device)
    checkpoint = torch.load(
        Path(checkpoint_path).expanduser(),
        map_location=selected_device,
        weights_only=True,
    )
    if isinstance(checkpoint, dict) and isinstance(
        checkpoint.get("model_state_dict"), dict
    ):
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and all(
        isinstance(name, str) and isinstance(value, Tensor)
        for name, value in checkpoint.items()
    ):
        state_dict = checkpoint
    else:
        raise ValueError(
            "checkpoint must contain 'model_state_dict' or be a plain tensor state dict"
        )
    model.to(selected_device)
    model.load_state_dict(state_dict)
    return model
