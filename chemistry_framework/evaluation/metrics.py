"""Regression metrics with explicit missing-target handling."""
from __future__ import annotations
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
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
    actual: ArrayLike,
    predicted: ArrayLike,
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

def _binary_auroc(actual: np.ndarray, scores: np.ndarray) -> float:
    positive = actual == 1
    n_positive = int(positive.sum())
    n_negative = int(len(actual) - n_positive)
    if n_positive == 0 or n_negative == 0:
        raise ValueError("AUROC requires both positive and negative examples")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - n_positive * (n_positive + 1) / 2.0) / (
        n_positive * n_negative
    )

def _binary_auprc(actual: np.ndarray, scores: np.ndarray) -> float:
    n_positive = int(np.count_nonzero(actual == 1))
    if n_positive == 0:
        raise ValueError("AUPRC requires at least one positive example")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_actual = actual[order]
    true_positives = np.cumsum(sorted_actual == 1)
    group_ends = np.r_[
        np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]),
        len(sorted_scores) - 1,
    ]
    true_positives_at_threshold = true_positives[group_ends]
    precision = true_positives_at_threshold / (group_ends + 1)
    recall_increments = np.diff(
        np.r_[0, true_positives_at_threshold]
    ) / n_positive
    return float(np.sum(precision * recall_increments))

def classification_metrics(
    actual: ArrayLike,
    predicted: ArrayLike,
    metrics: Sequence[str] = ("auroc",),
) -> dict[str, float | int]:
    """Compute binary or macro multi-label metrics from predicted probabilities."""
    y_true = np.asarray(actual, dtype=np.float64)
    y_score = np.asarray(predicted, dtype=np.float64)
    if y_true.shape != y_score.shape or y_true.ndim not in {1, 2}:
        raise ValueError("actual and predicted values must be equal-shaped 1-D or 2-D arrays")
    valid = np.isfinite(y_true) & np.isfinite(y_score)
    if not valid.any():
        raise ValueError("no finite actual/predicted pairs available for evaluation")
    valid_targets = y_true[valid]
    if np.any((valid_targets != 0) & (valid_targets != 1)):
        raise ValueError("classification targets must contain only 0, 1, or missing values")
    if np.any((y_score[valid] < 0) | (y_score[valid] > 1)):
        raise ValueError("classification predictions must be probabilities in [0, 1]")

    allowed = {"auroc", "auprc", "accuracy", "precision", "recall", "f1"}
    if not metrics or set(metrics) - allowed:
        unsupported = sorted(set(metrics) - allowed)
        raise ValueError(f"unsupported classification metrics: {', '.join(unsupported)}")
    columns = y_true.reshape(-1, 1) if y_true.ndim == 1 else y_true
    score_columns = y_score.reshape(-1, 1) if y_score.ndim == 1 else y_score
    result: dict[str, float | int] = {"count": int(valid.sum())}
    binary_targets = y_true[valid]
    binary_predictions = y_score[valid] >= 0.5
    true_positive = int(np.count_nonzero(binary_predictions & (binary_targets == 1)))
    false_positive = int(np.count_nonzero(binary_predictions & (binary_targets == 0)))
    false_negative = int(np.count_nonzero(~binary_predictions & (binary_targets == 1)))
    if "accuracy" in metrics:
        result["accuracy"] = float(np.mean(binary_predictions == binary_targets))
    if "precision" in metrics:
        result["precision"] = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
    if "recall" in metrics:
        result["recall"] = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
    if "f1" in metrics:
        denominator = 2 * true_positive + false_positive + false_negative
        result["f1"] = 2 * true_positive / denominator if denominator else 0.0
    for metric, score_function in (
        ("auroc", _binary_auroc),
        ("auprc", _binary_auprc),
    ):
        if metric not in metrics:
            continue
        values: list[float] = []
        for column, score_column in zip(columns.T, score_columns.T):
            column_valid = np.isfinite(column) & np.isfinite(score_column)
            if not column_valid.any():
                continue
            column_actual = column[column_valid]
            if metric == "auroc" and len(np.unique(column_actual)) < 2:
                continue
            if metric == "auprc" and not np.any(column_actual == 1):
                continue
            values.append(score_function(column_actual, score_column[column_valid]))
        if not values:
            raise ValueError(f"{metric.upper()} is undefined for all target columns")
        result[metric] = float(np.mean(values))
    return result

def evaluate_model(model: nn.Module, loader: DataLoader, metrics: Sequence[str] = ("mae", "rmse", "r2"), *, device: str | torch.device = "cpu", task_kind: str = "regression",) -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
    """Evaluate a tensor-output model and return metrics, targets, predictions."""
    if task_kind not in {
        "regression", "binary_classification", "multilabel_classification"
    }:
        raise ValueError(f"unsupported evaluation task_kind {task_kind!r}")
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
            actual_array = targets.detach().cpu().numpy()
            predicted_array = output.detach().cpu().numpy()
            if task_kind == "regression":
                actual_array = actual_array.reshape(-1)
                predicted_array = predicted_array.reshape(-1)
            actual_batches.append(actual_array)
            predicted_batches.append(predicted_array)
    if not actual_batches:
        raise ValueError("evaluation loader yielded no batches")
    actual = np.concatenate(actual_batches)
    predicted = np.concatenate(predicted_batches)
    if task_kind in {"binary_classification", "multilabel_classification"}:
        predicted = 1.0 / (1.0 + np.exp(-np.clip(predicted, -709.0, 709.0)))
        return classification_metrics(actual, predicted, metrics), actual, predicted
    return regression_metrics(actual, predicted, metrics), actual, predicted

def load_model_checkpoint(model: nn.Module, checkpoint_path: str | Path, *, device: str | torch.device = "cpu",) -> nn.Module:
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
