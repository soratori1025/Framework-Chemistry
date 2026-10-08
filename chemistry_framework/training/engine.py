"""Small PyTorch training loop for user-assembled models and tensor loaders."""
from __future__ import annotations
import copy
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from ..config.schema import TrainingConfig

@dataclass(frozen=True)
class TrainingResult:
    model: nn.Module
    history: tuple[dict[str, float], ...]
    best_epoch: int
    best_validation_loss: float
    checkpoint_path: Path | None

def _device(name: str) -> torch.device:
    if name != "auto":
        selected = torch.device(name)
        if selected.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("training requested CUDA but CUDA is unavailable")
        if selected.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("training requested MPS but MPS is unavailable")
        return selected
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _loss_function(config: TrainingConfig) -> nn.Module:
    if config.loss == "mse":
        return nn.MSELoss()
    if config.loss == "mae":
        return nn.L1Loss()
    if config.loss == "huber":
        return nn.HuberLoss()
    if config.loss == "bce":
        return nn.BCEWithLogitsLoss(reduction="none")
    raise ValueError("training.loss='custom' requires passing loss_fn")

def _prediction(model: nn.Module, inputs: Any) -> Tensor:
    output = model(inputs)
    if isinstance(output, Tensor):
        return output
    if isinstance(output, dict) and isinstance(output.get("prediction"), Tensor):
        return output["prediction"]
    raise TypeError("model must return a tensor or a mapping containing tensor 'prediction'")

def _batch_loss(model: nn.Module, batch: Any, loss_fn: nn.Module, device: torch.device,) -> tuple[Tensor, int]:
    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise TypeError("training batches must be (inputs, targets) pairs")
    inputs, targets = batch
    if isinstance(inputs, Tensor):
        inputs = inputs.to(device)
    elif isinstance(inputs, dict):
        inputs = {key: value.to(device) if isinstance(value, Tensor) else value
                  for key, value in inputs.items()}
    targets = targets.to(device)
    prediction = _prediction(model, inputs)
    if prediction.shape != targets.shape:
        if prediction.numel() == targets.numel():
            prediction = prediction.reshape_as(targets)
        else:
            raise ValueError(
                f"prediction shape {tuple(prediction.shape)} does not match "
                f"target shape {tuple(targets.shape)}"
            )
    if isinstance(loss_fn, nn.BCEWithLogitsLoss):
        valid = torch.isfinite(targets)
        if not valid.any():
            return prediction.sum() * 0.0, 0
        valid_targets = targets[valid]
        if torch.any((valid_targets != 0) & (valid_targets != 1)):
            raise ValueError("BCE targets must contain only 0, 1, or non-finite missing values")
        elementwise_loss = F.binary_cross_entropy_with_logits(
            prediction,
            torch.nan_to_num(targets),
            weight=loss_fn.weight,
            pos_weight=loss_fn.pos_weight,
            reduction="none",
        )
        return elementwise_loss[valid].mean(), int(valid.sum().item())
    valid = torch.isfinite(targets)
    if valid.ndim > 1:
        valid = valid.all(dim=tuple(range(1, valid.ndim)))
    if not valid.any():
        return prediction.sum() * 0.0, 0
    if targets.ndim > 1:
        valid_values = valid.unsqueeze(-1).expand_as(targets)
        loss = loss_fn(prediction[valid_values], targets[valid_values])
        count = int(valid_values.sum().item())
    else:
        loss = loss_fn(prediction[valid], targets[valid])
        count = int(valid.sum().item())
    return loss, count

def _epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device, optimizer: torch.optim.Optimizer | None,) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss, count = _batch_loss(model, batch, loss_fn, device)
            if count == 0:
                continue
            if training:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach().item()) * count
            total_count += count
    if total_count == 0:
        raise ValueError("epoch has no finite targets")
    return total_loss / total_count

def train_model(model: nn.Module, train_loader: DataLoader, validation_loader: DataLoader, config: TrainingConfig, *, loss_fn: nn.Module | None = None, checkpoint_path: str | Path | None = None, checkpoint_metadata: dict[str, Any] | None = None,) -> TrainingResult:
    """Train an assembled model with deterministic seeding and best-val restore."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = _device(config.device)
    model.to(device)
    criterion = loss_fn or _loss_function(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: list[dict[str, float]] = []
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, config.epochs + 1):
        train_loss = _epoch(model, train_loader, criterion, device, optimizer)
        validation_loss = _epoch(model, validation_loader, criterion, device, None)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if config.patience is not None and epochs_without_improvement >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("training completed without a valid best checkpoint")
    model.load_state_dict(best_state)
    destination = Path(checkpoint_path).expanduser() if checkpoint_path is not None else None
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": best_state,
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
                "training_config": asdict(config),
                "metadata": checkpoint_metadata or {},
            },
            destination,
        )
    return TrainingResult(
        model=model,
        history=tuple(history),
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        checkpoint_path=destination,
    )
