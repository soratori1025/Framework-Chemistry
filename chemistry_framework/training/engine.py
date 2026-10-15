from __future__ import annotations
import copy
import math
from pathlib import Path
import random
from dataclasses import asdict, dataclass
from typing import Any, Callable
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from ..config.schema import TrainingConfig
from ..data.graph import BatchMolGraph
from ..models.ensemble import EnsembleModel

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

def _transfer_to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, Tensor):
        return obj.to(device)
    if isinstance(obj, BatchMolGraph):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _transfer_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_transfer_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_transfer_to_device(v, device) for v in obj)
    return obj

def _gaussian_nll_loss(output: dict[str, Tensor], targets: Tensor) -> Tensor:
    pred = output.get("prediction", output.get("mean"))
    var = output.get("variance")
    if var is None:
        logvar = output.get("logvar", torch.zeros_like(pred))
        var = torch.exp(logvar)
    return 0.5 * torch.log(var.clamp(min=1e-8)) + 0.5 * ((targets - pred) ** 2) / (var.clamp(min=1e-8))

def _evidential_loss(output: dict[str, Tensor], targets: Tensor, coeff: float = 0.1) -> Tensor:
    gamma = output["gamma"]
    nu = output["nu"]
    alpha = output["alpha"]
    beta = output["beta"]

    omega = 2.0 * beta * (1.0 + nu)
    error_sq = (targets - gamma) ** 2
    nll = (
        0.5 * torch.log(math.pi / nu.clamp(min=1e-8))
        - alpha * torch.log(omega.clamp(min=1e-8))
        + (alpha + 0.5) * torch.log(error_sq * nu + omega + 1e-8)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )

    error = error_sq.sqrt()
    reg = error * (2.0 * nu + alpha)
    return nll + coeff * reg

def _focal_loss(logits: Tensor, targets: Tensor, gamma: float = 2.0, alpha: float = 0.25, pos_weight: Tensor | None = None,) -> Tensor:
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    focal_weight = alpha_t * ((1.0 - p_t).clamp(min=0.0, max=1.0) ** gamma)
    return focal_weight * bce

def _pairwise_ranking_loss(prediction: Tensor, targets: Tensor, base_loss: Tensor, margin: float = 0.1, rank_weight: float = 0.2,) -> Tensor:
    if prediction.shape[0] < 2:
        return base_loss
    pred_flat = prediction.reshape(-1)
    target_flat = targets.reshape(-1)
    diff_target = target_flat.unsqueeze(0) - target_flat.unsqueeze(1)
    valid_pairs = diff_target > 1e-4
    if not valid_pairs.any():
        return base_loss
    diff_pred = pred_flat.unsqueeze(0) - pred_flat.unsqueeze(1)
    margin_violations = F.relu(margin - diff_pred[valid_pairs])
    rank_loss = margin_violations.mean()
    return base_loss + rank_weight * rank_loss

def _loss_function(config: TrainingConfig) -> nn.Module:
    if config.loss == "mse":
        return nn.MSELoss()
    if config.loss == "mae":
        return nn.L1Loss()
    if config.loss == "huber":
        return nn.HuberLoss()
    if config.loss == "bce":
        pos_weight = torch.tensor([config.pos_weight]) if config.pos_weight is not None else None
        return nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    if config.loss == "pairwise_ranking":
        return nn.HuberLoss()
    if config.loss in {"gaussian_nll", "evidential", "focal"}:

        return nn.Identity()
    raise ValueError(f"training.loss={config.loss!r} requires passing loss_fn")

def _batch_loss(model: nn.Module, batch: Any, loss_fn: nn.Module, device: torch.device, config: TrainingConfig,) -> tuple[Tensor, int]:
    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise TypeError("training batches must be (inputs, targets) pairs")
    inputs, targets = batch
    inputs = _transfer_to_device(inputs, device)
    targets = targets.to(device)
    output = model(inputs)

    if config.loss == "gaussian_nll" and isinstance(output, dict):
        elementwise = _gaussian_nll_loss(output, targets)
        valid = torch.isfinite(targets)
        if not valid.any():
            return targets.sum() * 0.0, 0
        return elementwise[valid].mean(), int(valid.sum().item())
    if config.loss == "evidential" and isinstance(output, dict):
        elementwise = _evidential_loss(output, targets, coeff=config.evidential_lambda)
        valid = torch.isfinite(targets)
        if not valid.any():
            return targets.sum() * 0.0, 0
        return elementwise[valid].mean(), int(valid.sum().item())

    if isinstance(output, Tensor):
        prediction = output
    elif isinstance(output, dict) and isinstance(output.get("prediction"), Tensor):
        prediction = output["prediction"]
    elif isinstance(output, dict) and isinstance(output.get("mean"), Tensor):
        prediction = output["mean"]
    else:
        raise TypeError("model must return a tensor or a mapping containing tensor 'prediction'")
    if prediction.shape != targets.shape:
        if prediction.numel() == targets.numel():
            prediction = prediction.reshape_as(targets)
        else:
            raise ValueError(
                f"prediction shape {tuple(prediction.shape)} does not match "
                f"target shape {tuple(targets.shape)}"
            )
    if config.loss == "focal":
        valid = torch.isfinite(targets)
        if not valid.any():
            return prediction.sum() * 0.0, 0
        valid_targets = targets[valid]
        if torch.any((valid_targets != 0) & (valid_targets != 1)):
            raise ValueError("Focal loss targets must contain only 0, 1, or non-finite missing values")
        pos_weight = torch.tensor([config.pos_weight], device=device, dtype=prediction.dtype) if config.pos_weight is not None else None
        elementwise_loss = _focal_loss(
            prediction,
            torch.nan_to_num(targets),
            gamma=2.0,
            alpha=0.25,
            pos_weight=pos_weight,
        )
        return elementwise_loss[valid].mean(), int(valid.sum().item())
    if isinstance(loss_fn, nn.BCEWithLogitsLoss):
        valid = torch.isfinite(targets)
        if not valid.any():
            return prediction.sum() * 0.0, 0
        valid_targets = targets[valid]
        if torch.any((valid_targets != 0) & (valid_targets != 1)):
            raise ValueError("BCE targets must contain only 0, 1, or non-finite missing values")
        pos_weight = loss_fn.pos_weight.to(device) if loss_fn.pos_weight is not None else None
        elementwise_loss = F.binary_cross_entropy_with_logits(
            prediction,
            torch.nan_to_num(targets),
            weight=loss_fn.weight.to(device) if loss_fn.weight is not None else None,
            pos_weight=pos_weight,
            reduction="none",
        )
        return elementwise_loss[valid].mean(), int(valid.sum().item())
    if config.loss == "pairwise_ranking":
        valid = torch.isfinite(targets)
        if valid.ndim > 1:
            valid = valid.all(dim=tuple(range(1, valid.ndim)))
        if not valid.any():
            return prediction.sum() * 0.0, 0
        if targets.ndim > 1:
            valid_values = valid.unsqueeze(-1).expand_as(targets)
            pred_v = prediction[valid_values]
            targ_v = targets[valid_values]
            count = int(valid_values.sum().item())
        else:
            pred_v = prediction[valid]
            targ_v = targets[valid]
            count = int(valid.sum().item())
        base_loss = loss_fn(pred_v, targ_v)
        total_loss = _pairwise_ranking_loss(pred_v, targ_v, base_loss)
        return total_loss, count
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

def _build_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfig, total_steps: int) -> tuple[Any, bool]:
    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs)), False
    if config.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=max(1, (config.patience or 10) // 2)
        ), False
    if config.scheduler == "noam":
        warmup_steps = int(config.warmup_epochs * max(1, total_steps // max(1, config.epochs)))
        def lr_lambda(step: int) -> float:
            step = max(1, step)
            if warmup_steps > 0 and step <= warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return max(0.01, (step / max(1, warmup_steps)) ** -0.5)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), True
    return None, False

def _epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device, config: TrainingConfig, optimizer: torch.optim.Optimizer | None, scheduler: Any = None, is_step_scheduler: bool = False, scaler: Any = None,) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            if training:
                optimizer.zero_grad(set_to_none=True)
            use_amp = config.amp and device.type == "cuda"
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss, count = _batch_loss(model, batch, loss_fn, device, config)
            if count == 0:
                continue
            if training:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    if config.grad_clip is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if config.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    optimizer.step()
                if is_step_scheduler and scheduler is not None:
                    scheduler.step()
            total_loss += float(loss.detach().item()) * count
            total_count += count
    if total_count == 0:
        raise ValueError("epoch has no finite targets")
    return total_loss / total_count

def train_model(model: nn.Module, train_loader: DataLoader, validation_loader: DataLoader, config: TrainingConfig, *, loss_fn: nn.Module | None = None, checkpoint_path: str | Path | None = None, checkpoint_metadata: dict[str, Any] | None = None,) -> TrainingResult:
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
    total_steps = len(train_loader) * config.epochs
    scheduler, is_step_scheduler = _build_scheduler(optimizer, config, total_steps)
    scaler = torch.amp.GradScaler(device.type, enabled=config.amp and device.type == "cuda")
    history: list[dict[str, float]] = []
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, config.epochs + 1):
        train_loss = _epoch(
            model, train_loader, criterion, device, config, optimizer,
            scheduler=scheduler, is_step_scheduler=is_step_scheduler, scaler=scaler,
        )
        validation_loss = _epoch(
            model, validation_loader, criterion, device, config, None
        )
        if not is_step_scheduler and scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(validation_loss)
            else:
                scheduler.step()
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

def train_ensemble(model_factory: Callable[[], nn.Module], train_loader: DataLoader, validation_loader: DataLoader, config: TrainingConfig, *, checkpoint_dir: str | Path | None = None,) -> tuple[EnsembleModel, list[TrainingResult]]:
    models: list[nn.Module] = []
    results: list[TrainingResult] = []
    ckpt_dir = Path(checkpoint_dir or config.checkpoint_dir).expanduser()
    for idx in range(config.ensemble_size):
        member_seed = config.seed + idx * 1000
        member_config = TrainingConfig(
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            seed=member_seed,
            device=config.device,
            num_workers=config.num_workers,
            loss=config.loss,
            patience=config.patience,
            checkpoint_dir=config.checkpoint_dir,
            scheduler=config.scheduler,
            warmup_epochs=config.warmup_epochs,
            grad_clip=config.grad_clip,
            amp=config.amp,
            ensemble_size=1,
            evidential_lambda=config.evidential_lambda,
            multitask_weighting=config.multitask_weighting,
        )
        member_model = model_factory()
        ckpt_path = ckpt_dir / f"ensemble_member_{idx}.pt" if checkpoint_dir else None
        res = train_model(
            member_model,
            train_loader,
            validation_loader,
            member_config,
            checkpoint_path=ckpt_path,
        )
        models.append(res.model)
        results.append(res)
    ensemble = EnsembleModel(models)
    return ensemble, results

__all__ = [
    "TrainingResult",
    "train_ensemble",
    "train_model",
]
