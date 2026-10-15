from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from numpy.typing import ArrayLike
from .additive import LinearAdditivePrior

@dataclass(frozen=True)
class PriorFit:
    prior: LinearAdditivePrior
    feature_names: tuple[str, ...]
    train_count: int
    residual_std: float

def fit_linear_prior(features: ArrayLike, targets: ArrayLike, feature_names: Sequence[str], *, fixed: dict[str, float] | None = None, method: str = "ridge", ridge_alpha: float = 1e-3,) -> PriorFit:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    names = tuple(feature_names)
    fixed = fixed or {}
    if y.ndim == 1:
        y = y[:, None]
    if (
        x.ndim != 2
        or y.ndim != 2
        or y.shape[0] != x.shape[0]
        or x.shape[1] != len(names)
    ):
        raise ValueError("features, targets, and feature_names have incompatible shapes")
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique")
    if method not in {"ridge", "least_squares", "logistic", "none"}:
        raise ValueError(f"unsupported prior fit method {method!r}")
    if ridge_alpha < 0:
        raise ValueError("ridge_alpha must be non-negative")
    unknown = set(fixed) - set(names)
    if unknown:
        raise ValueError(f"fixed coefficients reference unknown columns: {sorted(unknown)}")
    valid = np.isfinite(y).all(axis=1) & np.isfinite(x).all(axis=1)
    x, y = x[valid], y[valid]
    if len(y) < 2:
        raise ValueError("at least two complete finite training targets are required to fit a prior")
    weights = np.zeros((x.shape[1], y.shape[1]), dtype=np.float64)
    fixed_mask = np.zeros(x.shape[1], dtype=bool)
    for column, name in enumerate(names):
        if name in fixed:
            weights[column, :] = float(fixed[name])
            fixed_mask[column] = True
    offset = (
        x[:, fixed_mask] @ weights[fixed_mask]
        if fixed_mask.any()
        else np.zeros_like(y)
    )
    free_mask = ~fixed_mask
    residual_target = y - offset
    if method == "logistic":
        from sklearn.linear_model import LogisticRegression
        c_val = 1.0 / (ridge_alpha * len(y)) if ridge_alpha > 0 else 1.0
        clf = LogisticRegression(C=c_val, random_state=42, max_iter=1000)
        if free_mask.any():
            x_free = x[:, free_mask]
            clf.fit(x_free, y.ravel())
            weights[free_mask, 0] = clf.coef_[0]
            intercept = np.array([clf.intercept_[0]], dtype=np.float64)
        else:
            intercept = np.array([0.0], dtype=np.float64)
    elif method == "none":
        if free_mask.any():
            raise ValueError("fit='none' requires every selected prior feature coefficient to be fixed")
        intercept = residual_target.mean(axis=0)
    elif free_mask.any():
        x_free = x[:, free_mask]
        means = x_free.mean(axis=0)
        scales = x_free.std(axis=0)
        active = scales > 1e-12
        free_weights = np.zeros((x_free.shape[1], y.shape[1]), dtype=np.float64)
        if active.any():
            standardized = (x_free[:, active] - means[active]) / scales[active]
            if method == "ridge":
                gram = standardized.T @ standardized
                gram += ridge_alpha * len(y) * np.eye(int(active.sum()))
                standardized_weights = np.linalg.solve(
                    gram,
                    standardized.T @ (residual_target - residual_target.mean(axis=0)),
                )
            else:
                standardized_weights = np.linalg.lstsq(
                    standardized,
                    residual_target - residual_target.mean(axis=0),
                    rcond=None,
                )[0]
            free_weights[active] = standardized_weights / scales[active, None]
        weights[free_mask, :] = free_weights
        intercept = residual_target.mean(axis=0) - means @ free_weights
    else:
        intercept = residual_target.mean(axis=0)
    prediction = x @ weights + intercept
    if method == "logistic":
        probs = 1.0 / (1.0 + np.exp(-np.clip(prediction, -20, 20)))
        residual_std = float(np.std(y - probs))
    else:
        residual_std = float(np.std(y - prediction))
    prior = LinearAdditivePrior(
        weights.tolist(),
        intercept=intercept.tolist(),
        feature_names=names,
    )
    return PriorFit(prior, names, int(len(y)), residual_std)
