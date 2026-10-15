"""Feature recommendation and selection routines."""
from __future__ import annotations

import math
from typing import Sequence
import numpy as np
from scipy import stats

def select_evidence_recommendation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: Sequence[str] | None = None,
    k: int = 50,
    redundancy_threshold: float = 0.85,
    min_variance: float = 1e-6,
) -> list[int]:
    """Select top-k features using training-set Spearman rank correlation and greedy redundancy pruning.

    Algorithm:
    1. Compute absolute Spearman rank correlation for every feature column against y_train on training fold.
    2. Sort features in descending order of correlation.
    3. Greedily select features if pairwise Pearson correlation |r| with all already selected features <= redundancy_threshold.
    4. If fewer than k features satisfy the threshold, pad with next highest scoring features.

    Returns:
        List of integer column indices selected.
    """
    X_tr = np.asarray(X_train, dtype=np.float64)
    y_tr = np.asarray(y_train, dtype=np.float64).ravel()
    n_features = X_tr.shape[1]

    scores: list[float] = []
    for j in range(n_features):
        col = X_tr[:, j]
        if np.std(col) < min_variance:
            scores.append(0.0)
            continue
        try:
            r, _ = stats.spearmanr(col, y_tr)
            scores.append(abs(r) if math.isfinite(r) else 0.0)
        except Exception:
            scores.append(0.0)

    ranked_cols = np.argsort(scores)[::-1]
    selected: list[int] = []

    for col in ranked_cols:
        if scores[col] < 1e-4:
            continue
        redundant = False
        for prev in selected:
            try:
                r_cross, _ = stats.pearsonr(X_tr[:, col], X_tr[:, prev])
                if abs(r_cross) > redundancy_threshold:
                    redundant = True
                    break
            except Exception:
                pass
        if not redundant:
            selected.append(int(col))
        if len(selected) >= k:
            break

    if len(selected) < k:
        for col in ranked_cols:
            c = int(col)
            if c not in selected:
                selected.append(c)
            if len(selected) >= k:
                break

    return selected[:k]

def select_evidence_recommendation_names(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: Sequence[str],
    k: int = 50,
    redundancy_threshold: float = 0.85,
) -> list[str]:
    """Select top-k feature names using training-set evidence recommendation."""
    indices = select_evidence_recommendation(
        X_train=X_train,
        y_train=y_train,
        feature_names=feature_names,
        k=k,
        redundancy_threshold=redundancy_threshold,
    )
    return [feature_names[i] for i in indices]
