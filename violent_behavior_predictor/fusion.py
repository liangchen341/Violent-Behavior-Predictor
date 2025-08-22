from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


def fuse_probabilities(static_prob: np.ndarray, behavior_prob: np.ndarray, weight_static: float) -> np.ndarray:
    """Linear late fusion: w*static + (1-w)*behavior, clipped to [0,1]."""
    weight_static = float(np.clip(weight_static, 0.0, 1.0))
    weight_behavior = 1.0 - weight_static
    return np.clip(weight_static * static_prob + weight_behavior * behavior_prob, 0.0, 1.0)


def grid_search_weight(
    y_true: np.ndarray, static_prob: np.ndarray, behavior_prob: np.ndarray, weights: Iterable[float]
) -> Tuple[float, float]:
    """Find fusion weight that maximizes AUC on validation targets."""
    from sklearn.metrics import roc_auc_score

    best_w, best_auc = 0.5, -1.0
    for w in weights:
        fused = fuse_probabilities(static_prob, behavior_prob, w)
        auc = float(roc_auc_score(y_true, fused))
        if auc > best_auc:
            best_auc = auc
            best_w = float(w)
    return best_w, best_auc


