from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)


@dataclass
class EvalResult:
    auc: float
    precision: float
    recall: float
    f1: float
    cm: np.ndarray
    report: Dict


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> EvalResult:
    """Compute AUC and classification metrics at a fixed threshold (default 0.5)."""
    y_pred = (y_prob >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    model_auc = float(auc(fpr, tpr))
    precision = float(precision_score(y_true, y_pred))
    recall = float(recall_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    return EvalResult(
        auc=model_auc,
        precision=precision,
        recall=recall,
        f1=f1,
        cm=cm,
        report=report,
    )


def calibrate_scores(prob: np.ndarray, target_mean: float = 0.5, target_std: float = 0.15) -> np.ndarray:
    """Simple distribution calibration to target mean/std, clipped to [0,1]."""
    mu = float(np.mean(prob))
    sd = float(np.std(prob))
    if sd == 0:
        return np.clip(np.full_like(prob, target_mean), 0.0, 1.0)
    calibrated = (prob - mu) / sd * target_std + target_mean
    return np.clip(calibrated, 0.0, 1.0)


