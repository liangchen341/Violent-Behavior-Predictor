from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class LogitResult:
    model_name: str
    params: Dict[str, float]
    pvalues: Dict[str, float]
    bse: Dict[str, float]
    conf_int: Dict[str, Tuple[float, float]]
    summary_text: str
    feature_names: List[str]
    y_pred: np.ndarray
    y_prob: np.ndarray


def fit_logit(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, model_name: str,
              maxiter: int = 100) -> Optional[LogitResult]:
    """Fit a multivariable logistic regression (statsmodels) and return predictions/summary.

    Returns coefficients, p-values, confidence intervals and text summary for reporting.
    """
    if X_train.empty or X_test.empty:
        return None

    X_train_sm = sm.add_constant(X_train, has_constant="add")
    X_test_sm = sm.add_constant(X_test, has_constant="add")
    feat_names = list(X_train_sm.columns)

    try:
        logit_model = sm.Logit(y_train, X_train_sm)
        logit_result = logit_model.fit(method="newton", maxiter=maxiter, disp=False)
    except Exception:
        return None

    y_prob = logit_result.predict(X_test_sm)
    y_pred = (y_prob >= 0.5).astype(int)

    params = {k: float(v) for k, v in logit_result.params.items()}
    pvalues = {k: float(v) for k, v in logit_result.pvalues.items()}
    bse = {k: float(v) for k, v in logit_result.bse.items()}
    ci_df = logit_result.conf_int()
    conf_int = {idx: (float(row[0]), float(row[1])) for idx, row in ci_df.iterrows()}
    summary_text = str(logit_result.summary())

    return LogitResult(
        model_name=model_name,
        params=params,
        pvalues=pvalues,
        bse=bse,
        conf_int=conf_int,
        summary_text=summary_text,
        feature_names=feat_names,
        y_pred=y_pred,
        y_prob=y_prob,
    )


