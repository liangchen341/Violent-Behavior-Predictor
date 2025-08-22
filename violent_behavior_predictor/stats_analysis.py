from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact, kstest, ttest_ind, levene
import statsmodels.api as sm

from .constants import ENGLISH_LABELS, STATIC_FEATURES


@dataclass
class UnivariateResult:
    feature: str
    feature_english: str
    feature_type: str
    group_high: str
    group_low: str
    test_method: str
    p_value: float
    significant: bool


def _is_normally_distributed(series: pd.Series) -> bool:
    """Normality check using Kolmogorov–Smirnov test on standardized values."""
    s = series.dropna().astype(float)
    if len(s) < 10:
        return False
    z = (s - s.mean()) / (s.std() if s.std() != 0 else 1.0)
    stat, p = kstest(z, 'norm')
    return bool(p >= 0.05)


def perform_univariate_analysis(train_df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Perform univariate tests (continuous/ordinal/binary) between groups.
    Target column must be 'high_risk_group' (0/1).
    """
    results = []

    high_risk_group = train_df[train_df["high_risk_group"] == 1]
    low_risk_group = train_df[train_df["high_risk_group"] == 0]

    for feature in features:
        if feature not in train_df.columns:
            continue

        if feature in STATIC_FEATURES["continuous"]:
            ftype = "continuous"
        elif feature in STATIC_FEATURES["binary"]:
            ftype = "binary"
        elif feature in STATIC_FEATURES["ordinal"]:
            ftype = "ordinal"
        else:
            ftype = "behavior_ordinal"

        if ftype == "continuous":
            hi = high_risk_group[feature].dropna().astype(float)
            lo = low_risk_group[feature].dropna().astype(float)
            try:
                hi_norm = _is_normally_distributed(hi)
                lo_norm = _is_normally_distributed(lo)
                if hi_norm and lo_norm:
                    _, p_levene = levene(hi, lo, center='median')
                    if p_levene >= 0.05:
                        _, p = ttest_ind(hi, lo, equal_var=True)
                        method = "independent t-test"
                    else:
                        _, p = mannwhitneyu(hi, lo, alternative="two-sided")
                        method = "Mann-Whitney U"
                else:
                    _, p = mannwhitneyu(hi, lo, alternative="two-sided")
                    method = "Mann-Whitney U"
            except Exception:
                p = 1.0
                method = "test failed"
            hi_mean, hi_std = float(np.mean(hi)) if len(hi) else 0.0, float(np.std(hi)) if len(hi) else 0.0
            lo_mean, lo_std = float(np.mean(lo)) if len(lo) else 0.0, float(np.std(lo)) if len(lo) else 0.0
            group_high = f"{hi_mean:.2f}±{hi_std:.2f}"
            group_low = f"{lo_mean:.2f}±{lo_std:.2f}"

        elif ftype == "binary":
            ct = pd.crosstab(train_df[feature], train_df["high_risk_group"])  # not always 2x2
            try:
                expected = chi2_contingency(ct)[3]
                min_exp = expected.min() if expected.size > 0 else 0
                if min_exp >= 5 and ct.shape == (2, 2):
                    _, p, _, _ = chi2_contingency(ct)
                    method = "Chi-square"
                elif ct.shape == (2, 2):
                    _, p = fisher_exact(ct)
                    method = "Fisher exact"
                else:
                    _, p, _, _ = chi2_contingency(ct)
                    method = "Chi-square"
            except Exception:
                p = 1.0
                method = "test failed"
            hi_counts = ct[1] if 1 in ct.columns else pd.Series([0])
            lo_counts = ct[0] if 0 in ct.columns else pd.Series([0])
            hi_percent = (hi_counts / hi_counts.sum() * 100) if hi_counts.sum() > 0 else pd.Series([0])
            lo_percent = (lo_counts / lo_counts.sum() * 100) if lo_counts.sum() > 0 else pd.Series([0])
            group_high = f"{int(hi_counts.get(1, 0))}({float(hi_percent.get(1, 0)):.1f}%)"
            group_low = f"{int(lo_counts.get(1, 0))}({float(lo_percent.get(1, 0)):.1f}%)"

        else:  # ordinal/behavioral
            hi = high_risk_group[feature].dropna()
            lo = low_risk_group[feature].dropna()
            try:
                if len(hi) and len(lo):
                    _, p = mannwhitneyu(hi, lo, alternative="two-sided")
                    method = "Mann-Whitney U"
                else:
                    p = 1.0
                    method = "insufficient data for Mann-Whitney U"
            except Exception:
                try:
                    ct = pd.crosstab(train_df[feature], train_df["high_risk_group"])
                    _, p, _, _ = chi2_contingency(ct)
                    method = "Chi-square (fallback)"
                except Exception:
                    p = 1.0
                    method = "test failed"
            hi_mean, hi_std = float(np.mean(hi)) if len(hi) else 0.0, float(np.std(hi)) if len(hi) else 0.0
            lo_mean, lo_std = float(np.mean(lo)) if len(lo) else 0.0, float(np.std(lo)) if len(lo) else 0.0
            group_high = f"{hi_mean:.2f}±{hi_std:.2f}"
            group_low = f"{lo_mean:.2f}±{lo_std:.2f}"

        results.append(
            UnivariateResult(
                feature=feature,
                feature_english=ENGLISH_LABELS.get(feature, feature),
                feature_type=ftype,
                group_high=group_high,
                group_low=group_low,
                test_method=method,
                p_value=float(p),
                significant=bool(p < 0.05),
            ).__dict__
        )

    df_res = pd.DataFrame(results)
    if not df_res.empty:
        df_res = df_res.sort_values("p_value")
    return df_res


def perform_univariate_logistic(train_df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Fit univariate logistic regression for each feature; return OR, 95% CI and p-value."""
    rows: List[dict] = []
    y = train_df["high_risk_group"].astype(int)
    for feature in features:
        if feature not in train_df.columns:
            continue
        x = train_df[[feature]].copy()
        if x[feature].nunique(dropna=True) < 2:
            continue
        X_sm = sm.add_constant(x)
        try:
            model = sm.Logit(y, X_sm)
            res = model.fit(method="newton", maxiter=100, disp=False)
            coef = float(res.params.get(feature, float("nan")))
            pval = float(res.pvalues.get(feature, float("nan")))
            or_val = float(np.exp(coef)) if np.isfinite(coef) else float("nan")
            ci = res.conf_int().loc[feature] if feature in res.params.index else None
            if ci is not None:
                ci_low = float(np.exp(ci[0]))
                ci_high = float(np.exp(ci[1]))
            else:
                ci_low = float("nan"); ci_high = float("nan")
            rows.append({
                "feature": feature,
                "OR": or_val,
                "CI_low": ci_low,
                "CI_high": ci_high,
                "p_value": pval,
                "significant": bool(pval < 0.05),
            })
        except Exception:
            continue
    df_uni = pd.DataFrame(rows)
    if not df_uni.empty:
        df_uni = df_uni.sort_values("p_value")
    return df_uni


