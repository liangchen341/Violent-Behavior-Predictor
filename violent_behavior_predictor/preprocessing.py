from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np

from .constants import (
    STATIC_FEATURES,
    BEHAVIOR_FEATURES,
    TARGET_COLUMN,
    BINARY_MAPPINGS,
    EDUCATION_MAPPING,
)


@dataclass
class DatasetSplits:
    X_static: pd.DataFrame
    X_behavior: pd.DataFrame
    X_all: pd.DataFrame
    y: pd.Series


def extract_years(value) -> Optional[float]:
    """Extract numeric year value from mixed strings, fallback to None."""
    if pd.isna(value):
        return None
    match = re.search(r"(\d+\.?\d*)", str(value))
    if match:
        try:
            return float(match.group(1))
        except Exception:
            return None
    return None


def encode_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features and parse duration columns (English columns)."""
    encoded_df = df.copy()

    # Binary categorical mappings (to 0/1)
    for col, mapping in BINARY_MAPPINGS.items():
        if col in encoded_df.columns:
            encoded_df[col] = encoded_df[col].map(lambda x: mapping.get(x, x) if pd.notna(x) else x)

    # Education level (ordinal mapping)
    if "education_level" in encoded_df.columns:
        encoded_df["education_level"] = encoded_df["education_level"].map(
            lambda x: EDUCATION_MAPPING.get(x, x) if pd.notna(x) else x
        )

    # Disease duration (years)
    if "disease_duration_years" in encoded_df.columns:
        encoded_df["disease_duration_years"] = encoded_df["disease_duration_years"].apply(extract_years)

    return encoded_df


def select_feature_frames(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Split dataframe into static, behavior, all features and target (English columns)."""
    all_static: List[str] = (
        STATIC_FEATURES["continuous"]
        + STATIC_FEATURES["binary"]
        + STATIC_FEATURES["ordinal"]
    )

    present_static = [c for c in all_static if c in df.columns]
    present_behavior = [c for c in BEHAVIOR_FEATURES if c in df.columns]

    X_static = df[present_static]
    X_behavior = df[present_behavior]
    X_all = df[present_static + present_behavior]
    y = df[TARGET_COLUMN]

    return X_static, X_behavior, X_all, y


def dropna_by_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Drop rows with NA in the specified columns only, minimizing sample loss."""
    return df.dropna(subset=columns)



def zscore_normalize_continuous(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Z-score scale continuous and behavioral (ordinal 0-3) features using train stats only.

    Binary and ordinal static categorical features (e.g., education_level) are not scaled.
    Returns scaled X_train, X_test and the list of columns that were scaled.
    """
    continuous_candidates: List[str] = (
        STATIC_FEATURES.get("continuous", []) + BEHAVIOR_FEATURES
    )

    # Scale only columns that are present
    scale_cols: List[str] = [c for c in continuous_candidates if c in X_train.columns]

    X_train_norm = X_train.copy()
    X_test_norm = X_test.copy()

    if len(scale_cols) == 0:
        return X_train_norm, X_test_norm, scale_cols

    # Fit on train mean/std
    mu = X_train_norm[scale_cols].mean(axis=0)
    sd = X_train_norm[scale_cols].std(axis=0)

    # Avoid divide-by-zero
    sd_repaired = sd.replace(0, 1.0)
    sd_repaired = sd_repaired.mask(~np.isfinite(sd_repaired), 1.0)

    # Apply to train and test
    X_train_norm[scale_cols] = (X_train_norm[scale_cols] - mu) / sd_repaired
    common_cols = [c for c in scale_cols if c in X_test_norm.columns]
    X_test_norm[common_cols] = (X_test_norm[common_cols] - mu[common_cols]) / sd_repaired[common_cols]

    return X_train_norm, X_test_norm, scale_cols


