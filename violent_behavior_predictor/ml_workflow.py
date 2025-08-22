from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from .constants import TARGET_COLUMN


BalanceMethod = str | None


def balance_fit_resample(X: pd.DataFrame, y: pd.Series, method: BalanceMethod, random_state: int) -> Tuple[pd.DataFrame, pd.Series]:
    if method is None:
        return X, y
    if method == "under":
        rus = RandomUnderSampler(random_state=random_state)
        return rus.fit_resample(X, y)
    if method == "over":
        ros = RandomOverSampler(random_state=random_state)
        return ros.fit_resample(X, y)
    if method == "smote":
        sm = SMOTE(random_state=random_state)
        return sm.fit_resample(X, y)
    return X, y


def get_model_spaces(random_state: int) -> Dict[str, Tuple[Pipeline, Dict[str, List]]]:
    spaces: Dict[str, Tuple[Pipeline, Dict[str, List]]] = {}

    spaces["logreg"] = (
        Pipeline([
            ("clf", LogisticRegression(max_iter=200, solver="liblinear", random_state=random_state)),
        ]),
        {
            "clf__C": [0.1, 1.0, 10.0],
            "clf__penalty": ["l1", "l2"],
        },
    )

    # SVM with RBF kernel; tune C and gamma
    spaces["svm"] = (
        Pipeline([
            ("clf", SVC(probability=True, random_state=random_state)),
        ]),
        {
            "clf__kernel": ["rbf"],
            "clf__C": [0.1, 1.0, 10.0, 100.0],
            "clf__gamma": [1e-3, 1e-2, 1e-1, "scale"],
        },
    )

    # KNN: tune k, weights and distance metric (p=1 Manhattan; p=2 Euclidean)
    spaces["knn"] = (
        Pipeline([
            ("clf", KNeighborsClassifier()),
        ]),
        {
            "clf__n_neighbors": [3, 5, 7, 9, 15, 20],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2],
        },
    )

    spaces["tree"] = (
        Pipeline([
            ("clf", DecisionTreeClassifier(random_state=random_state)),
        ]),
        {
            "clf__max_depth": [None, 3, 5, 7, 9],
            "clf__min_samples_split": [2, 5, 10],
        },
    )

    # Random Forest: n_estimators, max_depth, max_features, min_samples
    spaces["rf"] = (
        Pipeline([
            ("clf", RandomForestClassifier(random_state=random_state)),
        ]),
        {
            "clf__n_estimators": [200, 400, 800],
            "clf__max_depth": [None, 5, 10, 20],
            "clf__max_features": ["sqrt", "log2", None],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
        },
    )

    # MLP: hidden sizes, activation, L2 regularization, learning rate
    spaces["mlp"] = (
        Pipeline([
            ("clf", MLPClassifier(max_iter=500, random_state=random_state)),
        ]),
        {
            "clf__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
            "clf__activation": ["relu", "tanh"],
            "clf__alpha": [1e-4, 1e-3, 1e-2],
            "clf__learning_rate_init": [1e-3, 1e-2],
        },
    )

    spaces["gbdt"] = (
        Pipeline([
            ("clf", GradientBoostingClassifier(random_state=random_state)),
        ]),
        {
            "clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth": [3, 5],
        },
    )

    # Optional XGBoost (if installed)
    try:
        from xgboost import XGBClassifier  # type: ignore

        # XGBoost: depth, min_child_weight, gamma, subsample, colsample, lr, regularization
        spaces["xgb"] = (
            Pipeline([
                ("clf", XGBClassifier(
                    random_state=random_state,
                    eval_metric="logloss",
                    use_label_encoder=False,
                )),
            ]),
            {
                "clf__n_estimators": [200, 400, 600],
                "clf__max_depth": [3, 5, 7],
                "clf__min_child_weight": [1, 3, 5],
                "clf__gamma": [0, 0.1, 0.2],
                "clf__learning_rate": [0.03, 0.05, 0.1],
                "clf__subsample": [0.8, 1.0],
                "clf__colsample_bytree": [0.8, 1.0],
                "clf__reg_alpha": [0, 0.01, 0.1],
                "clf__reg_lambda": [1, 5, 10],
            },
        )
    except Exception:
        pass

    return spaces


def train_with_grid_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    random_state: int = 42,
) -> Dict[str, Dict[str, np.ndarray]]:
    from sklearn.metrics import roc_auc_score

    spaces = get_model_spaces(random_state)
    results: Dict[str, Dict[str, np.ndarray]] = {}

    for key, (pipe, grid) in spaces.items():
        gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=5, n_jobs=-1, refit=True)
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        prob = best.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_train, best.predict_proba(X_train)[:, 1]))
        results[key] = {
            "estimator": best,
            "y_prob": prob,
            "cv_best_params": gs.best_params_,
            "train_auc": auc,
        }

    return results


def evaluate_model_suite(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
) -> pd.DataFrame:
    from .evaluation import evaluate_predictions
    spaces_results = train_with_grid_search(X_train, y_train, X_test, random_state)
    rows = []
    for model_key, res in spaces_results.items():
        y_prob = res["y_prob"]
        ev = evaluate_predictions(y_test.values, y_prob)
        rows.append({
            "model": model_key,
            "auc": ev.auc,
            "precision": ev.precision,
            "recall": ev.recall,
            "f1": ev.f1,
            "best_params": res.get("cv_best_params", {}),
        })
    return pd.DataFrame(rows).sort_values("auc", ascending=False)


