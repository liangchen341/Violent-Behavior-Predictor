from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GridSearchCV, cross_val_predict, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import numpy as np

from violent_behavior_predictor.preprocessing import encode_dataframe, select_feature_frames, dropna_by_columns, zscore_normalize_continuous
from violent_behavior_predictor.constants import TARGET_COLUMN, BEHAVIOR_FEATURES, STATIC_FEATURES
from violent_behavior_predictor.stats_analysis import perform_univariate_analysis, perform_univariate_logistic
from violent_behavior_predictor.logit_model import fit_logit
from violent_behavior_predictor.evaluation import evaluate_predictions, calibrate_scores
from violent_behavior_predictor.fusion import fuse_probabilities, grid_search_weight
from violent_behavior_predictor.ml_workflow import evaluate_model_suite, get_model_spaces


def main() -> None:
    parser = argparse.ArgumentParser(description="Violent Behavior Predictor CLI")
    parser.add_argument("--input", default="merged_patient_data.csv", help="Input CSV file with English columns")
    parser.add_argument("--out", default="hierarchical_model_results", help="Output directory")
    parser.add_argument("--test_size", type=float, default=0.3, help="Test set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--balance", choices=["under", "over", "smote"], default=None, help="Class balancing method for training set")
    parser.add_argument("--weight_static", type=float, default=0.5, help="Fusion weight for static model (0-1)")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate output probabilities (to mean=0.5, std=0.15)")
    parser.add_argument("--ml_suite", action="store_true", help="Run additional ML baselines and save metrics")
    args = parser.parse_args()

    out_dir = Path(args.out if args.balance is None else f"{args.out}_{args.balance}")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    df = encode_dataframe(df)

    # Drop NA only on used columns to minimize sample loss
    all_static = STATIC_FEATURES["continuous"] + STATIC_FEATURES["binary"] + STATIC_FEATURES["ordinal"]
    use_cols = [c for c in all_static if c in df.columns] + [c for c in BEHAVIOR_FEATURES if c in df.columns] + [TARGET_COLUMN]
    df = dropna_by_columns(df, use_cols)

    X_static, X_behavior, X_all, y = select_feature_frames(df)

    # Group split by patient id if available to avoid leakage across sets
    if 'patient_id' in df.columns:
        groups = df['patient_id'].values
        gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        train_idx, test_idx = next(gss.split(X_all, y, groups=groups))
        X_train, X_test = X_all.iloc[train_idx], X_all.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y, test_size=args.test_size, random_state=args.seed, stratify=y
        )

    # Z-score normalization based on train statistics for continuous/behavioral features
    X_train, X_test, _scaled_cols = zscore_normalize_continuous(X_train, X_test)

    # Align static and behavioral subsets (on scaled data)
    X_train_static = X_train[X_static.columns]
    X_train_behavior = X_train[X_behavior.columns]
    X_test_static = X_test[X_static.columns]
    X_test_behavior = X_test[X_behavior.columns]

    # Keep full feature copies for ML baselines
    ml_X_train_static_full = X_train_static.copy()
    ml_X_test_static_full = X_test_static.copy()
    ml_X_train_behavior_full = X_train_behavior.copy()
    ml_X_test_behavior_full = X_test_behavior.copy()

    # Univariate analysis on the training set
    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train.values
    uni_static = perform_univariate_analysis(train_df, X_train_static.columns.tolist())
    uni_behavior = perform_univariate_analysis(train_df, X_train_behavior.columns.tolist())
    uni_static.to_csv(out_dir / "univariate_static_analysis.csv", index=False)
    uni_behavior.to_csv(out_dir / "univariate_behavior_analysis.csv", index=False)

    # Univariate logistic regression (per feature), output OR and 95% CI
    uni_logit_static = perform_univariate_logistic(train_df, X_train_static.columns.tolist())
    uni_logit_behavior = perform_univariate_logistic(train_df, X_train_behavior.columns.tolist())
    uni_logit_static.to_csv(out_dir / "univariate_static_logit.csv", index=False)
    uni_logit_behavior.to_csv(out_dir / "univariate_behavior_logit.csv", index=False)

    # Feature preselection by univariate logit significance (p<0.05)
    sig_static = uni_logit_static.loc[uni_logit_static["significant"] == True, "feature"].tolist()
    sig_behavior = uni_logit_behavior.loc[uni_logit_behavior["significant"] == True, "feature"].tolist()
    use_static_cols = sig_static if len(sig_static) > 0 else X_train_static.columns.tolist()
    use_behavior_cols = sig_behavior if len(sig_behavior) > 0 else X_train_behavior.columns.tolist()

    X_train_static = X_train[use_static_cols]
    X_test_static = X_test[use_static_cols]
    X_train_behavior = X_train[use_behavior_cols]
    X_test_behavior = X_test[use_behavior_cols]

    # Multivariable logistic regression (statsmodels) on preselected features
    logit_static = fit_logit(X_train_static, y_train, X_test_static, model_name="static_logit")
    logit_behavior = fit_logit(X_train_behavior, y_train, X_test_behavior, model_name="behavior_logit")

    results = {}

    # Summarize metrics: sensitivity/specificity/PPV/NPV/accuracy/AUC/confusion matrix
    def summarize_eval(ev):
        rep = ev.report if isinstance(ev.report, dict) else {}
        # confusion matrix: [[tn, fp], [fn, tp]]
        cm = ev.cm
        tn = int(cm[0, 0]) if cm is not None and cm.size == 4 else 0
        fp = int(cm[0, 1]) if cm is not None and cm.size == 4 else 0
        fn = int(cm[1, 0]) if cm is not None and cm.size == 4 else 0
        tp = int(cm[1, 1]) if cm is not None and cm.size == 4 else 0
        # Derive specificity/NPV from classification_report and confusion matrix
        specificity = float(rep.get("0", {}).get("recall", (tn / (tn + fp) if (tn + fp) > 0 else 0.0)))
        npv = float(rep.get("0", {}).get("precision", (tn / (tn + fn) if (tn + fn) > 0 else 0.0)))
        accuracy = float(rep.get("accuracy", ((tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0)))
        return {
            "auc": float(ev.auc),
            "sensitivity": float(ev.recall),
            "specificity": specificity,
            "ppv": float(ev.precision),
            "npv": npv,
            "accuracy": accuracy,
            "f1": float(ev.f1),
            "confusion_matrix": [
                [tn, fp],
                [fn, tp],
            ],
        }
    if logit_static is not None:
        eval_static = evaluate_predictions(y_test.values, logit_static.y_prob)
        results["static_logit"] = {
            "num_features": len(use_static_cols),
            **summarize_eval(eval_static),
        }
        # Save multivariable regression details
        keys = list(logit_static.params.keys())
        # Compute OR and 95% CI (coef and exponentiated intervals)
        coef_vals = [logit_static.params[k] for k in keys]
        ci_low_coef = [logit_static.conf_int.get(k, (float("nan"), float("nan")))[0] for k in keys]
        ci_high_coef = [logit_static.conf_int.get(k, (float("nan"), float("nan")))[1] for k in keys]
        df_coef = pd.DataFrame({
            "term": keys,
            "coef": coef_vals,
            "std_err": [logit_static.bse.get(k, float("nan")) for k in keys],
            "p_value": [logit_static.pvalues.get(k, float("nan")) for k in keys],
            "ci_low": ci_low_coef,
            "ci_high": ci_high_coef,
            "OR": [float(np.exp(v)) if np.isfinite(v) else float("nan") for v in coef_vals],
            "OR_CI_low": [float(np.exp(v)) if np.isfinite(v) else float("nan") for v in ci_low_coef],
            "OR_CI_high": [float(np.exp(v)) if np.isfinite(v) else float("nan") for v in ci_high_coef],
        })
        df_coef.to_csv(out_dir / "static_logit_coefficients.csv", index=False, encoding="utf-8")
        with open(out_dir / "static_logit_summary.txt", "w", encoding="utf-8") as f:
            f.write(logit_static.summary_text)
    if logit_behavior is not None:
        eval_behavior = evaluate_predictions(y_test.values, logit_behavior.y_prob)
        results["behavior_logit"] = {
            "num_features": len(use_behavior_cols),
            **summarize_eval(eval_behavior),
        }
        keys = list(logit_behavior.params.keys())
        coef_vals_b = [logit_behavior.params[k] for k in keys]
        ci_low_coef_b = [logit_behavior.conf_int.get(k, (float("nan"), float("nan")))[0] for k in keys]
        ci_high_coef_b = [logit_behavior.conf_int.get(k, (float("nan"), float("nan")))[1] for k in keys]
        df_coef = pd.DataFrame({
            "term": keys,
            "coef": coef_vals_b,
            "std_err": [logit_behavior.bse.get(k, float("nan")) for k in keys],
            "p_value": [logit_behavior.pvalues.get(k, float("nan")) for k in keys],
            "ci_low": ci_low_coef_b,
            "ci_high": ci_high_coef_b,
            "OR": [float(np.exp(v)) if np.isfinite(v) else float("nan") for v in coef_vals_b],
            "OR_CI_low": [float(np.exp(v)) if np.isfinite(v) else float("nan") for v in ci_low_coef_b],
            "OR_CI_high": [float(np.exp(v)) if np.isfinite(v) else float("nan") for v in ci_high_coef_b],
        })
        df_coef.to_csv(out_dir / "behavior_logit_coefficients.csv", index=False, encoding="utf-8")
        with open(out_dir / "behavior_logit_summary.txt", "w", encoding="utf-8") as f:
            f.write(logit_behavior.summary_text)

    # Secondary selection by multivariable logit significance (p<0.05, exclude const)
    final_static_cols = []
    final_behavior_cols = []
    if logit_static is not None:
        final_static_cols = [k for k, p in logit_static.pvalues.items() if k != 'const' and (p is not None) and (p < 0.05)]
    if logit_behavior is not None:
        final_behavior_cols = [k for k, p in logit_behavior.pvalues.items() if k != 'const' and (p is not None) and (p < 0.05)]

    # Fallback: if secondary selection is empty, use univariate significant lists
    if len(final_static_cols) == 0:
        final_static_cols = use_static_cols
    if len(final_behavior_cols) == 0:
        final_behavior_cols = use_behavior_cols

    results["final_feature_counts"] = {
        "static": len(final_static_cols),
        "behavior": len(final_behavior_cols),
    }

    # Regularized LR branch trained on independent predictors (separate from ML baselines)
    # Only run if both static and dynamic sides have available features
    if (len(final_static_cols) > 0) and (len(final_behavior_cols) > 0):
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression

        def build_regularized_lr(
            X_tr, y_tr, X_te,
        ):
            """Grid-search regularized LR (5-fold CV) and return best estimator and predictions."""
            pipe = Pipeline([
                ("clf", LogisticRegression(random_state=args.seed))
            ])

            # Solver/penalty combinations; C on log scale; elasticnet requires l1_ratio
            import numpy as np
            # additionally include 0.5 to match paper examples
            C_grid = list(np.logspace(-4, 4, 9)) + [0.5]
            C_grid = sorted(set(float(c) for c in C_grid))
            param_grid = [
                {  # l1: supported by liblinear/saga
                    "clf__penalty": ["l1"],
                    "clf__solver": ["liblinear", "saga"],
                    "clf__C": C_grid,
                    "clf__max_iter": [100, 200],
                },
                {  # l2: supported by lbfgs/liblinear/saga
                    "clf__penalty": ["l2"],
                    "clf__solver": ["lbfgs", "liblinear", "saga"],
                    "clf__C": C_grid,
                    "clf__max_iter": [100, 200],
                },
                {  # elasticnet: only saga supports
                    "clf__penalty": ["elasticnet"],
                    "clf__solver": ["saga"],
                    "clf__l1_ratio": [0.1, 0.5, 0.9],
                    "clf__C": C_grid,
                    "clf__max_iter": [100, 200],
                },
            ]

            gs = GridSearchCV(
                pipe,
                param_grid,
                scoring="roc_auc",
                cv=5,
                n_jobs=-1,
                refit=True,
            )
            gs.fit(X_tr, y_tr)

            best_est = gs.best_estimator_
            prob_test = best_est.predict_proba(X_te)[:, 1]
            best_cv_auc = float(gs.best_score_)
            best_params = gs.best_params_

            # Generate out-of-fold probabilities for alpha grid search
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
            oof_prob = cross_val_predict(
                best_est, X_tr, y_tr, cv=cv, method="predict_proba"
            )[:, 1]

            return best_est, oof_prob, prob_test, best_cv_auc, best_params

        # Build static/dynamic regularized LR submodels using significant multivariable predictors
        X_tr_s = X_train[final_static_cols]
        X_te_s = X_test[final_static_cols]
        X_tr_b = X_train[final_behavior_cols]
        X_te_b = X_test[final_behavior_cols]

        s_est_lr, oof_s_lr, prob_s_te_lr, s_cv_auc_lr, s_params_lr = build_regularized_lr(
            X_tr_s, y_train, X_te_s
        )
        b_est_lr, oof_b_lr, prob_b_te_lr, b_cv_auc_lr, b_params_lr = build_regularized_lr(
            X_tr_b, y_train, X_te_b
        )

        # Grid search alpha in [0,1] with step 0.05 using training OOF AUC
        weights = np.linspace(0, 1, 21)
        best_w_lr, best_auc_lr = grid_search_weight(y_train.values, oof_s_lr, oof_b_lr, weights)

        fused_lr = fuse_probabilities(prob_s_te_lr, prob_b_te_lr, best_w_lr)
        if args.calibrate:
            fused_lr = calibrate_scores(fused_lr)
        eval_fused_lr = evaluate_predictions(y_test.values, fused_lr)

        results["regularized_lr"] = {
            "static": {
                "num_features": len(final_static_cols),
                "cv_auc": float(s_cv_auc_lr),
                "best_params": s_params_lr,
            },
            "behavior": {
                "num_features": len(final_behavior_cols),
                "cv_auc": float(b_cv_auc_lr),
                "best_params": b_params_lr,
            },
            "fusion": {
                "weight_used": float(best_w_lr),
                "best_auc_on_val": float(best_auc_lr),
                **summarize_eval(eval_fused_lr),
            },
        }

    # ML baselines (MLP, RF, KNN, SVM, XGBoost): 5-fold CV select best per side, evaluate and fuse on test
    if (len(final_static_cols) > 0) and (len(final_behavior_cols) > 0):
        spaces = get_model_spaces(args.seed)
        wanted = {k for k in ["mlp", "rf", "knn", "svm", "xgb"] if k in spaces}

        def select_best_model(X_tr, y_tr, X_te):
            best_key = None
            best_est = None
            best_cv_auc = -1.0
            best_params = None
            best_prob_test = None
            for key in wanted:
                pipe, grid = spaces[key]
                gs = GridSearchCV(
                    pipe,
                    grid,
                    scoring="roc_auc",
                    cv=5,
                    n_jobs=-1,
                    refit=True,
                )
                gs.fit(X_tr, y_tr)
                this_cv_auc = float(gs.best_score_)
                if this_cv_auc > best_cv_auc:
                    best_cv_auc = this_cv_auc
                    best_key = key
                    best_est = gs.best_estimator_
                    best_params = gs.best_params_
                    best_prob_test = best_est.predict_proba(X_te)[:, 1]
            return best_key, best_est, best_cv_auc, best_params, best_prob_test

        # Per paper: static uses all 18 baseline features; dynamic uses all 39 behavioral features
        # Thus univariate significance is only for statistics; ML submodels use full feature sets
        s_key, s_est, s_cv_auc, s_params, prob_static_test = select_best_model(
            ml_X_train_static_full, y_train, ml_X_test_static_full
        )

        b_key, b_est, b_cv_auc, b_params, prob_behavior_test = select_best_model(
            ml_X_train_behavior_full, y_train, ml_X_test_behavior_full
        )

        # OOF-based alpha search in [0,1] step 0.05
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        oof_static = cross_val_predict(
            s_est, ml_X_train_static_full, y_train, cv=cv, method="predict_proba"
        )[:, 1]
        oof_behavior = cross_val_predict(
            b_est, ml_X_train_behavior_full, y_train, cv=cv, method="predict_proba"
        )[:, 1]
        weights = np.linspace(0, 1, 21)
        best_w, best_auc = grid_search_weight(y_train.values, oof_static, oof_behavior, weights)

        fused = fuse_probabilities(prob_static_test, prob_behavior_test, best_w)
        if args.calibrate:
            fused = calibrate_scores(fused)
        eval_fused = evaluate_predictions(y_test.values, fused)
        results["fusion"] = {
            "weight_used": float(best_w),
            **summarize_eval(eval_fused),
        }
        results["fusion_best_weight_train"] = {"best_weight": best_w, "best_auc_on_val": best_auc}

        # Record selected best static/dynamic models
        results["static_model_selected"] = {
            "model": s_key,
            "cv_auc": s_cv_auc,
            "best_params": s_params,
        }
        results["behavior_model_selected"] = {
            "model": b_key,
            "cv_auc": b_cv_auc,
            "best_params": b_params,
        }

        # Additionally: pair same algorithm type (static/dynamic), tune separately, OOF alpha, evaluate fusion
        fusion_pairs = {}
        for key in wanted:
            pipe_s, grid_s = spaces[key]
            pipe_b, grid_b = spaces[key]

            # Static: all 18 baseline features
            gs_s = GridSearchCV(pipe_s, grid_s, scoring="roc_auc", cv=5, n_jobs=-1, refit=True)
            gs_s.fit(ml_X_train_static_full, y_train)
            est_s = gs_s.best_estimator_
            prob_s_test = est_s.predict_proba(ml_X_test_static_full)[:, 1]

            # Dynamic: all 39 behavioral features
            gs_b = GridSearchCV(pipe_b, grid_b, scoring="roc_auc", cv=5, n_jobs=-1, refit=True)
            gs_b.fit(ml_X_train_behavior_full, y_train)
            est_b = gs_b.best_estimator_
            prob_b_test = est_b.predict_proba(ml_X_test_behavior_full)[:, 1]

            # OOF probabilities for alpha search
            cv_pair = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
            oof_s = cross_val_predict(est_s, ml_X_train_static_full, y_train, cv=cv_pair, method="predict_proba")[:, 1]
            oof_b = cross_val_predict(est_b, ml_X_train_behavior_full, y_train, cv=cv_pair, method="predict_proba")[:, 1]
            w_grid = np.linspace(0, 1, 21)
            best_w_pair, best_auc_pair = grid_search_weight(y_train.values, oof_s, oof_b, w_grid)

            fused_pair_test = fuse_probabilities(prob_s_test, prob_b_test, best_w_pair)
            if args.calibrate:
                fused_pair_test = calibrate_scores(fused_pair_test)
            ev_pair = evaluate_predictions(y_test.values, fused_pair_test)

            fusion_pairs[key] = {
                "static": {
                    "cv_auc": float(gs_s.best_score_),
                    "best_params": gs_s.best_params_,
                },
                "behavior": {
                    "cv_auc": float(gs_b.best_score_),
                    "best_params": gs_b.best_params_,
                },
                "fusion": {
                    "weight_used": float(best_w_pair),
                    "best_auc_on_val": float(best_auc_pair),
                    **summarize_eval(ev_pair),
                },
            }

        # Save full pair results and best pair summary
        results["fusion_pairs"] = fusion_pairs
        # Pick the pair with the largest AUC as best
        try:
            best_pair_key = max(fusion_pairs.keys(), key=lambda k: fusion_pairs[k]["fusion"]["auc"])
            results["fusion_pairs_best"] = {
                "model": best_pair_key,
                "details": fusion_pairs[best_pair_key],
            }
        except Exception:
            pass

    with open(out_dir / "results_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Optional: run ML suite metrics
    if args.ml_suite:
        # Use full features for ML evaluation
        ml_df_static = evaluate_model_suite(ml_X_train_static_full, y_train, ml_X_test_static_full, y_test, random_state=args.seed)
        ml_df_behavior = evaluate_model_suite(ml_X_train_behavior_full, y_train, ml_X_test_behavior_full, y_test, random_state=args.seed)
        ml_df_static.to_csv(out_dir / "ml_metrics_static.csv", index=False, encoding="utf-8")
        ml_df_behavior.to_csv(out_dir / "ml_metrics_behavior.csv", index=False, encoding="utf-8")

    print("Done. Output directory:", str(out_dir))


if __name__ == "__main__":
    main()


