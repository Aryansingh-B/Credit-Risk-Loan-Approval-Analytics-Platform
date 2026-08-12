"""
train.py
--------
Credora Finance | Credit Risk & Loan Approval Analytics Platform

Trains and compares four classifiers on the disbursed-loan population,
selects a champion on ROC-AUC + recall, and persists the model artefact
plus a feature-importance table and metrics report.

Usage:
    python src/train.py
    python src/train.py --db data/credit_risk.db --models-dir models
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from features import build_feature_table, get_model_ready

RANDOM_STATE = 42


def evaluate(name, model, X_test, y_test, needs_scaling=False, scaler=None):
    X_eval = scaler.transform(X_test) if needs_scaling else X_test
    proba = model.predict_proba(X_eval)[:, 1]
    pred = (proba >= 0.5).astype(int)

    return {
        "model": name,
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the loan-default classifiers.")
    parser.add_argument("--db", default="data/credit_risk.db")
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and engineering features...")
    raw = build_feature_table(args.db)
    model_df = get_model_ready(raw)

    y = model_df["loan_default"].astype(int)
    X = model_df.drop(columns=["loan_default"])
    feature_names = X.columns.tolist()

    # 80/20 stratified split -- touched once, at the end (PRD 13.4)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows | "
          f"Train default rate: {y_train.mean():.1%} | Test default rate: {y_test.mean():.1%}")

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    candidates = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
            True,
        ),
        "Decision Tree": (
            DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=RANDOM_STATE),
            False,
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=300, max_depth=10, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1,
            ),
            False,
        ),
        "XGBoost": (
            XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
                eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
            ),
            False,
        ),
    }

    results = []
    fitted_models = {}
    print("\nTraining and 5-fold cross-validating each candidate...")
    for name, (model, needs_scaling) in candidates.items():
        fit_X = X_train_scaled if needs_scaling else X_train
        cv_scores = []
        for tr_idx, val_idx in cv.split(fit_X, y_train):
            fold_X_tr = fit_X[tr_idx] if needs_scaling else fit_X.iloc[tr_idx]
            fold_X_val = fit_X[val_idx] if needs_scaling else fit_X.iloc[val_idx]
            fold_model = type(model)(**model.get_params())
            fold_model.fit(fold_X_tr, y_train.iloc[tr_idx])
            proba = fold_model.predict_proba(fold_X_val)[:, 1]
            cv_scores.append(roc_auc_score(y_train.iloc[val_idx], proba))

        model.fit(fit_X, y_train)
        fitted_models[name] = (model, needs_scaling)

        metrics = evaluate(name, model, X_test, y_test, needs_scaling, scaler)
        metrics["cv_roc_auc_mean"] = round(float(np.mean(cv_scores)), 4)
        metrics["cv_roc_auc_std"] = round(float(np.std(cv_scores)), 4)
        results.append(metrics)
        print(f"  {name:<22} ROC-AUC(test)={metrics['roc_auc']:.3f}  "
              f"CV ROC-AUC={metrics['cv_roc_auc_mean']:.3f}±{metrics['cv_roc_auc_std']:.3f}  "
              f"Recall={metrics['recall']:.3f}")

    # Champion selection: primary = ROC-AUC, tiebreak = recall (PRD 13.5)
    results_sorted = sorted(results, key=lambda r: (r["roc_auc"], r["recall"]), reverse=True)
    champion_name = results_sorted[0]["model"]
    champion_model, champion_needs_scaling = fitted_models[champion_name]
    print(f"\nChampion model: {champion_name} "
          f"(ROC-AUC={results_sorted[0]['roc_auc']:.3f}, Recall={results_sorted[0]['recall']:.3f})")

    # Feature importance (tree-based) or coefficients (linear)
    if hasattr(champion_model, "feature_importances_"):
        importance = pd.Series(champion_model.feature_importances_, index=feature_names)
    else:
        importance = pd.Series(np.abs(champion_model.coef_[0]), index=feature_names)
    importance = importance.sort_values(ascending=False)
    importance.to_csv(models_dir / "feature_importance.csv", header=["importance"])

    # Persist champion model + preprocessing artefacts
    with open(models_dir / "default_model.pkl", "wb") as f:
        pickle.dump({
            "model": champion_model,
            "model_name": champion_name,
            "needs_scaling": champion_needs_scaling,
            "scaler": scaler if champion_needs_scaling else None,
            "feature_names": feature_names,
            "random_state": RANDOM_STATE,
        }, f)

    with open(models_dir / "metrics_report.json", "w") as f:
        json.dump({
            "champion_model": champion_name,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "test_default_rate": round(float(y_test.mean()), 4),
            "all_candidates": results,
        }, f, indent=2)

    print(f"\nSaved: {models_dir / 'default_model.pkl'}")
    print(f"Saved: {models_dir / 'feature_importance.csv'}")
    print(f"Saved: {models_dir / 'metrics_report.json'}")
    print(f"\nTop 10 feature importances ({champion_name}):")
    print(importance.head(10).to_string())


if __name__ == "__main__":
    main()
