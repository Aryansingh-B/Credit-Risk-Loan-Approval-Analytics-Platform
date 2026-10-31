from __future__ import annotations

import argparse
import json
import pickle
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from calibration import PlattCalibrator, assign_category
from explain import build_explainer, importance_table, shap_values, summary_plot
from features import DEFAULT_DB, ROOT, TARGET, build_feature_table, get_model_ready

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
RECALL_TARGET = 0.70   # PRD section 16: recall on defaulters
LOW_BAND_RECALL = 0.90  # Low Risk band may contain at most ~10% of true defaulters
AUC_TARGET = 0.80       # PRD section 16


# --------------------------------------------------------------------------
def candidates(pos_weight: float) -> dict:
    """name -> (pipeline, search space, n_iter). Scaling only where it matters (linear model)."""
    rs = RANDOM_STATE
    return {
        "Logistic Regression": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=rs))]),
            {"clf__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]}, 8),
        "Decision Tree": (
            Pipeline([("scaler", "passthrough"),
                      ("clf", DecisionTreeClassifier(class_weight="balanced", random_state=rs))]),
            {"clf__max_depth": [3, 4, 5, 6, 8], "clf__min_samples_leaf": [10, 20, 40, 80],
             "clf__criterion": ["gini", "entropy"]}, 10),
        "Random Forest": (
            Pipeline([("scaler", "passthrough"),
                      ("clf", RandomForestClassifier(class_weight="balanced_subsample", random_state=rs, n_jobs=-1))]),
            {"clf__n_estimators": [200, 300], "clf__max_depth": [4, 6, 8, 12],
             "clf__min_samples_leaf": [5, 10, 20], "clf__max_features": ["sqrt", 0.3]}, 8),
        "XGBoost": (
            Pipeline([("scaler", "passthrough"),
                      ("clf", XGBClassifier(scale_pos_weight=pos_weight, eval_metric="logloss",
                                            random_state=rs, n_jobs=-1))]),
            {"clf__n_estimators": [150, 300], "clf__max_depth": [2, 3, 4, 5],
             "clf__learning_rate": [0.02, 0.05, 0.1], "clf__subsample": [0.7, 0.9],
             "clf__colsample_bytree": [0.6, 0.8, 1.0], "clf__min_child_weight": [1, 5, 10],
             "clf__reg_lambda": [1, 5, 10]}, 10),
    }


def metrics_at(y, proba, thr: float) -> dict:
    pred = (np.asarray(proba) >= thr).astype(int)
    return {
        "threshold": round(float(thr), 4),
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred, zero_division=0), 4),
        "f1": round(f1_score(y, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y, proba), 4),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def pick_thresholds(y, p) -> tuple:
    """Return (t_low, t_op) from out-of-fold calibrated probabilities."""
    y = np.asarray(y)
    grid = np.round(np.arange(0.02, 0.95, 0.005), 3)
    rec = np.array([(p[y == 1] >= t).mean() for t in grid])  # non-increasing in t
    t_op = float(grid[rec >= RECALL_TARGET].max())
    t_low = float(grid[rec >= LOW_BAND_RECALL].max())
    return min(t_low, t_op - 0.02), t_op


def _json_default(o):
    """json.dumps fallback: numpy scalars/arrays (np.bool_, np.float64, ...) -> plain Python."""
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple:
    """95% Wilson interval for a proportion (used for recall: successes = caught defaulters)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(float(centre - half), 4), round(float(centre + half), 4))


def band_validation(y, p, low_below: int, high_from: int) -> list:
    scores = np.round(np.asarray(p) * 100).astype(int)
    cats = assign_category(scores, low_below, high_from)
    out = []
    for c in ["Low Risk", "Medium Risk", "High Risk"]:
        m = cats == c
        out.append({"band": c, "loans": int(m.sum()),
                    "share_of_loans": round(float(m.mean()), 4),
                    "observed_default_rate": round(float(np.asarray(y)[m].mean()), 4) if m.any() else None})
    return out


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Tune, compare, calibrate and persist the default model.")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--reports-dir", default=str(ROOT / "reports"))
    args = ap.parse_args()

    t0 = time.time()
    models_dir, reports_dir = Path(args.models_dir), Path(args.reports_dir)
    fig_dir = reports_dir / "figures"
    for d in (models_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("Loading and engineering features...")
    raw = build_feature_table(args.db)
    model_df = get_model_ready(raw)
    y = model_df[TARGET].astype(int)
    X = model_df.drop(columns=[TARGET])
    groups = raw.loc[X.index, "customer_id"]
    feature_names = X.columns.tolist()

    # 80/20 stratified split, grouped by customer (first fold of a 5-fold grouped split)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    tr, te = next(splitter.split(X, y, groups))
    X_tr, X_te, y_tr, y_te, g_tr = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te], groups.iloc[tr]
    assert not set(groups.iloc[tr]) & set(groups.iloc[te]), "customer leaked across train/test"
    print(f"Train: {len(X_tr):,} loans / {g_tr.nunique():,} customers | Test: {len(X_te):,} loans / "
          f"{groups.iloc[te].nunique():,} customers | default rate train {y_tr.mean():.1%} test {y_te.mean():.1%}")
    print("Customers shared between train and test: 0")

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pos_weight = float((y_tr == 0).sum() / (y_tr == 1).sum())

    # ---- tune every candidate on the training set only ----
    results, fitted = [], {}
    print("\nTuning (RandomizedSearchCV, 5-fold grouped CV, scoring=roc_auc)...")
    for name, (pipe, space, n_iter) in candidates(pos_weight).items():
        s = time.time()
        n_iter = min(n_iter, int(np.prod([len(v) for v in space.values()])))
        search = RandomizedSearchCV(pipe, space, n_iter=n_iter, scoring="roc_auc", cv=cv,
                                    random_state=RANDOM_STATE, n_jobs=1, refit=True)
        search.fit(X_tr, y_tr, groups=g_tr)
        fitted[name] = search.best_estimator_
        test_raw = search.best_estimator_.predict_proba(X_te)[:, 1]
        m = metrics_at(y_te, test_raw, 0.5)
        m.update({"model": name,
                  "cv_roc_auc_mean": round(float(search.best_score_), 4),
                  "cv_roc_auc_std": round(float(search.cv_results_["std_test_score"][search.best_index_]), 4),
                  "best_params": {k.replace("clf__", ""): (v.item() if hasattr(v, "item") else v)
                                  for k, v in search.best_params_.items()},
                  "search_seconds": round(time.time() - s, 1)})
        results.append(m)
        print(f"  {name:<20} CV AUC {m['cv_roc_auc_mean']:.3f}+/-{m['cv_roc_auc_std']:.3f} | "
              f"test AUC {m['roc_auc']:.3f} @0.5 recall {m['recall']:.3f} | {m['search_seconds']}s")

    # ---- champion: highest cross-validated ROC-AUC (test set NOT used) ----
    champion = max(results, key=lambda r: r["cv_roc_auc_mean"])["model"]
    base = fitted[champion]
    print(f"\nChampion (by CV ROC-AUC): {champion}")

    # ---- calibration + thresholds from OUT-OF-FOLD training predictions ----
    oof_raw = cross_val_predict(clone(base), X_tr, y_tr, cv=cv, groups=g_tr, method="predict_proba")[:, 1]
    calibrator = PlattCalibrator().fit(oof_raw, y_tr)
    oof_cal = calibrator.predict(oof_raw)
    t_low, t_op = pick_thresholds(y_tr, oof_cal)
    low_below, high_from = int(round(t_low * 100)), int(round(t_op * 100))
    print(f"Operating threshold (recall >= {RECALL_TARGET:.2f} on OOF): {t_op:.3f} | "
          f"Low-risk cut: {t_low:.3f} -> bands: Low <{low_below}, Medium {low_below}-{high_from - 1}, High >={high_from}")

    # ---- single evaluation on the held-out test set ----
    p_raw = base.predict_proba(X_te)[:, 1]
    p_cal = calibrator.predict(p_raw)
    at_default = metrics_at(y_te, p_raw, 0.5)
    at_operating = metrics_at(y_te, p_cal, t_op)
    calib = {
        "mean_predicted_raw": round(float(p_raw.mean()), 4),
        "mean_predicted_calibrated": round(float(p_cal.mean()), 4),
        "actual_default_rate": round(float(y_te.mean()), 4),
        "brier_raw": round(brier_score_loss(y_te, p_raw), 4),
        "brier_calibrated": round(brier_score_loss(y_te, p_cal), 4),
    }
    bands_test = band_validation(y_te, p_cal, low_below, high_from)
    thr_table = [dict(metrics_at(y_te, p_cal, t), flagged_share=round(float((p_cal >= t).mean()), 4))
                 for t in np.round(np.arange(0.10, 0.61, 0.05), 2)]
    for r in thr_table:
        r.pop("confusion_matrix"); r.pop("accuracy"); r.pop("roc_auc")
    pd.DataFrame(thr_table).to_csv(reports_dir / "threshold_analysis.csv", index=False)

    print(f"\nHeld-out test ({len(y_te)} loans):")
    print(f"  ROC-AUC {at_default['roc_auc']:.3f}  | @0.50 raw: recall {at_default['recall']:.3f} precision {at_default['precision']:.3f}")
    print(f"  @{t_op:.3f} calibrated: recall {at_operating['recall']:.3f} precision {at_operating['precision']:.3f} F1 {at_operating['f1']:.3f}")
    print(f"  mean predicted P(default): raw {calib['mean_predicted_raw']:.3f} -> calibrated {calib['mean_predicted_calibrated']:.3f} (actual {calib['actual_default_rate']:.3f})")
    for b in bands_test:
        print(f"  {b['band']:<12} {b['loans']:>4} loans  observed default rate {b['observed_default_rate']:.1%}")

    # ---- SHAP (global) ----
    bg = X_tr.sample(n=min(100, len(X_tr)), random_state=RANDOM_STATE)
    explainer = build_explainer(base, bg)
    sv_te = shap_values(explainer, base, X_te)
    imp = importance_table(sv_te, feature_names)
    imp.to_csv(models_dir / "feature_importance.csv", index=False)
    summary_plot(sv_te, X_te, str(fig_dir / "shap_summary.png"))

    # ---- out-of-fold probability for EVERY disbursed loan (honest per-loan scores) ----
    full_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_all = calibrator.predict(
        cross_val_predict(clone(base), X, y, cv=full_cv, groups=groups, method="predict_proba")[:, 1])
    pd.DataFrame({
        "application_id": raw.loc[X.index, "application_id"].values,
        "customer_id": groups.values,
        "loan_default": y.values,
        "default_probability": np.round(oof_all, 4),
    }).to_csv(models_dir / "oof_predictions.csv", index=False)

    # ---- plots ----
    plt.figure(figsize=(5.5, 4.5))
    for name, est in fitted.items():
        fpr, tpr, _ = roc_curve(y_te, est.predict_proba(X_te)[:, 1])
        plt.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_te, est.predict_proba(X_te)[:, 1]):.2f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC curves (held-out test)")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(fig_dir / "roc_curves.png", dpi=130); plt.close()

    plt.figure(figsize=(5, 4.5))
    for lbl, pp in (("raw (balanced weights)", p_raw), ("calibrated", p_cal)):
        fo, mp = calibration_curve(y_te, pp, n_bins=8, strategy="quantile")
        plt.plot(mp, fo, marker="o", label=lbl)
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("Mean predicted probability"); plt.ylabel("Observed default rate")
    plt.title("Calibration (held-out test)"); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(fig_dir / "calibration.png", dpi=130); plt.close()

    cm = np.array(at_operating["confusion_matrix"])
    plt.figure(figsize=(4.4, 4))
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=14,
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.xticks([0, 1], ["No default", "Default"]); plt.yticks([0, 1], ["No default", "Default"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"Confusion matrix @ threshold {t_op:.2f}"); plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrix.png", dpi=130); plt.close()

    tt = pd.DataFrame(thr_table)
    plt.figure(figsize=(5.5, 4))
    plt.plot(tt["threshold"], tt["recall"], marker="o", label="recall")
    plt.plot(tt["threshold"], tt["precision"], marker="o", label="precision")
    plt.axhline(RECALL_TARGET, color="red", ls="--", lw=0.8, label="PRD recall target")
    plt.axvline(t_op, color="grey", ls=":", lw=0.8)
    plt.xlabel("Decision threshold (calibrated P)"); plt.title("Recall / precision trade-off")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(fig_dir / "threshold_tradeoff.png", dpi=130); plt.close()

    # ---- persist ----
    train_seconds = round(time.time() - t0, 1)
    bands = {"low_below": low_below, "high_from": high_from, "operating_threshold": round(t_op, 4),
             "low_cut_threshold": round(t_low, 4),
             "rule": "score = round(P x 100); Low < low_below <= Medium < high_from <= High"}
    (models_dir / "risk_bands.json").write_text(json.dumps(bands, indent=2, default=_json_default))
    with open(models_dir / "default_model.pkl", "wb") as f:
        pickle.dump({
            "model": base, "calibrator": calibrator, "model_name": champion,
            "feature_names": feature_names, "medians": X_tr.median(), "background": bg,
            "bands": bands, "random_state": RANDOM_STATE,
            "sklearn_version": sklearn.__version__,
        }, f)

    tp = int(at_operating["confusion_matrix"][1][1]); n_pos = int(sum(at_operating["confusion_matrix"][1]))
    recall_ci = wilson_ci(tp, n_pos)
    scorecard = {
        "roc_auc_target": AUC_TARGET, "roc_auc_achieved": at_default["roc_auc"],
        "roc_auc_met": bool(at_default["roc_auc"] >= AUC_TARGET),
        "recall_target": RECALL_TARGET, "recall_at_operating_threshold": at_operating["recall"],
        "recall_met": bool(at_operating["recall"] >= RECALL_TARGET),
        "recall_ci95_at_operating_threshold": list(recall_ci),
        "recall_target_inside_ci95": bool(recall_ci[0] <= RECALL_TARGET <= recall_ci[1]),
        "test_defaulters": n_pos,
        "recall_at_default_0.5": at_default["recall"],
        "training_under_5_min": train_seconds < 300, "training_seconds": train_seconds,
    }
    (models_dir / "metrics_report.json").write_text(json.dumps({
        "champion_model": champion,
        "selection_rule": "highest 5-fold grouped-CV ROC-AUC on the training set; test set not used for selection",
        "split": {"train_loans": len(X_tr), "test_loans": len(X_te),
                  "train_customers": int(g_tr.nunique()), "test_customers": int(groups.iloc[te].nunique()),
                  "customers_shared": 0, "test_default_rate": round(float(y_te.mean()), 4)},
        "n_features": len(feature_names),
        "champion_test_at_0.5_raw": at_default,
        "champion_test_at_operating_threshold": at_operating,
        "calibration": calib, "band_validation_test": bands_test, "bands": bands,
        "success_scorecard": scorecard,
        "all_candidates": results,
    }, indent=2, default=_json_default))

    print(f"\nSaved model, calibrator, bands, SHAP importance, OOF predictions and 5 figures ({train_seconds}s total)")
    print("Top 8 drivers (mean |SHAP|):")
    print(imp.head(8).to_string(index=False))


if __name__ == "__main__":
    main()