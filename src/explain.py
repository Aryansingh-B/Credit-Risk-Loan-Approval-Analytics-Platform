"""
explain.py
----------
Credora Finance | Credit Risk & Loan Approval Analytics Platform

SHAP-based explainability (PRD F6 / 13.6 and the NFR "every prediction is
traceable to ranked feature contributions").

Contributions are in log-odds of the base (pre-calibration) model. The
Platt calibrator is a monotonic transform, so the SIGN and RANK of a
feature's contribution are unchanged by calibration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _parts(pipe):
    return pipe.named_steps["scaler"], pipe.named_steps["clf"]


def _transform(pipe, X: pd.DataFrame) -> np.ndarray:
    scaler, _ = _parts(pipe)
    return X.values if isinstance(scaler, str) else scaler.transform(X)


def build_explainer(pipe, X_background: pd.DataFrame):
    _, clf = _parts(pipe)
    if hasattr(clf, "coef_"):
        return shap.LinearExplainer(clf, _transform(pipe, X_background))
    return shap.TreeExplainer(clf)


def shap_values(explainer, pipe, X: pd.DataFrame) -> np.ndarray:
    """(n_rows, n_features) matrix of contributions toward P(default)."""
    sv = explainer.shap_values(_transform(pipe, X))
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv


def importance_table(sv: np.ndarray, feature_names) -> pd.DataFrame:
    imp = pd.DataFrame({
        "feature": list(feature_names),
        "mean_abs_shap": np.abs(sv).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp["rank"] = np.arange(1, len(imp) + 1)
    return imp


def summary_plot(sv: np.ndarray, X: pd.DataFrame, path: str, max_display: int = 15) -> None:
    plt.figure()
    shap.summary_plot(sv, X, show=False, max_display=max_display)
    plt.title("What drives predicted default risk (SHAP)")
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close("all")


def top_drivers(sv: np.ndarray, feature_names, k: int = 3) -> list:
    """Per-row text: the k features pushing risk UP the most, e.g. 'debt_to_income_ratio (+0.41)'."""
    names = np.asarray(feature_names)
    order = np.argsort(-sv, axis=1)[:, :k]
    out = []
    for i, idx in enumerate(order):
        parts = [f"{names[j]} ({sv[i, j]:+.2f})" for j in idx if sv[i, j] > 0]
        out.append("; ".join(parts) if parts else "no risk-increasing factors")
    return out
