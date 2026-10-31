"""
calibration.py
--------------
Platt (sigmoid) probability calibration.

Why this exists: the champion is trained with class_weight='balanced' so it
catches more defaulters, but that inflates predicted probabilities (mean
P(default) ~0.43 against a real default rate of ~0.23). A risk score
built as round(P x 100) is only meaningful if P is a real probability, so
we fit a 2-parameter sigmoid on OUT-OF-FOLD predictions and apply it to
every score.

Lives in its own module (not train.py) so the pickled artefact can be
loaded from any entry point: notebooks, risk_score.py, the dashboard.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p)).reshape(-1, 1)


class PlattCalibrator:
    """Maps raw model probabilities to calibrated probabilities."""

    def fit(self, raw_proba, y):
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(raw_proba), y)
        return self

    def predict(self, raw_proba) -> np.ndarray:
        return self.lr_.predict_proba(_logit(raw_proba))[:, 1]


# --------------------------------------------------------------------------
# Risk banding (PRD 13.7) -- thresholds are TUNED, not hard-coded
# --------------------------------------------------------------------------
def assign_category(scores, low_below: int, high_from: int):
    """
    risk_score < low_below           -> Low Risk
    low_below <= score < high_from   -> Medium Risk
    risk_score >= high_from          -> High Risk
    Cut-offs come from models/risk_bands.json, set in train.py from
    out-of-fold predictions (see model card for how).
    """
    scores = np.asarray(scores)
    return np.where(scores < low_below, "Low Risk",
                    np.where(scores < high_from, "Medium Risk", "High Risk"))
