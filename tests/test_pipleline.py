import json
import pickle
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import LEAKAGE_COLUMNS, RATIO_CAPS, build_feature_table, to_matrix  # noqa: E402

DB = ROOT / "data" / "credit_risk.db"
MODELS = ROOT / "models"
pytestmark = pytest.mark.skipif(not ((MODELS / "default_model.pkl").exists() and DB.exists()),
                                reason="database/model missing - run `python src/run_pipeline.py` first")

BANNED_FEATURE_PREFIXES = ["first_name", "last_name", "email", "phone", "gender", "marital_status",
                           "kyc_status", "city", "state", "loan_purpose", "customer_id", "application_id",
                           "score_basis"]


@pytest.fixture(scope="module")
def artefact():
    with open(MODELS / "default_model.pkl", "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def metrics():
    return json.loads((MODELS / "metrics_report.json").read_text())


@pytest.fixture(scope="module")
def customers():
    return pd.read_csv(MODELS / "scored_customers.csv")


@pytest.fixture(scope="module")
def loans():
    return pd.read_csv(MODELS / "scored_loans.csv")


def test_database_integrity_and_row_counts():
    conn = sqlite3.connect(DB)
    try:
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in ["customers", "credit_history", "loan_applications", "transactions"]}
    finally:
        conn.close()
    assert counts == {"customers": 5000, "credit_history": 5000, "loan_applications": 7500, "transactions": 40079}


def test_no_leakage_or_pii_in_model_features(artefact):
    feats = artefact["feature_names"]
    assert not set(LEAKAGE_COLUMNS) & set(feats)
    bad = [f for f in feats if any(f == b or f.startswith(b + "_") for b in BANNED_FEATURE_PREFIXES)]
    assert bad == []


def test_unbounded_ratios_are_capped():
    raw = build_feature_table(str(DB))
    for col, cap in RATIO_CAPS.items():
        assert raw[col].max() <= cap


def test_split_is_grouped_by_customer(metrics):
    assert metrics["split"]["customers_shared"] == 0


def test_probabilities_are_calibrated(metrics):
    c = metrics["calibration"]
    assert abs(c["mean_predicted_calibrated"] - c["actual_default_rate"]) < 0.02
    assert c["brier_calibrated"] <= c["brier_raw"]


def test_every_customer_scored_once(customers):
    assert len(customers) == 5000 and customers.customer_id.is_unique
    assert customers.default_probability.between(0, 1).all()
    assert customers.risk_score.between(0, 100).all()
    assert (customers.risk_score == np.round(customers.default_probability * 100)).all()
    assert set(customers.risk_category) <= {"Low Risk", "Medium Risk", "High Risk"}
    assert set(customers.score_basis) == {"disbursed_loan", "application_not_disbursed", "reference_loan"}


def test_categories_follow_tuned_bands(customers, artefact):
    b = artefact["bands"]
    low = customers.risk_score < b["low_below"]
    high = customers.risk_score >= b["high_from"]
    assert (customers.loc[low, "risk_category"] == "Low Risk").all()
    assert (customers.loc[high, "risk_category"] == "High Risk").all()
    assert (customers.loc[~low & ~high, "risk_category"] == "Medium Risk").all()


def test_bands_rank_risk_on_disbursed_loans(loans):
    assert len(loans) == 3403
    r = loans.groupby("risk_category").loan_default.mean()
    assert r["Low Risk"] < r["Medium Risk"] < r["High Risk"]


def test_no_pii_in_outputs(customers):
    for col in ["first_name", "last_name", "email", "phone", "date_of_birth"]:
        assert col not in customers.columns
    for f in (ROOT / "reports" / "sql_results").glob("*.csv"):
        assert not {"first_name", "last_name", "email", "phone"} & set(pd.read_csv(f, nrows=1).columns), f.name


def test_model_artefact_scores_new_rows(artefact):
    raw = build_feature_table(str(DB)).head(25)
    X = to_matrix(raw, artefact["feature_names"], artefact["medians"])
    p = artefact["calibrator"].predict(artefact["model"].predict_proba(X)[:, 1])
    assert p.shape == (25,) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()


def test_all_ten_sql_queries_run_under_two_seconds():
    from sql_analytics import run_all
    conn = sqlite3.connect(DB)
    try:
        res = run_all(conn, verbose=False)
    finally:
        conn.close()
    assert sorted(res) == list(range(1, 11))
    assert all(r["ms"] < 2000 for r in res.values())


def test_dashboard_renders_all_components_without_error():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=120).run()
    assert not at.exception
    numbered = {s.value[0] for s in at.subheader if s.value[:1].isdigit()}
    assert numbered == set("12345678")
