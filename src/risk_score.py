"""
risk_score.py
-------------
Credora Finance | Credit Risk & Loan Approval Analytics Platform

Turns calibrated default probability into a 0-100 risk score and a
Low / Medium / High category for EVERY customer and every disbursed loan
(PRD F4, FR-04, FR-05, 13.7).

    risk_score    = round(P(default) x 100)
    risk_category = bands from models/risk_bands.json (tuned in train.py)

Outputs
  models/scored_loans.csv      one row per disbursed loan (3,403). Probabilities are
                               OUT-OF-FOLD: each loan is scored by a model that never
                               saw its customer, so these are honest estimates.
  models/scored_customers.csv  one row per customer (5,000), with a `score_basis`:
      disbursed_loan        customer's latest disbursed loan, out-of-fold probability
      application_not_disbursed  latest application never disbursed (rejected / under
                               review): scored by the final model. Treat with care --
                               the model learned only from disbursed loans.
      reference_loan        customer has never applied: scored against a standard
                               reference loan (see features.build_reference_applications).
  and the same two tables inside the database for the SQL layer / dashboard.

Every customer row also carries `top_risk_drivers`: the SHAP features pushing
that customer's risk UP the most (explainability NFR).
"""

from __future__ import annotations

import argparse
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from calibration import assign_category
from explain import build_explainer, shap_values, top_drivers
from features import DEFAULT_DB, ROOT, TARGET, build_feature_table, build_reference_applications, to_matrix


def score_all(db_path: str, models_dir: str):
    models_dir = Path(models_dir)
    with open(models_dir / "default_model.pkl", "rb") as f:
        art = pickle.load(f)
    base, calibrator = art["model"], art["calibrator"]
    feature_names, medians, bands = art["feature_names"], art["medians"], art["bands"]
    low_below, high_from = bands["low_below"], bands["high_from"]

    raw = build_feature_table(db_path).sort_values("application_date", kind="stable")
    ref = build_reference_applications(db_path)
    oof = pd.read_csv(models_dir / "oof_predictions.csv")

    conn = sqlite3.connect(db_path)
    try:
        customers = pd.read_sql("SELECT customer_id, city, state FROM customers", conn)
    finally:
        conn.close()

    # ---------------- loan-level (disbursed, out-of-fold) ----------------
    loans = oof.merge(raw[["application_id", "application_date", "loan_type"]], on="application_id", how="left")
    loans["risk_score"] = np.round(loans["default_probability"] * 100).astype(int)
    loans["risk_category"] = assign_category(loans["risk_score"], low_below, high_from)
    loans = loans[["application_id", "customer_id", "application_date", "loan_type",
                   "default_probability", "risk_score", "risk_category", "loan_default"]]

    # ---------------- customer-level: choose ONE row per customer ----------------
    disbursed_rows = raw[raw[TARGET].notna()].drop_duplicates("customer_id", keep="last").assign(
        score_basis="disbursed_loan")
    done = set(disbursed_rows["customer_id"])
    other_rows = raw[~raw["customer_id"].isin(done)].drop_duplicates("customer_id", keep="last").assign(
        score_basis="application_not_disbursed")
    ref_rows = ref.assign(score_basis="reference_loan") if len(ref) else ref
    chosen = pd.concat([disbursed_rows, other_rows, ref_rows], ignore_index=True)

    X = to_matrix(chosen, feature_names, medians)
    p = calibrator.predict(base.predict_proba(X)[:, 1])

    # disbursed customers: replace the in-sample model score with the out-of-fold one
    oof_map = oof.set_index("application_id")["default_probability"]
    is_disb = chosen["score_basis"].eq("disbursed_loan").values
    p[is_disb] = chosen.loc[is_disb, "application_id"].map(oof_map).values

    explainer = build_explainer(base, art["background"])
    drivers = top_drivers(shap_values(explainer, base, X), feature_names, k=3)

    scored = pd.DataFrame({
        "customer_id": chosen["customer_id"].values,
        "score_basis": chosen["score_basis"].values,
        "application_id": chosen["application_id"].values,
        "default_probability": np.round(p, 4),
    })
    scored["risk_score"] = np.round(scored["default_probability"] * 100).astype(int)
    scored["risk_category"] = assign_category(scored["risk_score"], low_below, high_from)
    scored["top_risk_drivers"] = drivers
    scored = scored.merge(customers, on="customer_id", how="left")
    scored = scored[["customer_id", "city", "state", "score_basis", "application_id",
                     "default_probability", "risk_score", "risk_category", "top_risk_drivers"]]
    scored = scored.sort_values(["risk_score", "default_probability"], ascending=False).reset_index(drop=True)
    return scored, loans


def persist(scored: pd.DataFrame, loans: pd.DataFrame, db_path: str, models_dir: str) -> None:
    models_dir = Path(models_dir)
    scored.to_csv(models_dir / "scored_customers.csv", index=False)
    loans.to_csv(models_dir / "scored_loans.csv", index=False)
    conn = sqlite3.connect(db_path)
    try:
        scored.to_sql("scored_customers", conn, if_exists="replace", index=False)
        loans.to_sql("scored_loans", conn, if_exists="replace", index=False)
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Score every customer and disbursed loan 0-100.")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    args = ap.parse_args()

    scored, loans = score_all(args.db, args.models_dir)
    persist(scored, loans, args.db, args.models_dir)

    print(f"Scored {len(scored):,} customers (unique: {scored.customer_id.nunique():,}) and {len(loans):,} disbursed loans")
    print("\nCustomers by score basis:")
    print(scored["score_basis"].value_counts().to_string())
    print("\nCustomers by risk category:")
    print(scored["risk_category"].value_counts().to_string())
    print("\nSanity check -- observed default rate by band on ALL disbursed loans (out-of-fold):")
    print((loans.groupby("risk_category")["loan_default"].agg(loans="count", observed_default_rate="mean")
           .reindex(["Low Risk", "Medium Risk", "High Risk"]).round(3)).to_string())
    print("\nTop 5 highest-risk customers:")
    print(scored.head(5)[["customer_id", "score_basis", "risk_score", "risk_category", "top_risk_drivers"]].to_string(index=False))


if __name__ == "__main__":
    main()
