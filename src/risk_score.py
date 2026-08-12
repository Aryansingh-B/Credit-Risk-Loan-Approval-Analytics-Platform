"""
risk_score.py
-------------
Credora Finance | Credit Risk & Loan Approval Analytics Platform

Transforms the trained model's default probability into a business-facing
0-100 risk score and a Low/Medium/High risk category (PRD 13.7), and
writes the result to models/scored_customers.csv (and a `scored_customers`
table in the database) -- the table the BI dashboard's high-risk list
reads from.

One row per customer: where a customer has multiple applications, the
most recent disbursed application's probability is used; customers with
no disbursed loan are scored using their most recent application's
features so the watchlist still covers the full book.

Usage:
    python src/risk_score.py
"""

from __future__ import annotations

import argparse
import pickle
import sqlite3
from pathlib import Path

import pandas as pd

from features import build_feature_table, NON_MODEL_COLUMNS, CATEGORICAL_COLUMNS

# Tunable thresholds against confusion matrix / business risk appetite (PRD 13.7)
LOW_MAX = 33
MEDIUM_MAX = 66


def categorize(score: int) -> str:
    if score <= LOW_MAX:
        return "Low Risk"
    if score <= MEDIUM_MAX:
        return "Medium Risk"
    return "High Risk"


def score_customers(db_path: str, models_dir: str) -> pd.DataFrame:
    models_dir = Path(models_dir)
    with open(models_dir / "default_model.pkl", "rb") as f:
        artefact = pickle.load(f)

    model = artefact["model"]
    feature_names = artefact["feature_names"]
    needs_scaling = artefact["needs_scaling"]
    scaler = artefact["scaler"]

    raw = build_feature_table(db_path)

    # One row per customer: keep the latest application per customer
    raw = raw.sort_values("application_date").drop_duplicates("customer_id", keep="last")

    numeric_cols = raw.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "loan_default"]
    for col in numeric_cols:
        raw[col] = raw[col].fillna(raw[col].median())
    for col in CATEGORICAL_COLUMNS:
        if col in raw.columns:
            raw[col] = raw[col].fillna("Unknown")

    model_input = raw.drop(columns=[c for c in NON_MODEL_COLUMNS if c in raw.columns] + ["loan_default"])
    present_categoricals = [c for c in CATEGORICAL_COLUMNS if c in model_input.columns]
    model_input = pd.get_dummies(model_input, columns=present_categoricals, drop_first=True)

    # Align to the exact training-time feature schema
    model_input = model_input.reindex(columns=feature_names, fill_value=0)

    X_score = scaler.transform(model_input) if needs_scaling else model_input
    proba = model.predict_proba(X_score)[:, 1]

    scored = pd.DataFrame({
        "customer_id": raw["customer_id"].values,
        "default_probability": proba.round(4),
    })
    scored["risk_score"] = (scored["default_probability"] * 100).round().astype(int)
    scored["risk_category"] = scored["risk_score"].apply(categorize)
    scored = scored.merge(
        raw[["customer_id", "city", "state"]].drop_duplicates("customer_id"),
        on="customer_id", how="left",
    )
    scored = scored.sort_values("risk_score", ascending=False).reset_index(drop=True)
    return scored


def persist(scored: pd.DataFrame, db_path: str, out_csv: str) -> None:
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_csv, index=False)

    conn = sqlite3.connect(db_path)
    try:
        scored.to_sql("scored_customers", conn, if_exists="replace", index=False)
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Score every customer 0-100 and bucket into risk categories.")
    parser.add_argument("--db", default="data/credit_risk.db")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--out", default="models/scored_customers.csv")
    args = parser.parse_args()

    scored = score_customers(args.db, args.models_dir)
    persist(scored, args.db, args.out)

    print(f"Scored {len(scored):,} customers -> {args.out} and 'scored_customers' table")
    print(scored["risk_category"].value_counts().to_string())
    print("\nTop 5 highest-risk customers:")
    print(scored.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
