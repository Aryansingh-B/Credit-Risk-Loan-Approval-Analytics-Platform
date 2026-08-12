"""
features.py
------------
Credora Finance | Credit Risk & Loan Approval Analytics Platform

Builds the modelling feature set: aggregates transactions to customer
grain, joins customers + credit_history + loan_applications, engineers
ratio features, and applies leakage control.

Key design decision (per PRD 13.2): transactions are aggregated to ONE
ROW PER CUSTOMER before joining to applications. Joining at transaction
grain would duplicate every application row per transaction -- the most
common leakage/duplication mistake in this project.

Leakage control (per PRD 13.3): approval_status, approved_amount,
decision_date and disbursed_flag are only known AFTER the underwriting
decision, so none of them are used as predictive features.

Usage:
    from src.features import build_feature_table
    df = build_feature_table("data/credit_risk.db")
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

# Columns known only after the credit decision -- excluded from modelling
LEAKAGE_COLUMNS = ["approval_status", "approved_amount", "decision_date", "disbursed_flag"]

# Identifier / contact / raw-date columns not used as model inputs
NON_MODEL_COLUMNS = [
    "customer_id", "application_id", "credit_id", "first_name", "last_name",
    "email", "phone", "date_of_birth", "customer_since", "as_of_date",
    "application_date", "occupation",
]

CATEGORICAL_COLUMNS = [
    "gender", "marital_status", "education", "employment_type",
    "residence_type", "loan_type", "loan_purpose", "city", "state", "kyc_status",
]


def _aggregate_transactions(conn: sqlite3.Connection) -> pd.DataFrame:
    """Collapse the transactions table to one row per customer (6-month cash-flow profile)."""
    txn = pd.read_sql("SELECT * FROM transactions", conn)

    inflow = txn.loc[txn.transaction_type == "Credit"].groupby("customer_id")["amount"].sum()
    outflow = txn.loc[txn.transaction_type == "Debit"].groupby("customer_id")["amount"].sum()
    emi_outflow = txn.loc[txn.category == "EMI"].groupby("customer_id")["amount"].sum()
    txn_count = txn.groupby("customer_id").size()
    avg_balance = txn.groupby("customer_id")["balance_after"].mean()

    agg = pd.DataFrame({
        "txn_total_inflow": inflow,
        "txn_total_outflow": outflow,
        "txn_emi_outflow": emi_outflow,
        "txn_count": txn_count,
        "txn_avg_balance": avg_balance,
    }).fillna(0.0)

    agg["txn_net_cashflow"] = agg["txn_total_inflow"] - agg["txn_total_outflow"]
    agg["txn_inflow_outflow_ratio"] = agg["txn_total_inflow"] / agg["txn_total_outflow"].replace(0, np.nan)
    agg["txn_inflow_outflow_ratio"] = agg["txn_inflow_outflow_ratio"].fillna(0.0)

    agg = agg.reset_index()
    return agg


def build_feature_table(db_path: str = "data/credit_risk.db") -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        customers = pd.read_sql("SELECT * FROM customers", conn)
        credit = pd.read_sql("SELECT * FROM credit_history", conn)
        loans = pd.read_sql("SELECT * FROM loan_applications", conn)
        txn_agg = _aggregate_transactions(conn)
    finally:
        conn.close()

    df = (
        loans
        .merge(customers, on="customer_id", how="left", suffixes=("", "_cust"))
        .merge(credit, on="customer_id", how="left", suffixes=("", "_credit"))
        .merge(txn_agg, on="customer_id", how="left")
    )

    # --- Engineered ratio / derived features ---
    df["debt_to_limit_ratio"] = df["total_outstanding_debt"] / df["total_credit_limit"].replace(0, np.nan)
    df["debt_to_limit_ratio"] = df["debt_to_limit_ratio"].fillna(0.0)

    df["loan_to_income_ratio"] = df["requested_amount"] / df["annual_income"].replace(0, np.nan)
    df["loan_to_income_ratio"] = df["loan_to_income_ratio"].fillna(0.0)

    df["customer_tenure_days"] = (
        pd.to_datetime(df["application_date"]) - pd.to_datetime(df["customer_since"])
    ).dt.days.clip(lower=0)

    # Fill transaction aggregates for customers with zero transactions in the window
    txn_feature_cols = [
        "txn_total_inflow", "txn_total_outflow", "txn_emi_outflow",
        "txn_count", "txn_avg_balance", "txn_net_cashflow", "txn_inflow_outflow_ratio",
    ]
    df[txn_feature_cols] = df[txn_feature_cols].fillna(0.0)

    # --- Leakage control: drop decision-time-only columns ---
    df = df.drop(columns=[c for c in LEAKAGE_COLUMNS if c in df.columns])

    return df


def get_model_ready(df: pd.DataFrame) -> pd.DataFrame:
    """
    Restrict to the disbursed population with an observed outcome (PRD 13.1),
    drop rows with a missing target, impute, and one-hot encode categoricals.
    """
    modelling_df = df.loc[df["loan_default"].notna()].copy()

    numeric_cols = modelling_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "loan_default"]
    for col in numeric_cols:
        if modelling_df[col].isna().any():
            modelling_df[col] = modelling_df[col].fillna(modelling_df[col].median())

    for col in CATEGORICAL_COLUMNS:
        if col in modelling_df.columns:
            modelling_df[col] = modelling_df[col].fillna("Unknown")

    model_input = modelling_df.drop(columns=[c for c in NON_MODEL_COLUMNS if c in modelling_df.columns])
    present_categoricals = [c for c in CATEGORICAL_COLUMNS if c in model_input.columns]
    model_input = pd.get_dummies(model_input, columns=present_categoricals, drop_first=True)

    return model_input


if __name__ == "__main__":
    features = build_feature_table()
    print(f"Feature table: {features.shape[0]:,} rows x {features.shape[1]} columns")
    model_df = get_model_ready(features)
    print(f"Model-ready (disbursed only): {model_df.shape[0]:,} rows x {model_df.shape[1]} columns")
    print(f"Default rate in modelling population: {model_df['loan_default'].mean():.1%}")
