"""
features.py
-----------
Credora Finance | Credit Risk & Loan Approval Analytics Platform

Builds the modelling feature set: aggregates transactions to customer
grain, joins customers + credit_history + loan_applications, engineers
ratio features, and applies leakage control.

Design decisions (PRD 13.2 / 13.3)
  * Transactions are aggregated to ONE ROW PER CUSTOMER before joining to
    applications. Joining at transaction grain would duplicate every
    application row per transaction.
  * approval_status, approved_amount, decision_date and disbursed_flag are
    only known AFTER the underwriting decision -> never model inputs.
  * Only the categoricals the PRD lists are one-hot encoded (loan_type,
    employment_type, residence_type, education). High-cardinality fields
    (city, state, loan_purpose) are not model inputs, and neither are
    gender / marital_status / kyc_status: protected or administrative
    attributes the PRD's feature list does not call for.
  * email / phone / names are never model inputs (privacy NFR).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

LEAKAGE_COLUMNS = ["approval_status", "approved_amount", "decision_date", "disbursed_flag"]

MODEL_CATEGORICALS = ["loan_type", "employment_type", "residence_type", "education"]

NON_MODEL_COLUMNS = [
    "customer_id", "application_id", "credit_id", "first_name", "last_name",
    "email", "phone", "date_of_birth", "customer_since", "as_of_date",
    "application_date", "occupation",
    "gender", "marital_status", "kyc_status", "city", "state", "loan_purpose",
    "score_basis",
]

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = str(ROOT / "data" / "credit_risk.db")

# Unbounded engineered ratios are capped at ~the training 99th percentile so a handful of
# extreme customers (e.g. Rs 278 outflow vs Rs 46,932 inflow -> ratio 168) cannot make the
# linear champion extrapolate. Applied identically at train and score time.
RATIO_CAPS = {"txn_inflow_outflow_ratio": 10.0, "loan_to_income_ratio": 20.0}

TARGET = "loan_default"
SNAPSHOT_DATE = "2024-06-30"  # bureau snapshot date; used for reference applications


# --------------------------------------------------------------------------
# Loading + aggregation
# --------------------------------------------------------------------------
def _aggregate_transactions(conn: sqlite3.Connection) -> pd.DataFrame:
    """Collapse transactions to one row per customer (6-month cash-flow profile)."""
    txn = pd.read_sql("SELECT * FROM transactions", conn)

    inflow = txn.loc[txn.transaction_type == "Credit"].groupby("customer_id")["amount"].sum()
    outflow = txn.loc[txn.transaction_type == "Debit"].groupby("customer_id")["amount"].sum()
    emi_outflow = txn.loc[txn.category == "EMI"].groupby("customer_id")["amount"].sum()

    agg = pd.DataFrame({
        "txn_total_inflow": inflow,
        "txn_total_outflow": outflow,
        "txn_emi_outflow": emi_outflow,
        "txn_count": txn.groupby("customer_id").size(),
        "txn_avg_balance": txn.groupby("customer_id")["balance_after"].mean(),
    }).fillna(0.0)

    agg["txn_net_cashflow"] = agg["txn_total_inflow"] - agg["txn_total_outflow"]
    ratio = agg["txn_total_inflow"] / agg["txn_total_outflow"].replace(0, np.nan)
    agg["txn_inflow_outflow_ratio"] = ratio.fillna(0.0)
    return agg.reset_index()


def _load_tables(db_path: str):
    conn = sqlite3.connect(db_path)
    try:
        customers = pd.read_sql("SELECT * FROM customers", conn)
        credit = pd.read_sql("SELECT * FROM credit_history", conn)
        loans = pd.read_sql("SELECT * FROM loan_applications", conn)
        txn_agg = _aggregate_transactions(conn)
    finally:
        conn.close()
    return customers, credit, loans, txn_agg


def _assemble(loans, customers, credit, txn_agg) -> pd.DataFrame:
    """Join the four sources at application grain and engineer ratio features."""
    df = (
        loans
        .merge(customers, on="customer_id", how="left")
        .merge(credit, on="customer_id", how="left")
        .merge(txn_agg, on="customer_id", how="left")
    )

    df["debt_to_limit_ratio"] = (
        df["total_outstanding_debt"] / df["total_credit_limit"].replace(0, np.nan)
    ).fillna(0.0)
    df["loan_to_income_ratio"] = (
        df["requested_amount"] / df["annual_income"].replace(0, np.nan)
    ).fillna(0.0)
    for col, cap in RATIO_CAPS.items():
        df[col] = df[col].clip(upper=cap)
    df["customer_tenure_days"] = (
        pd.to_datetime(df["application_date"]) - pd.to_datetime(df["customer_since"])
    ).dt.days.clip(lower=0)

    txn_cols = [
        "txn_total_inflow", "txn_total_outflow", "txn_emi_outflow", "txn_count",
        "txn_avg_balance", "txn_net_cashflow", "txn_inflow_outflow_ratio",
    ]
    df[txn_cols] = df[txn_cols].fillna(0.0)

    return df.drop(columns=[c for c in LEAKAGE_COLUMNS if c in df.columns])


def build_feature_table(db_path: str = DEFAULT_DB) -> pd.DataFrame:
    """One row per REAL loan application (7,500), fully feature-engineered."""
    customers, credit, loans, txn_agg = _load_tables(db_path)
    return _assemble(loans, customers, credit, txn_agg)


def build_reference_applications(db_path: str = DEFAULT_DB) -> pd.DataFrame:
    """
    Feature rows for customers who have NEVER applied for a loan.

    The model scores an application, so a customer with no application has
    nothing to score. To still give every customer a comparable 0-100 score
    (PRD: "a risk score for every customer") we score them against a standard
    reference loan: the median disbursed amount, term and rate, Personal
    loan type, no collateral, existing EMI taken from their own 6-month EMI
    outflow. These scores are flagged score_basis='reference_loan' downstream
    because they answer "how risky would this customer be on a typical loan?",
    not "how risky is this actual application?".
    """
    customers, credit, loans, txn_agg = _load_tables(db_path)
    disbursed = loans[loans["loan_default"].notna()]
    none = customers[~customers["customer_id"].isin(loans["customer_id"])].copy()
    if none.empty:
        return pd.DataFrame()

    amount = float(disbursed["requested_amount"].median())
    term = int(disbursed["loan_term_months"].median())
    rate = float(disbursed["interest_rate"].median())
    r = rate / 1200.0
    new_emi = amount * r * (1 + r) ** term / ((1 + r) ** term - 1)

    emi_monthly = none[["customer_id"]].merge(txn_agg[["customer_id", "txn_emi_outflow"]],
                                              on="customer_id", how="left")["txn_emi_outflow"].fillna(0.0) / 6.0
    monthly_income = (none["annual_income"] / 12.0).replace(0, np.nan)

    ref = pd.DataFrame({
        "application_id": "REF-" + none["customer_id"].values,
        "customer_id": none["customer_id"].values,
        "application_date": SNAPSHOT_DATE,
        "loan_type": "Personal",
        "loan_purpose": "Reference Loan",
        "requested_amount": amount,
        "loan_term_months": term,
        "interest_rate": rate,
        "applicant_monthly_income": monthly_income.values,
        "existing_emi": emi_monthly.values,
        "collateral_flag": 0,
        "approval_status": "Reference",
        "approved_amount": 0.0,
        "decision_date": None,
        "disbursed_flag": 0,
        "loan_default": np.nan,
    })
    ref["debt_to_income_ratio"] = (ref["existing_emi"] + new_emi) / ref["applicant_monthly_income"]
    ref["debt_to_income_ratio"] = ref["debt_to_income_ratio"].fillna(disbursed["debt_to_income_ratio"].median())
    return _assemble(ref, customers, credit, txn_agg)


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------
def encode(df: pd.DataFrame, drop_first: bool = True) -> pd.DataFrame:
    """Numeric + one-hot design matrix (no target column, no leakage, no PII)."""
    drop = NON_MODEL_COLUMNS + LEAKAGE_COLUMNS + [TARGET]
    m = df.drop(columns=[c for c in drop if c in df.columns]).copy()
    for c in MODEL_CATEGORICALS:
        m[c] = m[c].fillna("Unknown")
    m = pd.get_dummies(m, columns=MODEL_CATEGORICALS, drop_first=drop_first)
    return m.astype(float)


def get_model_ready(df: pd.DataFrame) -> pd.DataFrame:
    """
    Disbursed population with an observed outcome (PRD 13.1): impute, encode,
    and return the design matrix with `loan_default` appended. The original
    row index is preserved so callers can map rows back to applications.
    """
    base = df.loc[df[TARGET].notna()]
    X = encode(base, drop_first=True)
    X = X.fillna(X.median())
    X[TARGET] = base[TARGET].astype(int).values
    return X


def to_matrix(df: pd.DataFrame, feature_names: list, medians: pd.Series) -> pd.DataFrame:
    """Scoring-time matrix aligned to the exact training feature schema."""
    X = encode(df, drop_first=False).reindex(columns=feature_names, fill_value=0.0)
    return X.fillna(medians).fillna(0.0)


if __name__ == "__main__":
    features = build_feature_table()
    print(f"Feature table: {features.shape[0]:,} rows x {features.shape[1]} columns")
    model_df = get_model_ready(features)
    print(f"Model-ready (disbursed only): {model_df.shape[0]:,} rows x {model_df.shape[1]} columns")
    print(f"Default rate in modelling population: {model_df[TARGET].mean():.1%}")
