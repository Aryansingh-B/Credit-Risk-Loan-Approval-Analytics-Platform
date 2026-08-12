"""
data_load.py
------------
Credora Finance | Credit Risk & Loan Approval Analytics Platform

Loads the four raw CSVs (customers, credit_history, loan_applications,
transactions) into a governed relational database, enforcing the same
primary/foreign keys and check constraints as sql/credit_risk_platform.sql.

Default engine: SQLite (zero-setup, fully portable -- good for local dev,
CI and grading). The production DDL targets PostgreSQL 14+
(sql/credit_risk_platform.sql); swap DB_URL for a Postgres DSN and this
script's psycopg2 path to move environments without touching the schema
design.

Usage:
    python src/data_load.py
    python src/data_load.py --db data/credit_risk.db --datasets datasets/
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA_SQLITE = """
PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS loan_applications;
DROP TABLE IF EXISTS credit_history;
DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    customer_id         TEXT PRIMARY KEY,
    first_name           TEXT NOT NULL,
    last_name             TEXT NOT NULL,
    gender                 TEXT CHECK (gender IN ('Male','Female','Other')),
    date_of_birth           TEXT NOT NULL,
    age                       INTEGER CHECK (age BETWEEN 18 AND 100),
    marital_status             TEXT,
    dependents                  INTEGER DEFAULT 0 CHECK (dependents >= 0),
    education                     TEXT,
    employment_type                TEXT,
    occupation                       TEXT,
    annual_income                     REAL CHECK (annual_income >= 0),
    city                                TEXT,
    state                                 TEXT,
    residence_type                         TEXT,
    years_at_residence                       INTEGER CHECK (years_at_residence >= 0),
    email                                      TEXT,
    phone                                        TEXT,
    kyc_status                                     TEXT DEFAULT 'Pending' CHECK (kyc_status IN ('Verified','Pending','Rejected')),
    customer_since                                   TEXT NOT NULL
);

CREATE TABLE credit_history (
    credit_id                 TEXT PRIMARY KEY,
    customer_id                 TEXT NOT NULL UNIQUE,
    as_of_date                    TEXT NOT NULL,
    credit_score                    INTEGER CHECK (credit_score BETWEEN 300 AND 900),
    num_open_accounts                 INTEGER CHECK (num_open_accounts >= 0),
    num_credit_inquiries_6m             INTEGER CHECK (num_credit_inquiries_6m >= 0),
    credit_utilization_ratio              REAL CHECK (credit_utilization_ratio BETWEEN 0 AND 1),
    num_late_payments_30d                   INTEGER DEFAULT 0,
    num_late_payments_90d                     INTEGER DEFAULT 0,
    num_defaults_prior                          INTEGER DEFAULT 0,
    bankruptcies                                  INTEGER DEFAULT 0,
    total_credit_limit                              REAL CHECK (total_credit_limit >= 0),
    total_outstanding_debt                            REAL CHECK (total_outstanding_debt >= 0),
    oldest_account_age_months                           INTEGER CHECK (oldest_account_age_months >= 0),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
);

CREATE TABLE loan_applications (
    application_id            TEXT PRIMARY KEY,
    customer_id                  TEXT NOT NULL,
    application_date               TEXT NOT NULL,
    loan_type                        TEXT NOT NULL,
    loan_purpose                       TEXT,
    requested_amount                     REAL CHECK (requested_amount > 0),
    loan_term_months                       INTEGER CHECK (loan_term_months > 0),
    interest_rate                            REAL CHECK (interest_rate >= 0),
    applicant_monthly_income                   REAL CHECK (applicant_monthly_income >= 0),
    existing_emi                                 REAL DEFAULT 0,
    debt_to_income_ratio                           REAL CHECK (debt_to_income_ratio >= 0),
    collateral_flag                                  INTEGER DEFAULT 0 CHECK (collateral_flag IN (0,1)),
    approval_status                                    TEXT NOT NULL CHECK (approval_status IN ('Approved','Rejected','Under Review')),
    approved_amount                                      REAL DEFAULT 0,
    decision_date                                          TEXT,
    disbursed_flag                                           INTEGER DEFAULT 0 CHECK (disbursed_flag IN (0,1)),
    loan_default                                               INTEGER CHECK (loan_default IN (0,1)),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
);

CREATE TABLE transactions (
    transaction_id    TEXT PRIMARY KEY,
    customer_id          TEXT NOT NULL,
    transaction_date       TEXT NOT NULL,
    transaction_type         TEXT CHECK (transaction_type IN ('Credit','Debit')),
    category                    TEXT,
    channel                       TEXT,
    amount                          REAL CHECK (amount >= 0),
    balance_after                     REAL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
);

CREATE INDEX idx_credit_customer ON credit_history (customer_id);
CREATE INDEX idx_credit_score    ON credit_history (credit_score);
CREATE INDEX idx_loan_customer   ON loan_applications (customer_id);
CREATE INDEX idx_loan_status     ON loan_applications (approval_status);
CREATE INDEX idx_loan_appdate    ON loan_applications (application_date);
CREATE INDEX idx_loan_default    ON loan_applications (loan_default);
CREATE INDEX idx_txn_customer    ON transactions (customer_id);
CREATE INDEX idx_txn_date        ON transactions (transaction_date);
CREATE INDEX idx_txn_category    ON transactions (category);
"""

TABLE_ORDER = ["customers", "credit_history", "loan_applications", "transactions"]


def build_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQLITE)


def load_csvs(conn: sqlite3.Connection, datasets_dir: Path) -> dict:
    counts = {}
    for table in TABLE_ORDER:
        csv_path = datasets_dir / f"{table}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Expected dataset not found: {csv_path}")
        df = pd.read_csv(csv_path)
        df.to_sql(table, conn, if_exists="append", index=False)
        counts[table] = len(df)
    return counts


def validate_integrity(conn: sqlite3.Connection) -> None:
    """Fail loudly on orphan foreign keys -- data integrity is a hard NFR."""
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_key_check;")
    violations = cur.fetchall()
    if violations:
        raise ValueError(f"Foreign key violations detected: {violations}")

    checks = {
        "credit_history": "SELECT COUNT(*) FROM credit_history "
                           "WHERE customer_id NOT IN (SELECT customer_id FROM customers)",
        "loan_applications": "SELECT COUNT(*) FROM loan_applications "
                              "WHERE customer_id NOT IN (SELECT customer_id FROM customers)",
        "transactions": "SELECT COUNT(*) FROM transactions "
                         "WHERE customer_id NOT IN (SELECT customer_id FROM customers)",
    }
    for table, query in checks.items():
        orphan_count = cur.execute(query).fetchone()[0]
        if orphan_count:
            raise ValueError(f"{orphan_count} orphan rows found in {table}")

    dupe_emails = cur.execute(
        "SELECT COUNT(*) FROM (SELECT email FROM customers GROUP BY email HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    if dupe_emails:
        print(f"  [data-quality warning] {dupe_emails} duplicate customer email(s) found "
              f"(non-blocking; email is excluded from modelling per NFR security policy)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load Credora CSVs into the governed database.")
    parser.add_argument("--db", default="data/credit_risk.db", help="Output SQLite DB path")
    parser.add_argument("--datasets", default="datasets", help="Directory containing the 4 raw CSVs")
    args = parser.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        build_schema(conn)
        counts = load_csvs(conn, Path(args.datasets))
        validate_integrity(conn)
        conn.commit()
    finally:
        conn.close()

    print(f"Database built at {db_path}")
    for table, n in counts.items():
        print(f"  {table:<20} {n:>7,} rows loaded")
    print("Referential integrity: OK (0 orphan foreign keys)")


if __name__ == "__main__":
    main()
