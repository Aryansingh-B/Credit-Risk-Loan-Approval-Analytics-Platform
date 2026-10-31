"""
sql_analytics.py
----------------
Runs the ten analytical queries (sql/credit_risk_platform.sql, section 7)
against the SQLite database and saves each result to reports/sql_results/.

The .sql file is written for PostgreSQL 14+. SQLite lacks DATE_TRUNC, TO_CHAR
and PERCENTILE_CONT, so those three constructs are translated here; the
query logic is otherwise executed exactly as written in the .sql file.
"""

from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path

import pandas as pd

from features import DEFAULT_DB, ROOT

PII_COLUMNS = ["first_name", "last_name", "email", "phone", "date_of_birth"]  # never saved or displayed
SQL_FILE = ROOT / "sql" / "credit_risk_platform.sql"
NAMES = {
    1: "credit_score_analysis", 2: "loan_approval_trends", 3: "customer_segmentation",
    4: "high_risk_customers_rule_based", 5: "default_rate_by_loan_type", 6: "loan_amount_distribution",
    7: "monthly_approval_statistics", 8: "risk_categorisation", 9: "transaction_behaviour",
    10: "top_default_predictors",
}

# PERCENTILE_CONT(0.5) has no SQLite equivalent -> median via window functions
MEDIAN_QUERY_6 = """
WITH ranked AS (
    SELECT loan_type, requested_amount,
           ROW_NUMBER() OVER (PARTITION BY loan_type ORDER BY requested_amount) AS rn,
           COUNT(*)     OVER (PARTITION BY loan_type)                           AS cnt
    FROM loan_applications
), med AS (
    SELECT loan_type, AVG(requested_amount) AS median_amount
    FROM ranked WHERE rn IN ((cnt + 1) / 2, (cnt + 2) / 2) GROUP BY loan_type
)
SELECT a.loan_type, COUNT(*) AS applications,
       ROUND(MIN(a.requested_amount), 0) AS min_amount,
       ROUND(m.median_amount, 0)         AS median_amount,
       ROUND(AVG(a.requested_amount), 0) AS avg_amount,
       ROUND(MAX(a.requested_amount), 0) AS max_amount
FROM loan_applications a JOIN med m USING (loan_type)
GROUP BY a.loan_type, m.median_amount
ORDER BY avg_amount DESC
"""


def load_queries(sql_path: Path = SQL_FILE) -> dict:
    text = sql_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    body = text.split("7. ANALYTICAL QUERIES", 1)[1]
    chunks = re.split(r"^-- 7\.(\d+)\s+", body, flags=re.MULTILINE)[1:]
    queries = {}
    for num, chunk in zip(chunks[0::2], chunks[1::2]):
        title, _, sql = chunk.partition("\n")
        sql = re.split(r"\n-- ={5,}", sql)[0].strip().rstrip(";")
        queries[int(num)] = {"title": title.strip(), "sql": sql}
    return queries


def to_sqlite(num: int, sql: str) -> str:
    if num == 6:
        return MEDIAN_QUERY_6
    sql = re.sub(r"DATE_TRUNC\('month',\s*application_date\)::date", "strftime('%Y-%m-01', application_date)", sql)
    return sql.replace("TO_CHAR(application_date,'YYYY-MM')", "strftime('%Y-%m', application_date)")


def run_all(conn: sqlite3.Connection, out_dir: Path | None = None, verbose: bool = True) -> dict:
    results = {}
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    for num, q in load_queries().items():
        start = time.perf_counter()
        df = pd.read_sql(to_sqlite(num, q["sql"]), conn)
        ms = (time.perf_counter() - start) * 1000
        df = df.drop(columns=[c for c in PII_COLUMNS if c in df.columns])
        results[num] = {"title": q["title"], "df": df, "ms": ms}
        if out_dir:
            df.to_csv(Path(out_dir) / f"q7_{num:02d}_{NAMES[num]}.csv", index=False)
        if verbose:
            print(f"  7.{num:<2} {NAMES[num]:<32} {len(df):>4} rows  {ms:7.1f} ms")
    return results


def main() -> None:
    conn = sqlite3.connect(DEFAULT_DB)
    try:
        res = run_all(conn, ROOT / "reports" / "sql_results")
    finally:
        conn.close()
    slowest = max(r["ms"] for r in res.values())
    print(f"All {len(res)} queries ran. Slowest: {slowest:.1f} ms (PRD NFR: < 2000 ms)")


if __name__ == "__main__":
    main()
