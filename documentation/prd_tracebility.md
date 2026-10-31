# PRD Traceability Matrix

Every requirement in `PRD.docx`, where it is implemented, and the evidence. Numbers come from the final pipeline run.

## Functional requirements

| ID | Requirement | Status | Where / evidence |
|---|---|---|---|
| FR-01 | Relational schema with PKs, FKs, constraints | Met | `sql/credit_risk_platform.sql`; loaded and verified on PostgreSQL 16 and SQLite, 0 orphan keys |
| FR-02 | Bulk-load the four CSVs | Met | `\copy` block in the SQL file; `src/data_load.py` (SQLite) with integrity validation |
| FR-03 | 10 analytical SQL queries | Met | `sql/` section 7; `src/sql_analytics.py`; all 10 run on PostgreSQL (2-41 ms) and SQLite; results in `reports/sql_results/` |
| FR-04 | Default probability per disbursed loan | Met | `models/scored_loans.csv` - 3,403 loans, out-of-fold, calibrated |
| FR-05 | 0-100 score and Low/Medium/High label | Met | `models/scored_customers.csv` - 5,000 customers; tuned bands in `models/risk_bands.json` |
| FR-06 | Preprocessing (impute, encode, scale) | Met | `src/features.py`, scaling inside the model pipeline |
| FR-07 | Evaluation on held-out set | Met | `models/metrics_report.json`; accuracy, precision, recall, F1, ROC-AUC, confusion matrix |
| FR-08 | Ranked feature importance | Met | `models/feature_importance.csv` (mean |SHAP|) + `reports/figures/shap_summary.png` |
| FR-09 | Dashboard, 8 components | Met | `dashboard/app.py` (tested with Streamlit AppTest across filter combinations) |
| FR-10 | Filter high-risk list by category and region | Met | sidebar filters + watchlist table + CSV download |
| FR-11 | Persist the trained model | Met | `models/default_model.pkl` (model + calibrator + bands + feature schema) |
| FR-12 | Re-runnable end to end | Met | `python src/run_pipeline.py` |

## Success metrics (PRD section 16)

| Target | Result | Status |
|---|---|---|
| ROC-AUC >= 0.80 | 0.716 | **Not met** - four tuned algorithms agree at 0.69-0.72; data-limited |
| Recall >= 0.70 | 0.688 (95% CI 0.61-0.75) | **Not met on the point estimate**; target inside the interval |
| Training < 5 min | 85.6 s | Met |
| SQL queries < 2 s | max ~41 ms | Met (asserted by a test) |
| Zero orphan foreign keys | 0 | Met (asserted by a test) |

## Non-functional requirements

| NFR | Status | Evidence |
|---|---|---|
| Reproducibility | Met | fixed seeds; clean re-run reproduced identical metrics |
| Explainability | Met | SHAP global importance + per-customer `top_risk_drivers` |
| Leakage control | Met | `LEAKAGE_COLUMNS` excluded; asserted by a test |
| Privacy | Met with a caveat | no PII in model, scores, reports or dashboard; the synthetic raw `customers.csv` still contains names/emails/phones (provenance note in README) |
| Maintainability | Met | modular `src/`, no absolute paths, 12 tests |

## Documented deviations from the PRD

1. **Risk bands** are tuned (Low < 14, High >= 22) rather than the PRD's default 33/66, which assumes probabilities that calibrated scores on a ~23% default book almost never reach. The PRD allows tuning against the confusion matrix.
2. **Model inputs** follow the PRD feature list exactly; gender, marital status, city, state, loan purpose and KYC status are excluded (protected, administrative or high-cardinality).
3. **Customers with no application** are scored against a reference loan and flagged `reference_loan`.
4. **Approval rate** is approved / decided applications (PRD KPI definition); SQL 7.2 as written divides by all applications.
