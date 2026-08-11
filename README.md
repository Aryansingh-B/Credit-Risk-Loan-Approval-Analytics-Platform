# Credora Finance — Credit Risk & Loan Approval Analytics Platform

An end-to-end decision-support system for retail lending: a governed relational database,
a loan-default prediction model, a 0–100 risk-scoring engine, and a BI dashboard — built on
a book of ~5,000 customers and ~7,500 loan applications.

> Built for the Internmo Applied Data Science internship track (project code `CRDA-RISK-2024`).
> Full requirements: [`documentation/PRD.docx`](documentation/PRD.docx).

## What this does

Every loan application reaches underwriting with two open questions: *should this be
approved*, and *if disbursed, how likely is this borrower to default?* This project answers
both by unifying customer, bureau, application and transaction data into one schema, training
a default classifier on the disbursed book, converting its output into a business-facing risk
score, and surfacing everything on an interactive dashboard — without replacing the
underwriter's final call.

## Results at a glance

- **Data foundation:** 5,000 customers · 5,000 bureau records · 7,500 applications · 40,079
  transactions loaded with **0 orphan foreign keys**.
- **Champion model:** Logistic Regression — **ROC-AUC 0.704**, recall 0.648 on a held-out
  test set (see [`models/model_card.md`](models/model_card.md) for the full comparison
  against Decision Tree, Random Forest and XGBoost, and an honest discussion of why this
  falls short of the PRD's 0.80 stretch target).
- **Risk scoring:** every customer with an application gets a 0–100 score and a
  Low / Medium / High band (`models/scored_customers.csv`).
- **10 analytical SQL queries** and an **8-component BI dashboard**, both runnable end to end.

## Project structure

```
credit-risk-loan-analytics/
├── datasets/                      # raw CSVs (customers, credit_history, loan_applications, transactions)
├── sql/
│   └── credit_risk_platform.sql   # production DDL (PostgreSQL) + 10 analytical queries
├── notebooks/
│   ├── 01_eda.ipynb                # portfolio profiling + the 10 SQL queries, run and interpreted
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_risk_scoring.ipynb
├── src/
│   ├── data_load.py                # CSV -> governed DB, with integrity validation
│   ├── features.py                 # transaction aggregation, ratios, leakage control
│   ├── train.py                    # trains + compares 4 classifiers, selects champion
│   └── risk_score.py               # probability -> 0-100 score -> Low/Med/High
├── models/
│   ├── default_model.pkl           # persisted champion model + preprocessing
│   ├── feature_importance.csv
│   ├── metrics_report.json
│   ├── scored_customers.csv
│   └── model_card.md
├── dashboard/
│   └── app.py                      # Streamlit BI dashboard, all 8 required components
├── documentation/
│   ├── PRD.docx
│   └── data_dictionary.md
├── requirements.txt
└── README.md
```

## Quickstart

```bash
git clone https://github.com/<your-username>/credit-risk-loan-analytics.git
cd credit-risk-loan-analytics
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Build the governed database from the raw CSVs
python src/data_load.py

# 2. Train and compare all 4 classifiers, persist the champion
python src/train.py

# 3. Score every customer 0-100 and bucket into risk categories
python src/risk_score.py

# 4. Launch the dashboard
streamlit run dashboard/app.py
```

Fixed random seeds (`random_state=42`) throughout — re-running the pipeline reproduces the
reported metrics exactly.

The default engine is **SQLite** (`data/credit_risk.db`) for zero-setup, portable local
development. `sql/credit_risk_platform.sql` targets **PostgreSQL 14+** for a production
deployment — same schema, same constraints; only the load path differs (`\copy` in psql vs.
`src/data_load.py`'s `pandas.to_sql`).

## Notebooks vs. src/

The four notebooks are the narrative — read them in order for the analysis story (EDA →
features → modelling → scoring), each with charts and commentary. They call into `src/` for
the actual logic, so the notebooks and the CLI pipeline (`python src/train.py`, etc.) always
agree — there's exactly one implementation of each step, not two.

## Known limitations

- ROC-AUC (0.704) and recall (0.648) fall short of the PRD's 0.80 / 0.70 targets — discussed
  in full in the model card, along with what would close the gap.
- Synthetic dataset; a fictional lending business. The methodology, not the absolute
  metrics, is the deliverable.
- No production API, MLOps pipeline, automated pricing, or fairness/bias review — explicitly
  out of scope for v1 (see PRD §3, non-goals).

## Tech stack

PostgreSQL (production) / SQLite (dev) · Python 3.10+ · pandas, NumPy · scikit-learn, XGBoost
· SHAP · matplotlib, Seaborn · Streamlit + Plotly · Jupyter · Git

## License

Internal project for the Internmo Applied Data Science internship track. Not for production
lending use.
