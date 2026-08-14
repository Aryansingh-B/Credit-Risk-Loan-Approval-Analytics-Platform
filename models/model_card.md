# Model Card — Credora Loan-Default Classifier

**Project:** Credit Risk & Loan Approval Analytics Platform
**Owner:** Data & Analytics Team, Credora Finance
**Version:** 1.0

## 1. Model overview

| | |
|---|---|
| **Task** | Binary classification — predict `loan_default` (1 = default, 0 = repaid) |
| **Champion algorithm** | Logistic Regression (`class_weight='balanced'`) |
| **Population** | Disbursed loans only, outcome observed (n = 3,403) |
| **Train / test split** | 80 / 20, stratified on `loan_default`, `random_state=42` |
| **Selection metric** | ROC-AUC (primary), recall (tiebreak) |
| **Artefact** | `models/default_model.pkl` |

## 2. Why Logistic Regression won

Four algorithms were trained and 5-fold cross-validated: Logistic Regression, Decision Tree,
Random Forest, and XGBoost. On this dataset the tree-based ensembles overfit the training
folds and generalised worse to the held-out test set than the regularised linear baseline —
a common outcome on a dataset of this size (~3,400 modelling rows) with a moderate number of
engineered features. Logistic Regression's balanced class weighting also gave it materially
higher recall, which the PRD weights heavily (catching true defaulters matters more than raw
accuracy on an imbalanced book).

## 3. Held-out test performance (all four candidates)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | 5-fold CV ROC-AUC |
|---|---|---|---|---|---|---|
| **Logistic Regression (champion)** | 0.681 | 0.390 | **0.648** | 0.487 | **0.704** | 0.722 ± 0.015 |
| Decision Tree | 0.687 | 0.384 | 0.560 | 0.455 | 0.659 | 0.666 ± 0.011 |
| Random Forest | 0.733 | 0.403 | 0.302 | 0.345 | 0.666 | 0.716 ± 0.013 |
| XGBoost | 0.714 | 0.394 | 0.421 | 0.407 | 0.655 | 0.696 ± 0.024 |

Full numbers, per-model confusion matrices and run metadata: `models/metrics_report.json`.

**Champion confusion matrix** (test set, n = 681):

| | Predicted: No-Default | Predicted: Default |
|---|---|---|
| **Actual: No-Default** | 361 | 161 |
| **Actual: Default** | 56 | 103 |

## 4. Against the PRD success targets

| Target (PRD §16) | Result | Met? |
|---|---|---|
| ROC-AUC ≥ 0.80 | 0.704 | **No** — see §6 |
| Recall on defaulters ≥ 0.70 | 0.648 | **No**, close — see §6 |

These targets are aspirational thresholds set before the data was explored, not requirements
gates. Reporting the shortfall here — rather than tuning the threshold or metric until the
number looks better — is the honest and defensible choice for a model that feeds real
underwriting decisions.

## 5. Feature importance (top 10, |coefficient|)

1. `debt_to_income_ratio`
2. `credit_score`
3. `loan_to_income_ratio`
4. `applicant_monthly_income`
5. `annual_income`
6. `requested_amount`
7. `age`
8. `txn_avg_balance`
9. `collateral_flag`
10. `num_late_payments_30d`

This matches the PRD's expected top predictors (`credit_score`, `debt_to_income_ratio`,
utilisation, prior defaults) and cross-checks against analytical query 7.10 in
`sql/credit_risk_platform.sql`. Full ranking: `models/feature_importance.csv`.

## 6. Known limitations

- **Ceiling on ROC-AUC.** The disbursed population is only 3,403 rows, and the strongest
  behavioural signal (transactions) is a 6-month window aggregated to a handful of summary
  stats. This caps achievable discrimination below the 0.80 target on this dataset. Bringing
  in longer transaction history, bureau time-series (rather than a single snapshot), and a
  larger disbursed sample are the highest-leverage next steps (see PRD §19).
- **Synthetic data.** This is a generated dataset (Credora Finance is a fictional entity for
  the internship project), so absolute metric values should not be read as representative of
  a live lending book — the *pipeline, methodology and evaluation discipline* are the
  deliverable, not a claim about real-world portfolio performance.
- **No fairness/bias review.** Explicitly out of scope for v1 per the PRD; flagged as a future
  enhancement before any production use.
- **Static model.** No retraining or drift monitoring pipeline exists yet (PRD §19).

## 7. Excluded features (leakage control)

`approval_status`, `approved_amount`, `decision_date`, `disbursed_flag` are excluded — these
are only known *after* the underwriting decision the model is meant to support.
`email` / `phone` are excluded per the NFR privacy requirement (no PII in modelling).

## 8. Intended use

Advisory only. Output is a probability and a 0–100 risk score presented to a human
underwriter alongside the existing bureau score and judgement — it does not auto-approve,
auto-reject, or price any loan. Not validated for, and not intended for, real lending
decisions without further work (see §6).
