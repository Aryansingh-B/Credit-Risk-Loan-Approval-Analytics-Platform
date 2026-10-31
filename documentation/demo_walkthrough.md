# Demo Walkthrough (about 6 minutes)

## 1. The problem (30 s)
Underwriters decide on a loan with a bureau score and judgement. Credora wants a second opinion: *if this loan is disbursed, how likely is default?* - explainable, and advisory only.

## 2. The data and the database (45 s)
- Four tables: 5,000 customers, 5,000 bureau records, 7,500 applications, 40,079 transactions; 0 orphan foreign keys.
- `sql/credit_risk_platform.sql` runs on PostgreSQL 16; the pipeline uses SQLite for zero-setup. Ten analytical queries all run in milliseconds.
- Show `notebooks/01_eda.ipynb`: default rate falls from ~40% (score < 580) to ~12% (750+).

## 3. Doing it carefully (90 s) - what makes this credible
- **No leakage**: decision-time fields excluded; transactions aggregated to one row per customer before the join.
- **No PII / protected attributes** in the model.
- **Grouped split**: a customer's loans never straddle train and test.
- **Tuned and calibrated**: 4 algorithms tuned with grouped CV; champion = Logistic Regression; probabilities calibrated (0.45 -> 0.24 vs actual 0.23).

## 4. The honest result (60 s)
- ROC-AUC **0.716** vs a 0.80 target; recall **0.688** vs 0.70 (95% CI 0.61-0.75).
- Four very different algorithms land within 0.690-0.719: the ceiling is the data.
- The bands still rank risk correctly on unseen loans: 6.6% -> 21.3% -> 36.2%.

## 5. Dashboard tour (2 min) - `streamlit run dashboard/app.py`
1. KPI row (7,500 applications, 46.4% approval of decided, 23.4% default, Rs 441.2 crore).
2. Approval trend and default-by-loan-type -> Home (~32%) is riskiest, Personal (~20%) safest.
3. Risk donut -> toggle *Score basis* to *Disbursed loan* to show the trustworthy view.
4. Credit-score trend/histogram, city map, monthly volume, approval-vs-rejection stack.
5. Watchlist -> filter to High Risk + one state, read a customer's `top_risk_drivers`, download CSV.
6. Bottom tab: SHAP drivers and the band-validation table.

## 6. Limits and next steps (45 s)
Longer transaction history, bureau time series and more disbursed loans (the real lever); out-of-time validation; fairness review; drift monitoring.

## Likely questions
- **Why is ROC-AUC below target?** Tuned linear, tree, forest and boosting all agree; signal in the data is the limit. We reported it rather than tuning on the test set.
- **Why did the recall target move?** We chose the threshold on training out-of-fold predictions, not on the test set; test recall 0.688 is within noise of 0.70 (n=160 defaulters).
- **Why are so many customers High Risk?** Customers whose application was never disbursed (mostly rejected) are extrapolations; filter by score basis.
- **Why calibrate?** Balanced class weights inflate probabilities; a 0-100 score should mean what it says.
- **Is it fair?** Protected attributes are excluded, but no formal bias review was done (PRD non-goal).
