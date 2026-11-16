# Credora Finance | Credit Risk & Loan Approval Analytics Platform

> **Turning Lending Data into Risk Intelligence**
>
> An end-to-end credit risk analytics platform built using **Python, SQL, Machine Learning, and Streamlit** to support loan approval and default-risk assessment. The platform integrates **5,000 customers, 7,500 loan applications, and 40,079 transactions**, delivering a 0–100 customer risk score and an interactive dashboard. The champion Logistic Regression model achieved **0.704 ROC-AUC** and **0.648 recall** on held-out test data.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![SQL](https://img.shields.io/badge/SQL-PostgreSQL%20%7C%20SQLite-336791)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--learn-orange)
![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-red)
![Status](https://img.shields.io/badge/Project-Complete-brightgreen)

**Quick Links:** [Dashboard](#dashboard-preview) · [Dataset](#dataset-overview) · [Key Findings](#key-insights--findings)

---

## 1. Business Problem

Financial institutions must assess whether a loan applicant is suitable for approval and how likely an approved borrower is to default.

Fragmented customer, credit bureau, loan application, and transaction data can make risk assessment difficult. Poor lending decisions may increase defaults and financial losses, while overly restrictive approvals can limit lending opportunities.

**Business goal:** Build a data-driven decision-support platform that consolidates lending data, predicts default risk, and helps underwriters prioritize applications for review.

## 2. Dataset Overview & Data Dictionary

The project uses a **synthetic retail lending dataset** representing a fictional finance business.

| Dataset | Records | Business Purpose |
|---|---:|---|
| Customers | 5,000 | Customer profile information |
| Credit History | 5,000 | Credit bureau information |
| Loan Applications | 7,500 | Loan application details |
| Transactions | 40,079 | Customer financial activity |

**Total:** 57,579 records across four datasets.

### Key Data Fields

| Field / Feature | Data Type | Business Definition |
|---|---|---|
| Customer ID | Integer | Unique customer identifier |
| Credit History | Structured | Customer credit bureau information |
| Loan Application | Structured | Loan request and application details |
| Transactions | Numeric / Date | Customer financial transaction activity |
| Risk Score | Numeric (0–100) | Model-derived customer risk score |
| Risk Band | Categorical | Low, Medium, or High risk |
| Default Probability | Numeric | Model-estimated probability of default |

*Refer to `documentation/data_dictionary.md` for the detailed field definitions.*

## 3. Project Objectives & Key Business Questions

### Objectives
- Build a governed relational database integrating customer, credit, application, and transaction data.
- Analyze lending data to identify customer and application risk patterns.
- Train and compare classification models for loan-default prediction.
- Develop a 0–100 risk-scoring system and interactive decision-support dashboard.

### Business Questions
1. How can customer, credit, application, and transaction data be consolidated reliably?
2. Which customer features help identify potential loan defaults?
3. How accurately can machine learning classify default risk?
4. How can predicted default probability be translated into an understandable risk score?
5. Which customers fall into Low, Medium, and High risk categories?
6. How can underwriters use risk insights to prioritize application reviews?

## 4. Methodology & Analytical Workflow

| Stage | Work Performed | Tools |
|---|---|---|
| Data Integration | Loaded four datasets and validated relational integrity | Python, Pandas, SQLite |
| Database Design | Created relational schema with primary and foreign key constraints | SQL, PostgreSQL |
| Data Analysis | Performed EDA and executed 10 analytical SQL queries | SQL, Pandas |
| Feature Engineering | Aggregated transactions, created financial ratios, and controlled data leakage | Python, NumPy |
| Model Development | Trained and compared four classification algorithms | Scikit-learn, XGBoost |
| Risk Scoring | Converted predicted default probability into a 0–100 score and risk category | Python |
| Dashboard | Built an 8-component interactive BI dashboard | Streamlit, Plotly |

### Model Performance

Four classifiers were evaluated: Logistic Regression, Decision Tree, Random Forest, and XGBoost.

| Metric | Champion Model |
|---|---:|
| Algorithm | Logistic Regression |
| ROC-AUC | **0.704** |
| Recall | **0.648** |

**Model selection note:** Logistic Regression was selected as the champion model. Its performance falls short of the PRD targets of 0.80 ROC-AUC and 0.70 recall. The model card documents the comparison and improvement opportunities.

## 5. Dashboard Preview & Visual Highlights

The Streamlit dashboard includes **8 components** designed to present lending performance, customer risk, and model outputs.

### Dashboard Overview
![Dashboard Overview](assets/dashboard_overview.png)

*Overview of lending data, portfolio metrics, and risk distribution.*

### Customer Risk Analysis
![Customer Risk Analysis](assets/customer_risk_analysis.png)

*Customer risk scores and Low, Medium, and High risk segments.*

### Model Performance
![Model Performance](assets/model_performance.png)

*Classification model evaluation and performance comparison.*

> Replace the image paths with your actual dashboard screenshots before publishing.

## 6. Key Insights & Findings

- **Data integrity:** Successfully loaded **5,000 customers, 5,000 bureau records, 7,500 applications, and 40,079 transactions**, with **zero orphan foreign keys**.
- **Model performance:** Logistic Regression achieved **0.704 ROC-AUC** and **0.648 recall** on the held-out test set.
- **Model comparison:** Four classifiers were evaluated to select the champion model rather than relying on a single algorithm.
- **Customer risk assessment:** Every customer with an application receives a **0–100 risk score** and a Low, Medium, or High risk classification.
- **Analytical coverage:** Developed **10 SQL analytical queries** to support credit risk and lending analysis.
- **Decision support:** Combined database, predictive modeling, risk scoring, and dashboard outputs into one reproducible workflow.

## 7. Actionable Business Recommendations

| Business Need | Recommendation | Expected Value |
|---|---|---|
| High-risk applications | Prioritize High-risk cases for additional underwriter review | Focus review effort on potentially riskier applications |
| Consistent risk assessment | Use the 0–100 score alongside existing underwriting checks | Support more consistent application reviews |
| Model improvement | Investigate additional features and improve model performance against PRD targets | Improve default-risk identification |
| Data reliability | Maintain database constraints and integrity validation during data loading | Reduce inconsistent or disconnected records |
| Responsible lending | Keep final approval decisions with human underwriters | Avoid fully automated lending decisions |

*These are operational recommendations. No realized reduction in defaults, financial savings, or approval improvement has been measured in this project.*

## 8. Repository Structure & How to Run

```text
credit-risk-loan-analytics/
│
├── datasets/                     # Raw lending CSV datasets
│
├── sql/
│   └── credit_risk_platform.sql  # PostgreSQL schema + 10 SQL queries
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_risk_scoring.ipynb
│
├── src/
│   ├── data_load.py
│   ├── features.py
│   ├── train.py
│   └── risk_score.py
│
├── models/
│   ├── default_model.pkl
│   ├── feature_importance.csv
│   ├── metrics_report.json
│   ├── scored_customers.csv
│   └── model_card.md
│
├── dashboard/
│   └── app.py
│
├── documentation/
│   ├── PRD.docx
│   └── data_dictionary.md
│
├── requirements.txt
└── README.md
