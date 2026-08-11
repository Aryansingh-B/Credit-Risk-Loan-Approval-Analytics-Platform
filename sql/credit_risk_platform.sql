-- =====================================================================
-- Credora Finance | Credit Risk & Loan Approval Analytics Platform
-- Relational schema, seed data, indexes, constraints and analytical queries
-- Target engine: PostgreSQL 14+  (MySQL 8+ compatible with minor edits)
-- Owner: Data & Analytics Team, Credora Finance
-- =====================================================================

-- ---------------------------------------------------------------------
-- 0. Clean start (safe re-run)
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS transactions      CASCADE;
DROP TABLE IF EXISTS loan_applications CASCADE;
DROP TABLE IF EXISTS credit_history    CASCADE;
DROP TABLE IF EXISTS customers         CASCADE;

-- ---------------------------------------------------------------------
-- 1. DDL : CUSTOMERS  (parent table)
-- ---------------------------------------------------------------------
CREATE TABLE customers (
    customer_id         VARCHAR(12)   PRIMARY KEY,
    first_name          VARCHAR(50)   NOT NULL,
    last_name           VARCHAR(50)   NOT NULL,
    gender              VARCHAR(10)   CHECK (gender IN ('Male','Female','Other')),
    date_of_birth       DATE          NOT NULL,
    age                 SMALLINT      CHECK (age BETWEEN 18 AND 100),
    marital_status      VARCHAR(15),
    dependents          SMALLINT      DEFAULT 0 CHECK (dependents >= 0),
    education           VARCHAR(30),
    employment_type     VARCHAR(20),
    occupation          VARCHAR(60),
    annual_income       NUMERIC(12,2) CHECK (annual_income >= 0),
    city                VARCHAR(50),
    state               VARCHAR(50),
    residence_type      VARCHAR(20),
    years_at_residence  SMALLINT      CHECK (years_at_residence >= 0),
    email               VARCHAR(120)  UNIQUE,
    phone               VARCHAR(20),
    kyc_status          VARCHAR(15)   DEFAULT 'Pending'
                                      CHECK (kyc_status IN ('Verified','Pending','Rejected')),
    customer_since      DATE          NOT NULL
);

-- ---------------------------------------------------------------------
-- 2. DDL : CREDIT_HISTORY  (1:1 with customer, bureau snapshot)
-- ---------------------------------------------------------------------
CREATE TABLE credit_history (
    credit_id                 VARCHAR(12)   PRIMARY KEY,
    customer_id               VARCHAR(12)   NOT NULL UNIQUE,
    as_of_date                DATE          NOT NULL,
    credit_score              SMALLINT      CHECK (credit_score BETWEEN 300 AND 900),
    num_open_accounts         SMALLINT      CHECK (num_open_accounts >= 0),
    num_credit_inquiries_6m   SMALLINT      CHECK (num_credit_inquiries_6m >= 0),
    credit_utilization_ratio  NUMERIC(4,3)  CHECK (credit_utilization_ratio BETWEEN 0 AND 1),
    num_late_payments_30d     SMALLINT      DEFAULT 0,
    num_late_payments_90d     SMALLINT      DEFAULT 0,
    num_defaults_prior        SMALLINT      DEFAULT 0,
    bankruptcies              SMALLINT      DEFAULT 0,
    total_credit_limit        NUMERIC(12,2) CHECK (total_credit_limit >= 0),
    total_outstanding_debt    NUMERIC(12,2) CHECK (total_outstanding_debt >= 0),
    oldest_account_age_months SMALLINT      CHECK (oldest_account_age_months >= 0),
    CONSTRAINT fk_credit_customer
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
);

-- ---------------------------------------------------------------------
-- 3. DDL : LOAN_APPLICATIONS  (many:1 with customer, holds ML target)
-- ---------------------------------------------------------------------
CREATE TABLE loan_applications (
    application_id           VARCHAR(12)   PRIMARY KEY,
    customer_id              VARCHAR(12)   NOT NULL,
    application_date         DATE          NOT NULL,
    loan_type                VARCHAR(20)   NOT NULL,
    loan_purpose             VARCHAR(50),
    requested_amount         NUMERIC(12,2) CHECK (requested_amount > 0),
    loan_term_months         SMALLINT      CHECK (loan_term_months > 0),
    interest_rate            NUMERIC(5,2)  CHECK (interest_rate >= 0),
    applicant_monthly_income NUMERIC(12,2) CHECK (applicant_monthly_income >= 0),
    existing_emi             NUMERIC(12,2) DEFAULT 0,
    debt_to_income_ratio     NUMERIC(5,3)  CHECK (debt_to_income_ratio >= 0),
    collateral_flag          SMALLINT      DEFAULT 0 CHECK (collateral_flag IN (0,1)),
    approval_status          VARCHAR(15)   NOT NULL
                                           CHECK (approval_status IN ('Approved','Rejected','Under Review')),
    approved_amount          NUMERIC(12,2) DEFAULT 0,
    decision_date            DATE,
    disbursed_flag           SMALLINT      DEFAULT 0 CHECK (disbursed_flag IN (0,1)),
    loan_default             SMALLINT      CHECK (loan_default IN (0,1)),   -- NULL for non-disbursed
    CONSTRAINT fk_loan_customer
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE,
    CONSTRAINT chk_decision_after_application
        CHECK (decision_date IS NULL OR decision_date >= application_date)
);

-- ---------------------------------------------------------------------
-- 4. DDL : TRANSACTIONS  (many:1 with customer, behavioural signal)
-- ---------------------------------------------------------------------
CREATE TABLE transactions (
    transaction_id     VARCHAR(14)   PRIMARY KEY,
    customer_id        VARCHAR(12)   NOT NULL,
    transaction_date   DATE          NOT NULL,
    transaction_type   VARCHAR(10)   CHECK (transaction_type IN ('Credit','Debit')),
    category           VARCHAR(30),
    channel            VARCHAR(20),
    amount             NUMERIC(12,2) CHECK (amount >= 0),
    balance_after      NUMERIC(12,2),
    CONSTRAINT fk_txn_customer
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
);

-- ---------------------------------------------------------------------
-- 5. INDEXES  (query-path optimisation for analytics & joins)
-- ---------------------------------------------------------------------
CREATE INDEX idx_credit_customer      ON credit_history (customer_id);
CREATE INDEX idx_credit_score         ON credit_history (credit_score);
CREATE INDEX idx_loan_customer        ON loan_applications (customer_id);
CREATE INDEX idx_loan_status          ON loan_applications (approval_status);
CREATE INDEX idx_loan_appdate         ON loan_applications (application_date);
CREATE INDEX idx_loan_default         ON loan_applications (loan_default);
CREATE INDEX idx_txn_customer         ON transactions (customer_id);
CREATE INDEX idx_txn_date             ON transactions (transaction_date);
CREATE INDEX idx_txn_category         ON transactions (category);

-- =====================================================================
-- 6. SEED DATA  (representative rows; full dataset loads from /datasets)
--    Bulk load in PostgreSQL:
--      \copy customers        FROM 'datasets/customers.csv'         CSV HEADER;
--      \copy credit_history   FROM 'datasets/credit_history.csv'    CSV HEADER;
--      \copy loan_applications FROM 'datasets/loan_applications.csv' CSV HEADER;
--      \copy transactions     FROM 'datasets/transactions.csv'      CSV HEADER;
-- =====================================================================

INSERT INTO customers
(customer_id,first_name,last_name,gender,date_of_birth,age,marital_status,dependents,
 education,employment_type,occupation,annual_income,city,state,residence_type,
 years_at_residence,email,phone,kyc_status,customer_since) VALUES
('CUST100001', 'Kabir', 'Kulkarni', 'Male', '2004-09-09', 21, 'Married', 1, 'Diploma', 'Salaried', 'Software Engineer', 316400, 'Noida', 'Uttar Pradesh', 'Family Owned', 3, 'kabir.kulkarni614@outlook.com', '+91-8574623268', 'Verified', '2023-06-16'),
('CUST100002', 'Manish', 'Gupta', 'Male', '1997-03-06', 29, 'Married', 1, 'Graduate', 'Unemployed', 'Student', 120000, 'Mysuru', 'Karnataka', 'Mortgaged', 11, 'manish.gupta34@gmail.com', '+91-9252972703', 'Verified', '2014-08-15'),
('CUST100003', 'Sandeep', 'Mehta', 'Male', '2001-08-02', 24, 'Married', 0, 'Diploma', 'Salaried', 'Nurse', 175800, 'Kochi', 'Kerala', 'Owned', 6, 'sandeep.mehta434@outlook.com', '+91-9113744447', 'Pending', '2023-04-29'),
('CUST100004', 'Kabir', 'Reddy', 'Male', '1976-03-30', 50, 'Single', 3, 'Graduate', 'Retired', 'Retired Private Sector', 275100, 'Bengaluru', 'Karnataka', 'Rented', 24, 'kabir.reddy27@gmail.com', '+91-7697018056', 'Verified', '2016-11-03'),
('CUST100005', 'Saanvi', 'Patel', 'Female', '1993-04-05', 33, 'Married', 1, 'Diploma', 'Self-Employed', 'Doctor (Practice)', 384700, 'Pune', 'Maharashtra', 'Owned', 7, 'saanvi.patel597@outlook.com', '+91-7148556853', 'Verified', '2017-09-24'),
('CUST100006', 'Deepak', 'Kapoor', 'Male', '1973-01-03', 53, 'Married', 1, 'Graduate', 'Self-Employed', 'Architect', 739600, 'Hyderabad', 'Telangana', 'Rented', 24, 'deepak.kapoor22@gmail.com', '+91-8265448136', 'Verified', '2015-08-05');

INSERT INTO credit_history
(credit_id,customer_id,as_of_date,credit_score,num_open_accounts,num_credit_inquiries_6m,
 credit_utilization_ratio,num_late_payments_30d,num_late_payments_90d,num_defaults_prior,
 bankruptcies,total_credit_limit,total_outstanding_debt,oldest_account_age_months) VALUES
('CR500001', 'CUST100001', '2024-06-30', 631, 4, 1, 0.707, 2, 0, 0, 0, 184000, 130100, 6),
('CR500002', 'CUST100002', '2024-06-30', 429, 5, 1, 0.748, 3, 2, 1, 1, 58000, 43400, 66),
('CR500003', 'CUST100003', '2024-06-30', 503, 2, 4, 0.658, 2, 2, 0, 0, 128000, 84200, 30),
('CR500004', 'CUST100004', '2024-06-30', 576, 5, 4, 0.373, 0, 0, 0, 0, 218000, 81300, 344),
('CR500005', 'CUST100005', '2024-06-30', 681, 6, 1, 0.155, 1, 1, 0, 0, 91000, 14100, 105),
('CR500006', 'CUST100006', '2024-06-30', 701, 6, 0, 0.583, 0, 0, 0, 0, 413000, 240800, 340);

INSERT INTO loan_applications
(application_id,customer_id,application_date,loan_type,loan_purpose,requested_amount,
 loan_term_months,interest_rate,applicant_monthly_income,existing_emi,debt_to_income_ratio,
 collateral_flag,approval_status,approved_amount,decision_date,disbursed_flag,loan_default) VALUES
('APP800001', 'CUST101595', '2022-12-20', 'Business', 'Equipment Purchase', 6297000, 84, 15.25, 48400, 11600, 2.769, 0, 'Rejected', 0, '2022-12-27', 0, NULL),
('APP800002', 'CUST104584', '2022-12-15', 'Home', 'Balance Transfer', 6050000, 24, 10.07, 149200, 21700, 2.017, 1, 'Rejected', 0, '2022-12-31', 0, NULL),
('APP800003', 'CUST103833', '2022-10-05', 'Business', 'Equipment Purchase', 6441000, 48, 17.51, 88500, 13900, 2.277, 0, 'Rejected', 0, '2022-10-19', 0, NULL),
('APP800004', 'CUST103841', '2022-08-29', 'Auto', 'Used Car', 2498000, 24, 9.79, 42300, 14400, 3.0, 1, 'Rejected', 0, '2022-09-09', 0, NULL),
('APP800005', 'CUST100115', '2024-05-22', 'Gold', 'Gold Loan', 393000, 24, 12.86, 47300, 5200, 0.505, 1, 'Under Review', 0, NULL, 0, NULL),
('APP800006', 'CUST102505', '2022-07-22', 'Personal', 'Wedding', 838000, 84, 19.18, 35200, 4300, 0.64, 0, 'Approved', 590000, '2022-08-08', 1, 0.0);

INSERT INTO transactions
(transaction_id,customer_id,transaction_date,transaction_type,category,channel,amount,balance_after) VALUES
('TXN1004290', 'CUST100532', '2024-01-01', 'Debit', 'Dining', 'Net Banking', 1446.39, 3787.47),
('TXN1024738', 'CUST103082', '2024-01-01', 'Credit', 'Salary', 'Debit Card', 42535.79, 133222.97),
('TXN1029304', 'CUST103653', '2024-01-01', 'Debit', 'Travel', 'Net Banking', 6418.24, 149138.52),
('TXN1021143', 'CUST102626', '2024-01-01', 'Debit', 'ATM Withdrawal', 'Credit Card', 7319.62, 122052.09),
('TXN1018214', 'CUST102262', '2024-01-01', 'Debit', 'EMI', 'UPI', 3950.95, 21520.3),
('TXN1037813', 'CUST104718', '2024-01-01', 'Debit', 'ATM Withdrawal', 'UPI', 22015.5, 297020.35);

-- =====================================================================
-- 7. ANALYTICAL QUERIES
-- =====================================================================

-- 7.1  Credit score analysis: distribution across standard risk bands
SELECT
    CASE
        WHEN credit_score >= 750 THEN 'Excellent (750-900)'
        WHEN credit_score >= 700 THEN 'Good (700-749)'
        WHEN credit_score >= 650 THEN 'Fair (650-699)'
        WHEN credit_score >= 580 THEN 'Poor (580-649)'
        ELSE 'Very Poor (300-579)'
    END                                   AS credit_band,
    COUNT(*)                              AS customers,
    ROUND(AVG(credit_score),0)            AS avg_score,
    ROUND(AVG(credit_utilization_ratio),3) AS avg_utilization,
    ROUND(AVG(num_late_payments_30d),2)   AS avg_late_30d
FROM credit_history
GROUP BY 1
ORDER BY MIN(credit_score) DESC;

-- 7.2  Loan approval trends: monthly approval rate over time
SELECT
    DATE_TRUNC('month', application_date)::date       AS month,
    COUNT(*)                                          AS applications,
    SUM(CASE WHEN approval_status='Approved' THEN 1 ELSE 0 END) AS approved,
    ROUND(100.0 * SUM(CASE WHEN approval_status='Approved' THEN 1 ELSE 0 END)
          / NULLIF(COUNT(*),0), 1)                    AS approval_rate_pct
FROM loan_applications
GROUP BY 1
ORDER BY 1;

-- 7.3  Customer segmentation: income band x employment, with approval behaviour
SELECT
    c.employment_type,
    CASE
        WHEN c.annual_income >= 1500000 THEN 'High (15L+)'
        WHEN c.annual_income >=  600000 THEN 'Mid (6-15L)'
        ELSE 'Entry (<6L)'
    END                                              AS income_band,
    COUNT(DISTINCT c.customer_id)                    AS customers,
    ROUND(AVG(ch.credit_score),0)                    AS avg_credit_score,
    ROUND(100.0*AVG(CASE WHEN la.approval_status='Approved' THEN 1.0 ELSE 0 END),1) AS approval_rate_pct
FROM customers c
JOIN credit_history ch    ON ch.customer_id = c.customer_id
LEFT JOIN loan_applications la ON la.customer_id = c.customer_id
GROUP BY 1,2
ORDER BY 1,2;

-- 7.4  High-risk customer identification (rule-based watchlist)
SELECT
    c.customer_id, c.first_name, c.last_name, c.city,
    ch.credit_score, ch.credit_utilization_ratio,
    ch.num_late_payments_90d, ch.num_defaults_prior
FROM customers c
JOIN credit_history ch ON ch.customer_id = c.customer_id
WHERE ch.credit_score < 600
   OR ch.credit_utilization_ratio > 0.80
   OR ch.num_defaults_prior >= 1
   OR ch.num_late_payments_90d >= 2
ORDER BY ch.credit_score ASC
LIMIT 100;

-- 7.5  Default rate analysis by loan type (disbursed loans only)
SELECT
    loan_type,
    COUNT(*)                                    AS disbursed_loans,
    SUM(loan_default)                           AS defaults,
    ROUND(100.0*AVG(loan_default),2)            AS default_rate_pct,
    ROUND(AVG(interest_rate),2)                 AS avg_interest_rate
FROM loan_applications
WHERE disbursed_flag = 1 AND loan_default IS NOT NULL
GROUP BY loan_type
ORDER BY default_rate_pct DESC;

-- 7.6  Loan amount distribution by loan type
SELECT
    loan_type,
    COUNT(*)                                        AS applications,
    ROUND(MIN(requested_amount),0)                  AS min_amount,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY requested_amount),0) AS median_amount,
    ROUND(AVG(requested_amount),0)                  AS avg_amount,
    ROUND(MAX(requested_amount),0)                  AS max_amount
FROM loan_applications
GROUP BY loan_type
ORDER BY avg_amount DESC;

-- 7.7  Monthly approval statistics with disbursed value
SELECT
    TO_CHAR(application_date,'YYYY-MM')             AS month,
    COUNT(*)                                        AS applications,
    SUM(CASE WHEN approval_status='Approved' THEN 1 ELSE 0 END) AS approved,
    SUM(CASE WHEN approval_status='Rejected' THEN 1 ELSE 0 END) AS rejected,
    ROUND(SUM(approved_amount)/1000000.0,2)         AS disbursed_amount_mn
FROM loan_applications
GROUP BY 1
ORDER BY 1;

-- 7.8  Risk categorisation blending bureau score and utilisation
SELECT
    risk_category, COUNT(*) AS customers,
    ROUND(AVG(credit_score),0) AS avg_score
FROM (
    SELECT customer_id, credit_score,
        CASE
            WHEN credit_score >= 700 AND credit_utilization_ratio < 0.4 THEN 'Low Risk'
            WHEN credit_score >= 620                                     THEN 'Medium Risk'
            ELSE 'High Risk'
        END AS risk_category
    FROM credit_history
) t
GROUP BY risk_category
ORDER BY MIN(credit_score) DESC;

-- 7.9  Transaction behaviour analysis: 6-month cash-flow profile per customer
SELECT
    t.customer_id,
    COUNT(*)                                                   AS txn_count,
    SUM(CASE WHEN transaction_type='Credit' THEN amount ELSE 0 END) AS total_inflow,
    SUM(CASE WHEN transaction_type='Debit'  THEN amount ELSE 0 END) AS total_outflow,
    SUM(CASE WHEN category='EMI' THEN amount ELSE 0 END)       AS emi_outflow,
    ROUND(AVG(balance_after),0)                                AS avg_balance
FROM transactions t
GROUP BY t.customer_id
ORDER BY total_outflow DESC
LIMIT 50;

-- 7.10  Top predictors of loan default: default rate sliced by driver
--        (feeds feature-selection / EDA before modelling)
SELECT 'credit_score<600' AS driver,
       ROUND(100.0*AVG(la.loan_default),2) AS default_rate_pct, COUNT(*) AS n
FROM loan_applications la JOIN credit_history ch USING (customer_id)
WHERE la.loan_default IS NOT NULL AND ch.credit_score < 600
UNION ALL
SELECT 'dti>0.5',
       ROUND(100.0*AVG(loan_default),2), COUNT(*)
FROM loan_applications WHERE loan_default IS NOT NULL AND debt_to_income_ratio > 0.5
UNION ALL
SELECT 'prior_default>=1',
       ROUND(100.0*AVG(la.loan_default),2), COUNT(*)
FROM loan_applications la JOIN credit_history ch USING (customer_id)
WHERE la.loan_default IS NOT NULL AND ch.num_defaults_prior >= 1
UNION ALL
SELECT 'utilization>0.8',
       ROUND(100.0*AVG(la.loan_default),2), COUNT(*)
FROM loan_applications la JOIN credit_history ch USING (customer_id)
WHERE la.loan_default IS NOT NULL AND ch.credit_utilization_ratio > 0.8
ORDER BY default_rate_pct DESC;

