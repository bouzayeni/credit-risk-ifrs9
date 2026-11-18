-- Star schema for the loan book. Run by credit_risk.warehouse.load as credit_etl, inside the
-- same transaction as the data load, so readers see either the old or the new warehouse.
-- Grain of dw.fact_loan: one row per loan, outcome as of the 2019-03-31 snapshot.

DROP VIEW IF EXISTS bi.vintage_default_12m, bi.open_book CASCADE;
DROP TABLE IF EXISTS dw.fact_loan, dw.dim_month, dw.dim_grade, dw.dim_purpose, dw.dim_state,
    dw.dim_status CASCADE;

CREATE TABLE dw.dim_month (
    month_key     integer  PRIMARY KEY,          -- yyyymm
    month_start   date     NOT NULL UNIQUE,
    year          smallint NOT NULL,
    quarter       smallint NOT NULL CHECK (quarter BETWEEN 1 AND 4),
    month         smallint NOT NULL CHECK (month BETWEEN 1 AND 12),
    month_name    text     NOT NULL,
    year_quarter  text     NOT NULL,
    CHECK (month_key = year * 100 + month)
);
COMMENT ON TABLE dw.dim_month IS 'Calendar months covering issue, last payment and default months.';

CREATE TABLE dw.dim_grade (
    grade_key       smallint PRIMARY KEY,
    sub_grade       text     NOT NULL UNIQUE,
    grade           text     NOT NULL CHECK (grade IN ('A','B','C','D','E','F','G')),
    grade_rank      smallint NOT NULL,           -- A = 1 (lowest risk)
    sub_grade_rank  smallint NOT NULL UNIQUE     -- A1 = 1 ... G5 = 35
);
COMMENT ON TABLE dw.dim_grade IS 'Lending Club risk grade at origination (benchmark, not a scorecard input).';

CREATE TABLE dw.dim_purpose (
    purpose_key    smallint PRIMARY KEY,
    purpose        text NOT NULL UNIQUE,
    purpose_label  text NOT NULL
);

CREATE TABLE dw.dim_state (
    state_key   smallint PRIMARY KEY,
    state_code  char(2) NOT NULL UNIQUE,
    state_name  text    NOT NULL,
    region      text    NOT NULL CHECK (region IN ('Northeast','Midwest','South','West'))
);
COMMENT ON TABLE dw.dim_state IS 'Borrower US state, with US Census region.';

CREATE TABLE dw.dim_status (
    status_key         smallint PRIMARY KEY,
    loan_status        text     NOT NULL UNIQUE,
    status_group       text     NOT NULL CHECK (status_group IN ('Performing','Delinquent','Defaulted','Repaid')),
    days_past_due      text     NOT NULL,
    is_open            boolean  NOT NULL,
    is_default         boolean  NOT NULL,
    is_default_strict  boolean  NOT NULL,
    sort_order         smallint NOT NULL UNIQUE
);
COMMENT ON TABLE dw.dim_status IS 'Loan status at the snapshot and how the project classifies it.';

CREATE TABLE dw.fact_loan (
    loan_id                  bigint   PRIMARY KEY,
    issue_month_key          integer  NOT NULL REFERENCES dw.dim_month,
    grade_key                smallint NOT NULL REFERENCES dw.dim_grade,
    purpose_key              smallint NOT NULL REFERENCES dw.dim_purpose,
    state_key                smallint NOT NULL REFERENCES dw.dim_state,
    status_key               smallint NOT NULL REFERENCES dw.dim_status,
    last_payment_month_key   integer  REFERENCES dw.dim_month,
    default_month_key        integer  REFERENCES dw.dim_month,
    -- Contract and pricing
    term_months              smallint      NOT NULL CHECK (term_months IN (36, 60)),
    funded_amnt              numeric(12,2) NOT NULL CHECK (funded_amnt > 0),
    int_rate                 numeric(5,2)  NOT NULL CHECK (int_rate > 0),
    installment              numeric(10,2) NOT NULL CHECK (installment > 0),
    -- Borrower at application
    annual_inc               numeric(14,2) CHECK (annual_inc > 0),
    dti                      numeric(7,2)  CHECK (dti >= 0),
    fico                     numeric(5,1)  CHECK (fico BETWEEN 300 AND 850),
    emp_length_years         smallint      CHECK (emp_length_years BETWEEN 0 AND 10),
    home_ownership           text          NOT NULL,
    verification_status      text          NOT NULL,
    is_joint                 boolean       NOT NULL,
    -- Performance (post-origination: never a model input)
    last_fico                numeric(5,1)  CHECK (last_fico BETWEEN 300 AND 850),
    out_prncp                numeric(12,2) NOT NULL CHECK (out_prncp >= 0),
    total_pymnt              numeric(12,2) NOT NULL CHECK (total_pymnt >= 0),
    total_rec_prncp          numeric(12,2) NOT NULL CHECK (total_rec_prncp >= 0),
    total_rec_int            numeric(12,2) NOT NULL CHECK (total_rec_int >= 0),
    total_rec_late_fee       numeric(12,2) NOT NULL CHECK (total_rec_late_fee >= 0),
    recoveries               numeric(12,2) NOT NULL CHECK (recoveries >= 0),
    collection_recovery_fee  numeric(12,2) NOT NULL CHECK (collection_recovery_fee >= 0),
    hardship_flag            boolean       NOT NULL,
    debt_settlement_flag     boolean       NOT NULL,
    -- Outcome
    is_default               boolean  NOT NULL,
    is_default_strict        boolean  NOT NULL,
    is_closed                boolean  NOT NULL,
    months_to_default        smallint CHECK (months_to_default >= 1),
    eligible_12m             boolean  NOT NULL,
    default_12m              boolean  NOT NULL,
    default_12m_strict       boolean  NOT NULL,
    months_observed          smallint NOT NULL CHECK (months_observed >= 0),
    CHECK ((months_to_default IS NOT NULL) = is_default),
    CHECK ((default_month_key IS NOT NULL) = is_default),
    CHECK (NOT default_12m OR is_default),
    CHECK (NOT default_12m_strict OR default_12m),
    CHECK (NOT is_default_strict OR is_default)
);
COMMENT ON TABLE dw.fact_loan IS 'One row per loan; status and amounts as of 2019-03-31.';
COMMENT ON COLUMN dw.fact_loan.default_12m IS 'First missed payment within 12 months of issue. Use only where eligible_12m.';
COMMENT ON COLUMN dw.fact_loan.months_to_default IS 'Months from issue to the first missed payment of a defaulted loan.';
