-- Model results (Phases 4-7) for reporting. Run by credit_risk.warehouse.load_results as
-- credit_etl in one transaction, after credit_risk.warehouse.load (re-running that load drops
-- these tables' foreign keys and views, so re-run this afterwards).
-- Rates and PDs are fractions (0.058 = 5.8%); amounts are USD.

DROP VIEW IF EXISTS bi.ecl_loan, bi.ecl_by_stage, bi.rating_scale, bi.pd_term_structure,
    bi.ecl_scenario, bi.monitoring_quarter, bi.monitoring_feature CASCADE;
DROP TABLE IF EXISTS dw.fact_ecl, dw.fact_pd_curve, dw.dim_rating, dw.ecl_scenario,
    dw.monitoring_quarter, dw.monitoring_feature CASCADE;

CREATE TABLE dw.dim_rating (
    rating          text     PRIMARY KEY CHECK (rating ~ '^R([1-9]|10)$'),
    rating_rank     smallint NOT NULL UNIQUE,              -- R1 = 1 (lowest risk)
    pd_lower        numeric(7,5) NOT NULL,
    pd_upper        numeric(7,5),                          -- NULL for R10
    pd_12m          numeric(7,5) NOT NULL,                 -- point-in-time grade PD (Phase 4)
    pd_12m_upside   numeric(7,5) NOT NULL,
    pd_12m_downside numeric(7,5) NOT NULL,
    lgd             numeric(7,5) NOT NULL                  -- discounted LGD (Phase 5)
);
COMMENT ON TABLE dw.dim_rating IS 'R1-R10 master scale with 12-month PDs by scenario and LGD.';

CREATE TABLE dw.fact_ecl (
    loan_id              bigint   PRIMARY KEY REFERENCES dw.fact_loan,
    stage                smallint NOT NULL CHECK (stage IN (1, 2, 3)),
    rating_orig          text     NOT NULL REFERENCES dw.dim_rating,
    rating_now           text     NOT NULL REFERENCES dw.dim_rating,
    sicr_delinquent      boolean  NOT NULL,
    sicr_hardship        boolean  NOT NULL,
    sicr_pd              boolean  NOT NULL,
    pd_orig              numeric(8,6) NOT NULL,
    pd_now               numeric(8,6) NOT NULL,
    fico_change          numeric(6,1),
    gross_carrying_amt   numeric(12,2) NOT NULL CHECK (gross_carrying_amt > 0),
    ecl                  numeric(12,2) NOT NULL CHECK (ecl >= 0),
    ecl_upside           numeric(12,2) NOT NULL,
    ecl_base             numeric(12,2) NOT NULL,
    ecl_downside         numeric(12,2) NOT NULL,
    ecl_12m_base         numeric(12,2) NOT NULL,
    ecl_lifetime_base    numeric(12,2) NOT NULL,
    pd_12m_fwd_base      numeric(8,6) NOT NULL,
    pd_lifetime_fwd_base numeric(8,6) NOT NULL,
    CHECK (ecl <= gross_carrying_amt)
);
COMMENT ON TABLE dw.fact_ecl IS 'IFRS 9 ECL per open loan at 2019-03-31 (Phase 6); ecl = probability-weighted.';

CREATE TABLE dw.fact_pd_curve (
    rating          text     NOT NULL REFERENCES dw.dim_rating,
    term_months     smallint NOT NULL CHECK (term_months IN (36, 60)),
    mob             smallint NOT NULL CHECK (mob >= 1),
    hazard_default  numeric(9,7) NOT NULL,
    hazard_repaid   numeric(9,7) NOT NULL,
    marginal_pd     numeric(9,7) NOT NULL,
    cumulative_pd   numeric(9,7) NOT NULL,
    survival        numeric(9,7) NOT NULL,
    borrowed        boolean  NOT NULL,
    PRIMARY KEY (rating, term_months, mob)
);
COMMENT ON TABLE dw.fact_pd_curve IS 'Lifetime PD curves by rating and term (Phase 4), anchored to the grade PD.';

CREATE TABLE dw.ecl_scenario (
    scenario      text     PRIMARY KEY,
    sort_order    smallint NOT NULL UNIQUE,
    factor_shift  numeric(4,2),                            -- NULL for the weighted total
    weight        numeric(4,2) NOT NULL,
    ecl           numeric(14,2) NOT NULL
);

CREATE TABLE dw.monitoring_quarter (
    issue_quarter       text     PRIMARY KEY,
    quarter_start       date     NOT NULL UNIQUE,
    loans               integer  NOT NULL,
    mean_score          numeric(6,2) NOT NULL,
    psi_score           numeric(7,4) NOT NULL,
    psi_status          text     NOT NULL,
    max_csi             numeric(7,4) NOT NULL,
    max_csi_feature     text     NOT NULL,
    gini                numeric(6,4),
    gini_se             numeric(6,4),
    gini_status         text,
    observed_12m        numeric(7,5),
    scorecard_pd        numeric(7,5),
    scorecard_ratio     numeric(6,3),
    scorecard_status    text,
    production_pd       numeric(7,5),
    production_ratio    numeric(6,3),
    production_status   text,
    gini_6m             numeric(6,4),
    early_observed      numeric(7,5),
    early_expected      numeric(7,5),
    early_ratio         numeric(6,3),
    early_status        text
);
COMMENT ON TABLE dw.monitoring_quarter IS 'Scorecard and PD monitoring by issue quarter (Phase 7).';

CREATE TABLE dw.monitoring_feature (
    issue_quarter  text NOT NULL REFERENCES dw.monitoring_quarter,
    feature        text NOT NULL,
    csi            numeric(7,4) NOT NULL,
    status         text NOT NULL,
    PRIMARY KEY (issue_quarter, feature)
);
COMMENT ON TABLE dw.monitoring_feature IS 'Characteristic stability index of each scorecard feature by issue quarter.';
