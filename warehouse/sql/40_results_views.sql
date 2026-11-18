-- Reporting views on the model results, read by Power BI as credit_ro.

CREATE INDEX fact_ecl_rating_now_idx ON dw.fact_ecl (rating_now);
CREATE INDEX fact_ecl_stage_idx ON dw.fact_ecl (stage);

-- One row per open loan: ECL with the loan attributes a report slices by.
CREATE VIEW bi.ecl_loan AS
SELECT e.loan_id,
       e.stage,
       CASE e.stage WHEN 1 THEN 'Stage 1' WHEN 2 THEN 'Stage 2' ELSE 'Stage 3' END AS stage_label,
       CASE
           WHEN e.stage = 3 THEN 'Credit-impaired (31+ days past due)'
           WHEN e.sicr_delinquent THEN '1-30 days past due'
           WHEN e.sicr_hardship THEN 'Hardship plan'
           WHEN e.sicr_pd THEN 'PD at least 2x origination'
           ELSE 'Performing'
       END                       AS stage_reason,
       e.rating_orig,
       ro.rating_rank            AS rating_orig_rank,
       e.rating_now,
       rn.rating_rank            AS rating_now_rank,
       ro.rating_rank - rn.rating_rank AS notches_upgraded,  -- negative = downgraded
       e.pd_orig,
       e.pd_now,
       e.fico_change,
       e.gross_carrying_amt,
       e.ecl,
       e.ecl_upside,
       e.ecl_base,
       e.ecl_downside,
       e.ecl_12m_base,
       e.ecl_lifetime_base,
       e.pd_12m_fwd_base,
       e.pd_lifetime_fwd_base,
       f.term_months,
       f.int_rate / 100.0        AS int_rate,
       s.loan_status,
       g.grade                   AS lc_grade,
       p.purpose_label           AS purpose,
       st.state_code,
       st.region,
       m.year                    AS issue_year,
       m.year_quarter            AS issue_quarter
FROM dw.fact_ecl e
JOIN dw.fact_loan f    ON f.loan_id = e.loan_id
JOIN dw.dim_status s   ON s.status_key = f.status_key
JOIN dw.dim_grade g    ON g.grade_key = f.grade_key
JOIN dw.dim_purpose p  ON p.purpose_key = f.purpose_key
JOIN dw.dim_state st   ON st.state_key = f.state_key
JOIN dw.dim_month m    ON m.month_key = f.issue_month_key
JOIN dw.dim_rating ro  ON ro.rating = e.rating_orig
JOIN dw.dim_rating rn  ON rn.rating = e.rating_now;

CREATE VIEW bi.ecl_by_stage AS
SELECT stage,
       count(*)                                  AS loans,
       sum(gross_carrying_amt)                   AS gross_carrying_amt,
       sum(ecl)                                  AS ecl,
       (sum(ecl) / sum(gross_carrying_amt))::numeric(7,5) AS coverage
FROM dw.fact_ecl
GROUP BY stage;

CREATE VIEW bi.rating_scale AS SELECT * FROM dw.dim_rating;

CREATE VIEW bi.pd_term_structure AS SELECT * FROM dw.fact_pd_curve;

CREATE VIEW bi.ecl_scenario AS SELECT * FROM dw.ecl_scenario;

CREATE VIEW bi.monitoring_quarter AS SELECT * FROM dw.monitoring_quarter;

CREATE VIEW bi.monitoring_feature AS SELECT * FROM dw.monitoring_feature;

COMMENT ON VIEW bi.ecl_loan IS 'IFRS 9 ECL per open loan with stage reason, ratings and loan attributes.';
COMMENT ON VIEW bi.ecl_by_stage IS 'ECL, gross carrying amount and coverage by IFRS 9 stage.';
