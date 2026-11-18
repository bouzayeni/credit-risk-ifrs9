-- Reporting views read by Power BI (as credit_ro). Created after the load, in the same
-- transaction. Rates are returned as fractions (0.058 = 5.8%) so BI tools format them.

CREATE INDEX fact_loan_issue_month_idx ON dw.fact_loan (issue_month_key);
CREATE INDEX fact_loan_grade_idx ON dw.fact_loan (grade_key);
CREATE INDEX fact_loan_status_idx ON dw.fact_loan (status_key);

-- 12-month default rate by issue month and grade, on loans with a complete 12-month window.
CREATE VIEW bi.vintage_default_12m AS
SELECT m.month_start                         AS vintage_month,
       m.year                                AS vintage_year,
       g.grade,
       count(*)                              AS loans,
       sum(f.funded_amnt)                    AS funded_amnt,
       count(*) FILTER (WHERE f.default_12m) AS defaults_12m,
       avg(f.default_12m::int)::numeric(7,5) AS default_rate_12m
FROM dw.fact_loan f
JOIN dw.dim_month m ON m.month_key = f.issue_month_key
JOIN dw.dim_grade g ON g.grade_key = f.grade_key
WHERE f.eligible_12m
GROUP BY m.month_start, m.year, g.grade;

-- Open loans at the snapshot (the book IFRS 9 provisions are calculated on).
CREATE VIEW bi.open_book AS
SELECT s.loan_status,
       s.status_group,
       s.days_past_due,
       g.grade,
       f.term_months,
       count(*)              AS loans,
       sum(f.out_prncp)      AS outstanding_principal
FROM dw.fact_loan f
JOIN dw.dim_status s ON s.status_key = f.status_key
JOIN dw.dim_grade g ON g.grade_key = f.grade_key
WHERE s.is_open
GROUP BY s.loan_status, s.status_group, s.days_past_due, s.sort_order, g.grade, f.term_months;

COMMENT ON VIEW bi.vintage_default_12m IS '12-month default rate by issue month and grade (eligible loans only).';
COMMENT ON VIEW bi.open_book IS 'Open loans and outstanding principal at 2019-03-31, by status, grade and term.';
