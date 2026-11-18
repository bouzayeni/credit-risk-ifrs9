"""Which raw columns the project keeps, and in which role.

Roles decide where a column may be used:

- APPLICATION: known when the loan is approved. The only inputs allowed in the scorecard.
- PRICING: Lending Club's own risk grade and price. Used as a benchmark and to rebuild the
  repayment schedule (EAD), never as scorecard inputs (that would copy LC's model).
- OUTCOME: known only after origination. Used for the target, LGD, EAD and IFRS 9 staging;
  never as model inputs.

Columns not listed (free text, ids, URLs, investor-side amounts, hardship and settlement
details) are dropped.
"""

KEY = ("id", "issue_d", "loan_status", "term", "funded_amnt")

APPLICATION_NUMERIC = (
    "loan_amnt", "annual_inc", "dti", "delinq_2yrs", "fico_range_low", "fico_range_high",
    "inq_last_6mths", "mths_since_last_delinq", "mths_since_last_record", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "collections_12_mths_ex_med",
    "mths_since_last_major_derog", "annual_inc_joint", "dti_joint", "acc_now_delinq",
    "tot_coll_amt", "tot_cur_bal", "total_rev_hi_lim", "acc_open_past_24mths", "avg_cur_bal",
    "bc_open_to_buy", "bc_util", "chargeoff_within_12_mths", "delinq_amnt", "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mort_acc",
    "mths_since_recent_bc", "mths_since_recent_inq", "num_accts_ever_120_pd", "num_actv_bc_tl",
    "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_il_tl", "num_op_rev_tl", "num_rev_accts",
    "num_rev_tl_bal_gt_0", "num_sats", "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m", "pct_tl_nvr_dlq", "percent_bc_gt_75", "pub_rec_bankruptcies",
    "tax_liens", "tot_hi_cred_lim", "total_bal_ex_mort", "total_bc_limit",
    "total_il_high_credit_limit",
    # Collected from 2016-01 only: missing before that date means "not collected".
    "open_acc_6m", "open_act_il", "open_il_12m", "open_il_24m", "mths_since_rcnt_il",
    "total_bal_il", "il_util", "open_rv_12m", "open_rv_24m", "max_bal_bc", "all_util", "inq_fi",
    "total_cu_tl", "inq_last_12m",
)  # fmt: skip

FIELDS_FROM_2016 = APPLICATION_NUMERIC[APPLICATION_NUMERIC.index("open_acc_6m") :]

APPLICATION_CATEGORICAL = (
    "emp_length", "home_ownership", "verification_status", "purpose", "addr_state",
    "application_type", "earliest_cr_line",
)  # fmt: skip

PRICING = ("grade", "sub_grade", "int_rate", "installment")

OUTCOME = (
    "last_pymnt_d", "next_pymnt_d", "last_pymnt_amnt", "out_prncp", "total_pymnt",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries",
    "collection_recovery_fee", "last_fico_range_low", "last_fico_range_high",
    "last_credit_pull_d", "hardship_flag", "debt_settlement_flag",
)  # fmt: skip

RAW_COLUMNS = KEY + APPLICATION_NUMERIC + APPLICATION_CATEGORICAL + PRICING + OUTCOME
