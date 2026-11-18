"""Cleaning rules and the default definition, as pure functions on a DataFrame.

Every rule here is applied the same way to the full file and to the test fixtures, so the
tests in tests/test_clean.py document the behaviour.
"""

import numpy as np
import pandas as pd

from credit_risk.data.columns import (
    APPLICATION_NUMERIC,
    OUTCOME,
    PRICING,
)

# Status snapshot: "Current" loans last paid in 2019-03, so the data is as of the end of
# March 2019 (checked in Phase 1; see docs/phase1_report.md).
SNAPSHOT_DATE = pd.Timestamp("2019-03-31")

# Main default definition: written off, flagged default, or 31-120 days past due. The
# regulatory definition is 90+ days past due; the 31-120 bucket cannot be split, so a
# strict variant (written off or flagged default only) is kept for sensitivity analysis.
DEFAULT_STATUSES = frozenset({"Charged Off", "Default", "Late (31-120 days)"})
STRICT_DEFAULT_STATUSES = frozenset({"Charged Off", "Default"})
CLOSED_STATUSES = frozenset({"Fully Paid", "Charged Off"})
OLD_POLICY_PREFIX = "Does not meet the credit policy."

# 12-month outcome: a missed payment due in month 12 is visible as "Late (31-120 days)" about
# two months later, so a vintage needs 12 + 3 months before the snapshot to be complete.
OUTCOME_MONTHS = 12
RECOGNITION_LAG_MONTHS = 3
LAST_COMPLETE_VINTAGE_12M = (
    SNAPSHOT_DATE.to_period("M") - OUTCOME_MONTHS - RECOGNITION_LAG_MONTHS
).to_timestamp()

# Fraction of an instalment treated as rounding when counting instalments paid.
INSTALMENT_TOLERANCE = 0.1

DATE_FORMAT = "%b-%Y"
DATE_COLUMNS = ("issue_d", "earliest_cr_line", "last_pymnt_d", "next_pymnt_d", "last_credit_pull_d")
EMP_LENGTH = {
    "< 1 year": 0,
    "1 year": 1,
    **{f"{n} years": n for n in range(2, 10)},
    "10+ years": 10,
}
HOME_OWNERSHIP = {"MORTGAGE": "MORTGAGE", "RENT": "RENT", "OWN": "OWN"}  # ANY/NONE/OTHER -> OTHER

CATEGORICAL_OUT = (
    "loan_status", "home_ownership", "verification_status", "purpose", "addr_state",
    "application_type", "grade", "sub_grade",
)  # fmt: skip


def months_between(start: pd.Series, end: pd.Series) -> pd.Series:
    """Whole calendar months from start to end (both month-start dates); NaN if either is."""
    out = (end.dt.year - start.dt.year) * 12 + (end.dt.month - start.dt.month)
    return out.astype("Float64")


def drop_footer_and_old_policy(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove the file's footer lines and loans approved under the pre-2010 policy."""
    footer = raw["loan_status"].isna()
    old_policy = raw["loan_status"].fillna("").str.startswith(OLD_POLICY_PREFIX)
    counts = {"footer_lines": int(footer.sum()), "old_policy": int(old_policy.sum())}
    return raw[~footer & ~old_policy], counts


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Typed, cleaned loan-level frame. Returns the frame and counts of values changed."""
    df = raw.copy()
    counts: dict[str, int] = {}

    df["id"] = df["id"].astype("int64")
    for col in DATE_COLUMNS:
        df[col] = pd.to_datetime(df[col], format=DATE_FORMAT, errors="coerce")
    df["term"] = df["term"].str.extract(r"(\d+)", expand=False).astype("int16")

    numeric = (
        ("funded_amnt",)
        + APPLICATION_NUMERIC
        + PRICING[2:]
        + tuple(c for c in OUTCOME if c not in DATE_COLUMNS and not c.endswith("_flag"))
    )
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    # Placeholders and impossible values become missing; extreme but possible values are kept
    # (binning in Phase 3 caps their influence) and counted.
    bad_dti = (df["dti"] < 0) | (df["dti"] >= 999)
    counts["dti_set_missing"] = int(bad_dti.sum())
    df.loc[bad_dti, "dti"] = np.nan
    zero_income = df["annual_inc"] <= 0
    counts["annual_inc_zero_set_missing"] = int(zero_income.sum())
    df.loc[zero_income, "annual_inc"] = np.nan
    counts["annual_inc_above_1m_kept"] = int((df["annual_inc"] > 1_000_000).sum())
    for col in ("last_fico_range_low", "last_fico_range_high"):
        zero = df[col] <= 0  # LC reports 0 when no recent pull exists
        counts[f"{col}_zero_set_missing"] = int(zero.sum())
        df.loc[zero, col] = np.nan

    df["fico"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    df["last_fico"] = (df["last_fico_range_low"] + df["last_fico_range_high"]) / 2
    df["emp_length"] = df["emp_length"].map(EMP_LENGTH).astype("Float64")
    df["home_ownership"] = df["home_ownership"].map(HOME_OWNERSHIP).fillna("OTHER")
    df["credit_history_months"] = months_between(df["earliest_cr_line"], df["issue_d"])
    df["is_joint"] = df["application_type"].ne("Individual")
    for col in ("hardship_flag", "debt_settlement_flag"):
        df[col] = df[col].eq("Y")

    df = df.drop(columns=["fico_range_low", "fico_range_high", "last_fico_range_low",
                          "last_fico_range_high", "earliest_cr_line"])  # fmt: skip
    for col in CATEGORICAL_OUT:
        df[col] = df[col].astype("string")
    return df, counts


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Default flags and the estimated month of default.

    The default month is the due date of the first missed payment, estimated two ways because
    neither is reliable alone:

    - from money repaid: (principal + interest received) / installment gives the number of
      instalments paid. The ratio usually lands just below a whole number (e.g. 4.95, from
      rounding of the final amounts), so INSTALMENT_TOLERANCE is added before rounding down.
      Extra repayments make this estimate too late.
    - from last_pymnt_d (the k-th instalment is paid k months after issue, as fully paid
      loans confirm). Later partial or post-charge-off payments make this too late.

    Both errors push the estimate later, so the earlier of the two is used.
    """
    out = df.copy()
    status = out["loan_status"]
    out["is_default"] = status.isin(DEFAULT_STATUSES)
    out["is_default_strict"] = status.isin(STRICT_DEFAULT_STATUSES)
    out["is_closed"] = status.isin(CLOSED_STATUSES)

    paid = out["total_rec_prncp"] + out["total_rec_int"]
    by_amount = np.floor(paid / out["installment"] + INSTALMENT_TOLERANCE).astype("Float64")
    by_date = months_between(out["issue_d"], out["last_pymnt_d"])
    instalments_paid = pd.concat([by_amount, by_date], axis=1).min(axis=1, skipna=True)
    months_to_default = (instalments_paid.clip(lower=0) + 1).astype("Float64")
    out["months_to_default"] = months_to_default.where(out["is_default"])
    out["default_month_by_amount"] = (by_amount + 1).where(out["is_default"])
    out["default_month_by_date"] = (by_date + 1).where(out["is_default"])

    in_window = out["months_to_default"] <= OUTCOME_MONTHS
    out["eligible_12m"] = out["issue_d"] <= LAST_COMPLETE_VINTAGE_12M
    out["default_12m"] = (out["is_default"] & in_window).fillna(False).astype(bool)
    out["default_12m_strict"] = (out["is_default_strict"] & in_window).fillna(False).astype(bool)
    out["months_observed"] = months_between(
        out["issue_d"], pd.Series(SNAPSHOT_DATE.to_period("M").to_timestamp(), index=out.index)
    )
    for col in ("is_default", "is_default_strict", "is_closed", "eligible_12m"):
        out[col] = out[col].astype(bool)
    return out
