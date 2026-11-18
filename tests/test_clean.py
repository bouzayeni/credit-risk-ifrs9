import numpy as np
import pandas as pd
import pytest

from credit_risk.data import clean as c
from credit_risk.data.columns import RAW_COLUMNS


def raw_loans(rows: list[dict]) -> pd.DataFrame:
    """Raw-format rows (strings as in the CSV); unspecified columns are missing."""
    base = {
        "id": 1, "issue_d": "Jan-2016", "loan_status": "Fully Paid", "term": " 36 months",
        "funded_amnt": 10000.0, "loan_amnt": 10000.0, "annual_inc": 50000.0, "dti": 15.0,
        "fico_range_low": 700.0, "fico_range_high": 704.0, "emp_length": "3 years",
        "home_ownership": "RENT", "verification_status": "Verified", "purpose": "credit_card",
        "addr_state": "CA", "application_type": "Individual", "earliest_cr_line": "Jan-2006",
        "grade": "B", "sub_grade": "B2", "int_rate": 10.0, "installment": 322.67,
        "last_pymnt_d": "Jan-2019", "total_rec_prncp": 10000.0, "total_rec_int": 1616.0,
        "recoveries": 0.0, "last_fico_range_low": 710.0, "last_fico_range_high": 714.0,
        "hardship_flag": "N", "debt_settlement_flag": "N",
    }  # fmt: skip
    df = pd.DataFrame([{**base, **r} for r in rows])
    for col in RAW_COLUMNS:
        if col not in df:
            df[col] = np.nan
    return df[list(RAW_COLUMNS)]


def prepare(rows: list[dict]) -> pd.DataFrame:
    kept, _ = c.drop_footer_and_old_policy(raw_loans(rows))
    df, _ = c.clean(kept)
    return c.add_target(df)


def test_drops_footer_and_old_policy_loans():
    raw = raw_loans([
        {"id": 1},
        {"id": 2, "loan_status": "Does not meet the credit policy. Status:Charged Off"},
    ])  # fmt: skip
    raw.loc[len(raw)] = {"id": "Total amount funded in policy code 1: 99"}
    kept, counts = c.drop_footer_and_old_policy(raw)
    assert kept["id"].tolist() == [1]
    assert counts == {"footer_lines": 1, "old_policy": 1}


def test_cleaning_rules():
    raw = raw_loans([
        {"id": 1, "dti": -1.0, "annual_inc": 0.0, "emp_length": "10+ years",
         "home_ownership": "ANY", "last_fico_range_low": 0.0, "last_fico_range_high": 0.0},
        {"id": 2, "dti": 999.0, "emp_length": "< 1 year", "application_type": "Joint App",
         "hardship_flag": "Y"},
        {"id": 3, "dti": 45.5, "emp_length": np.nan, "term": " 60 months"},
    ])  # fmt: skip
    df, counts = c.clean(raw)

    assert df["dti"].isna().tolist() == [True, True, False]
    assert np.isnan(df.loc[0, "annual_inc"])
    assert df["emp_length"].tolist()[:2] == [10, 0] and pd.isna(df["emp_length"].iloc[2])
    assert df["home_ownership"].tolist() == ["OTHER", "RENT", "RENT"]
    assert df["fico"].iloc[1] == 702.0
    assert np.isnan(df.loc[0, "last_fico"])
    assert df["term"].tolist() == [36, 36, 60]
    assert df["credit_history_months"].iloc[0] == 120
    assert df["is_joint"].tolist() == [False, True, False]
    assert df["hardship_flag"].tolist() == [False, True, False]
    assert counts["dti_set_missing"] == 2
    assert counts["annual_inc_zero_set_missing"] == 1


def test_default_flags_by_status():
    df = prepare([
        {"id": 1, "loan_status": "Fully Paid"},
        {"id": 2, "loan_status": "Charged Off", "total_rec_prncp": 1000.0, "total_rec_int": 300.0,
         "last_pymnt_d": "May-2016"},
        {"id": 3, "loan_status": "Late (31-120 days)", "total_rec_prncp": 2000.0,
         "total_rec_int": 500.0, "last_pymnt_d": "Aug-2016"},
        {"id": 4, "loan_status": "In Grace Period"},
        {"id": 5, "loan_status": "Current"},
    ])  # fmt: skip
    assert df["is_default"].tolist() == [False, True, True, False, False]
    assert df["is_default_strict"].tolist() == [False, True, False, False, False]
    assert df["is_closed"].tolist() == [True, True, False, False, False]
    assert df["months_to_default"].isna().tolist() == [True, False, False, True, True]


def test_default_month_takes_earlier_estimate():
    # Paid 4 instalments (1300/322.67), but a recovery in 2018 moved last_pymnt_d: use amount.
    recovery = {
        "loan_status": "Charged Off",
        "total_rec_prncp": 1000.0,
        "total_rec_int": 300.0,
        "last_pymnt_d": "Jun-2018",
    }
    # Prepaid a lump sum, so the amount suggests 20 instalments, but payments stopped in month 6.
    prepaid = {
        "loan_status": "Charged Off",
        "total_rec_prncp": 6000.0,
        "total_rec_int": 500.0,
        "last_pymnt_d": "Jul-2016",
    }
    # Never paid anything.
    no_payment = {
        "loan_status": "Charged Off",
        "total_rec_prncp": 0.0,
        "total_rec_int": 0.0,
        "last_pymnt_d": np.nan,
    }
    # Paid 4.95 instalments: rounding in the final amounts, counts as 5 paid.
    rounding = {
        "loan_status": "Charged Off",
        "total_rec_prncp": 1300.0,
        "total_rec_int": 297.22,
        "last_pymnt_d": "Jun-2016",
    }
    df = prepare([recovery, prepaid, no_payment, rounding])
    assert df["months_to_default"].tolist() == [5, 7, 1, 6]
    assert df["default_12m"].tolist() == [True, True, True, True]


def test_late_default_is_outside_the_12m_window():
    df = prepare([{"loan_status": "Charged Off", "total_rec_prncp": 6000.0,
                   "total_rec_int": 1000.0, "last_pymnt_d": "Oct-2017"}])  # fmt: skip
    assert df["months_to_default"].iloc[0] == 22  # 21 instalments paid
    assert df["is_default"].iloc[0] and not df["default_12m"].iloc[0]


@pytest.mark.parametrize(
    ("issue", "eligible"), [("Dec-2017", True), ("Jan-2018", False), ("Jun-2007", True)]
)
def test_12m_eligibility_needs_a_complete_window(issue, eligible):
    df = prepare([{"issue_d": issue, "earliest_cr_line": "Jan-2000"}])
    assert (
        df["eligible_12m"].iloc[0] is np.bool_(eligible) or df["eligible_12m"].iloc[0] == eligible
    )


def test_snapshot_constants():
    assert pd.Timestamp("2017-12-01") == c.LAST_COMPLETE_VINTAGE_12M
    df = prepare([{"issue_d": "Mar-2018"}])
    assert df["months_observed"].iloc[0] == 12
