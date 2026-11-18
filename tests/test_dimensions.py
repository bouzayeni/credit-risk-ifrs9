import pandas as pd
import pytest

from credit_risk.data import clean as c
from credit_risk.warehouse import dimensions as d
from tests.test_clean import raw_loans


def loans(rows: list[dict]) -> pd.DataFrame:
    kept, _ = c.drop_footer_and_old_policy(raw_loans(rows))
    df, _ = c.clean(kept)
    return c.add_target(df)


def dims_for(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "grade": d.dim_grade(df["sub_grade"]),
        "purpose": d.dim_purpose(df["purpose"]),
        "state": d.dim_state(df["addr_state"]),
        "status": d.dim_status(df["loan_status"]),
    }


def test_dim_month_keys_and_labels():
    m = d.dim_month(pd.Timestamp("2018-11-15"), pd.Timestamp("2019-02-01"))
    assert m["month_key"].tolist() == [201811, 201812, 201901, 201902]
    assert m["year_quarter"].tolist() == ["2018 Q4", "2018 Q4", "2019 Q1", "2019 Q1"]
    assert m["month_name"].iloc[0] == "Nov 2018"


def test_dim_grade_orders_sub_grades():
    g = d.dim_grade(pd.Series(["B2", "A1", "G5", "A1", None]))
    assert g["sub_grade"].tolist() == ["A1", "B2", "G5"]
    assert g["grade_rank"].tolist() == [1, 2, 7]
    assert g["grade_key"].tolist() == [1, 2, 3]


def test_dim_state_covers_all_states_and_rejects_unknown():
    assert len(d.dim_state(pd.Series(["CA"]))) == 51  # 50 states + DC
    with pytest.raises(ValueError, match="XX"):
        d.dim_state(pd.Series(["CA", "XX"]))


def test_dim_status_matches_default_definition():
    s = d.dim_status(pd.Series(["Current"])).set_index("loan_status")
    assert set(s.index[s["is_default"]]) == c.DEFAULT_STATUSES
    assert set(s.index[s["is_default_strict"]]) == c.STRICT_DEFAULT_STATUSES
    assert not s.loc[["Fully Paid", "Charged Off"], "is_open"].any()
    with pytest.raises(ValueError):
        d.dim_status(pd.Series(["Written off"]))


def test_fact_rows_keys_and_default_month():
    df = loans([
        {"id": 7, "sub_grade": "C3", "purpose": "car", "addr_state": "NY"},
        {"id": 8, "loan_status": "Charged Off", "issue_d": "Nov-2016", "total_rec_prncp": 1000.0,
         "total_rec_int": 300.0, "last_pymnt_d": "Mar-2017"},
    ])  # fmt: skip
    fact = d.fact_rows(df, dims_for(df))

    assert list(fact.columns) == list(d.FACT_COLUMNS)
    assert fact["issue_month_key"].tolist() == [201601, 201611]
    # 4 instalments paid, first missed due in month 5 after Nov 2016 = Apr 2017
    assert fact["months_to_default"].tolist()[1] == 5
    assert fact["default_month_key"].tolist()[1] == 201704
    assert pd.isna(fact["default_month_key"].iloc[0])
    assert fact["last_payment_month_key"].tolist() == [201901, 201703]


def test_fact_rows_rejects_values_missing_from_a_dimension():
    df = loans([{"sub_grade": "B2"}])
    dims = dims_for(df)
    dims["grade"] = d.dim_grade(pd.Series(["A1"]))
    with pytest.raises(ValueError, match="sub_grade"):
        d.fact_rows(df, dims)
