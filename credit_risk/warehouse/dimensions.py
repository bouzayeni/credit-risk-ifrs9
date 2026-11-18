"""Build the star schema's dimensions and fact rows from the clean loan data.

Pure functions on DataFrames, so the mapping rules are unit-tested without a database. Any
value without a dimension row (an unknown state or status) raises instead of loading a
NULL key.
"""

import pandas as pd

# US Census Bureau regions; DC counts as South.
STATES = {
    "AL": ("Alabama", "South"), "AK": ("Alaska", "West"), "AZ": ("Arizona", "West"),
    "AR": ("Arkansas", "South"), "CA": ("California", "West"), "CO": ("Colorado", "West"),
    "CT": ("Connecticut", "Northeast"), "DE": ("Delaware", "South"),
    "DC": ("District of Columbia", "South"), "FL": ("Florida", "South"),
    "GA": ("Georgia", "South"), "HI": ("Hawaii", "West"), "ID": ("Idaho", "West"),
    "IL": ("Illinois", "Midwest"), "IN": ("Indiana", "Midwest"), "IA": ("Iowa", "Midwest"),
    "KS": ("Kansas", "Midwest"), "KY": ("Kentucky", "South"), "LA": ("Louisiana", "South"),
    "ME": ("Maine", "Northeast"), "MD": ("Maryland", "South"),
    "MA": ("Massachusetts", "Northeast"), "MI": ("Michigan", "Midwest"),
    "MN": ("Minnesota", "Midwest"), "MS": ("Mississippi", "South"),
    "MO": ("Missouri", "Midwest"), "MT": ("Montana", "West"), "NE": ("Nebraska", "Midwest"),
    "NV": ("Nevada", "West"), "NH": ("New Hampshire", "Northeast"),
    "NJ": ("New Jersey", "Northeast"), "NM": ("New Mexico", "West"),
    "NY": ("New York", "Northeast"), "NC": ("North Carolina", "South"),
    "ND": ("North Dakota", "Midwest"), "OH": ("Ohio", "Midwest"), "OK": ("Oklahoma", "South"),
    "OR": ("Oregon", "West"), "PA": ("Pennsylvania", "Northeast"),
    "RI": ("Rhode Island", "Northeast"), "SC": ("South Carolina", "South"),
    "SD": ("South Dakota", "Midwest"), "TN": ("Tennessee", "South"), "TX": ("Texas", "South"),
    "UT": ("Utah", "West"), "VT": ("Vermont", "Northeast"), "VA": ("Virginia", "South"),
    "WA": ("Washington", "West"), "WV": ("West Virginia", "South"),
    "WI": ("Wisconsin", "Midwest"), "WY": ("Wyoming", "West"),
}  # fmt: skip

# status: (group, days past due, open, default, strict default, sort order)
STATUSES = {
    "Current": ("Performing", "0", True, False, False, 1),
    "In Grace Period": ("Delinquent", "1-15", True, False, False, 2),
    "Late (16-30 days)": ("Delinquent", "16-30", True, False, False, 3),
    "Late (31-120 days)": ("Defaulted", "31-120", True, True, False, 4),
    "Default": ("Defaulted", "121+", True, True, True, 5),
    "Charged Off": ("Defaulted", "written off", False, True, True, 6),
    "Fully Paid": ("Repaid", "n/a", False, False, False, 7),
}

FACT_COLUMNS = (
    "loan_id", "issue_month_key", "grade_key", "purpose_key", "state_key", "status_key",
    "last_payment_month_key", "default_month_key", "term_months", "funded_amnt", "int_rate",
    "installment", "annual_inc", "dti", "fico", "emp_length_years", "home_ownership",
    "verification_status", "is_joint", "last_fico", "out_prncp", "total_pymnt",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries",
    "collection_recovery_fee", "hardship_flag", "debt_settlement_flag", "is_default",
    "is_default_strict", "is_closed", "months_to_default", "eligible_12m", "default_12m",
    "default_12m_strict", "months_observed",
)  # fmt: skip

# Columns the fact build needs from loans.parquet.
SOURCE_COLUMNS = (
    "id", "issue_d", "sub_grade", "purpose", "addr_state", "loan_status", "last_pymnt_d",
    "term", "funded_amnt", "int_rate", "installment", "annual_inc", "dti", "fico", "emp_length",
    "home_ownership", "verification_status", "is_joint", "last_fico", "out_prncp",
    "total_pymnt", "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries",
    "collection_recovery_fee", "hardship_flag", "debt_settlement_flag", "is_default",
    "is_default_strict", "is_closed", "months_to_default", "eligible_12m", "default_12m",
    "default_12m_strict", "months_observed",
)  # fmt: skip


def month_key(dates: pd.Series) -> pd.Series:
    """yyyymm integer key for month-start dates; missing stays missing."""
    return (dates.dt.year * 100 + dates.dt.month).astype("Int64")


def dim_month(first: pd.Timestamp, last: pd.Timestamp) -> pd.DataFrame:
    months = pd.date_range(first.to_period("M").to_timestamp(), last, freq="MS")
    return pd.DataFrame({
        "month_key": month_key(pd.Series(months)),
        "month_start": months.date,
        "year": months.year,
        "quarter": months.quarter,
        "month": months.month,
        "month_name": months.strftime("%b %Y"),
        "year_quarter": [f"{y} Q{q}" for y, q in zip(months.year, months.quarter, strict=True)],
    })  # fmt: skip


def dim_grade(sub_grades: pd.Series) -> pd.DataFrame:
    values = sorted(sub_grades.dropna().unique())  # "A1" < "A2" < ... < "G5"
    df = pd.DataFrame({"sub_grade": values})
    df["grade"] = df["sub_grade"].str[0]
    df["grade_rank"] = df["grade"].map(lambda g: "ABCDEFG".index(g) + 1)
    df["sub_grade_rank"] = range(1, len(df) + 1)
    df.insert(0, "grade_key", df["sub_grade_rank"])
    return df


def dim_purpose(purposes: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"purpose": sorted(purposes.dropna().unique())})
    df.insert(0, "purpose_key", range(1, len(df) + 1))
    df["purpose_label"] = df["purpose"].str.replace("_", " ").str.capitalize()
    return df


def dim_state(codes: pd.Series) -> pd.DataFrame:
    unknown = sorted(set(codes.dropna()) - STATES.keys())
    if unknown:
        raise ValueError(f"States without a dimension row: {unknown}")
    rows = [(code, *STATES[code]) for code in sorted(STATES)]
    df = pd.DataFrame(rows, columns=["state_code", "state_name", "region"])
    df.insert(0, "state_key", range(1, len(df) + 1))
    return df


def dim_status(statuses: pd.Series) -> pd.DataFrame:
    unknown = sorted(set(statuses.dropna()) - STATUSES.keys())
    if unknown:
        raise ValueError(f"Statuses without a dimension row: {unknown}")
    cols = ["status_group", "days_past_due", "is_open", "is_default", "is_default_strict",
            "sort_order"]  # fmt: skip
    df = pd.DataFrame.from_dict(STATUSES, orient="index", columns=cols)
    df = df.rename_axis("loan_status").reset_index().sort_values("sort_order")
    df.insert(0, "status_key", df["sort_order"])
    return df


def lookup(values: pd.Series, dim: pd.DataFrame, natural: str, key: str) -> pd.Series:
    """Map natural values to surrogate keys; raise if any value has no dimension row."""
    keys = values.map(dict(zip(dim[natural], dim[key], strict=True)))
    missing = values.notna() & keys.isna()
    if missing.any():
        raise ValueError(f"{natural}: no key for {sorted(values[missing].unique())[:5]}")
    return keys.astype("Int64")


def fact_rows(loans: pd.DataFrame, dims: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # Default month = issue month + months_to_default (the first missed due date).
    months = loans["months_to_default"]
    issue_period = loans["issue_d"].dt.to_period("M")
    default_month = pd.Series(pd.NaT, index=loans.index, dtype="datetime64[ns]")
    has = months.notna()
    default_month[has] = (issue_period[has] + months[has].astype(int).to_numpy()).dt.to_timestamp()

    fact = pd.DataFrame({
        "loan_id": loans["id"],
        "issue_month_key": month_key(loans["issue_d"]),
        "grade_key": lookup(loans["sub_grade"], dims["grade"], "sub_grade", "grade_key"),
        "purpose_key": lookup(loans["purpose"], dims["purpose"], "purpose", "purpose_key"),
        "state_key": lookup(loans["addr_state"], dims["state"], "state_code", "state_key"),
        "status_key": lookup(loans["loan_status"], dims["status"], "loan_status", "status_key"),
        "last_payment_month_key": month_key(loans["last_pymnt_d"]),
        "default_month_key": month_key(default_month),
        "term_months": loans["term"],
        "emp_length_years": loans["emp_length"].astype("Int64"),
        "months_to_default": loans["months_to_default"].astype("Int64"),
        "months_observed": loans["months_observed"].astype("Int64"),
    })  # fmt: skip
    same_name = [c for c in FACT_COLUMNS if c not in fact.columns]
    for col in same_name:
        fact[col] = loans[col]
    return fact[list(FACT_COLUMNS)]
