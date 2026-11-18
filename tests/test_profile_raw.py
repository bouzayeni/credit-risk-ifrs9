import gzip

import pandas as pd

from credit_risk.data import profile_raw


def test_profile_counts_loans_and_drops_footer(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_raw, "CHUNK_ROWS", 2)  # force several chunks
    rows = [
        {"id": 1, "loan_status": "Fully Paid", "issue_d": "Jan-2015", "term": " 36 months",
         "grade": "A"},
        {"id": 2, "loan_status": "Charged Off", "issue_d": "Mar-2016", "term": " 60 months",
         "grade": "C"},
        {"id": 3, "loan_status": "Current", "issue_d": "Dec-2018", "term": " 36 months",
         "grade": "B"},
        {"id": "Total amount funded in policy code 1: 123"},
    ]  # fmt: skip
    df = pd.DataFrame(rows)
    for col in profile_raw.NUMERIC_SUMMARY:
        df[col] = [1.0, 2.0, 3.0, None]
    path = tmp_path / "loans.csv.gz"
    with gzip.open(path, "wt", newline="") as fh:
        df.to_csv(fh, index=False)

    result = profile_raw.profile(path)

    assert result["rows"] == 3
    assert result["issue_date_range"] == ["2015-01-01", "2018-12-01"]
    assert result["loan_status"] == {"Fully Paid": 1, "Charged Off": 1, "Current": 1}
    assert result["term"] == {"36 months": 2, "60 months": 1}
    assert result["status_by_issue_year"]["2016"] == {"Charged Off": 1}
    assert result["numeric_summary"]["loan_amnt"] == {"min": 1.0, "max": 3.0, "mean": 2.0}
