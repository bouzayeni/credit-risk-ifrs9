import gzip

import pandas as pd

from credit_risk.data import build_loans
from tests.test_clean import raw_loans


def test_build_writes_typed_parquet_across_chunks(tmp_path):
    raw = raw_loans([
        {"id": 1, "loan_status": "Fully Paid"},
        {"id": 2, "loan_status": "Charged Off", "total_rec_prncp": 1000.0,
         "total_rec_int": 300.0, "last_pymnt_d": "May-2016"},
        {"id": 3, "loan_status": "Does not meet the credit policy. Status:Fully Paid"},
        {"id": 4, "loan_status": "Current", "issue_d": "Jun-2018", "emp_length": None},
    ])  # fmt: skip
    raw.loc[len(raw)] = {"id": "Total amount funded in policy code 1: 99"}
    src = tmp_path / "raw.csv.gz"
    with gzip.open(src, "wt", newline="") as fh:
        raw.to_csv(fh, index=False)

    out = tmp_path / "loans.parquet"
    counts = build_loans.build(src, out, chunk_rows=2)  # three chunks, schema must hold

    df = pd.read_parquet(out)
    assert counts["raw_lines"] == 5 and counts["loans"] == 3
    assert counts["footer_lines"] == 1 and counts["old_policy"] == 1
    assert df["id"].tolist() == [1, 2, 4]
    assert df["default_12m"].tolist() == [False, True, False]
    assert df["eligible_12m"].tolist() == [True, True, False]
    assert not out.with_suffix(".parquet.tmp").exists()

    summary = build_loans.summarise(out, counts)
    assert summary["eligible_12m"] == 2 and summary["default_12m_rate_pct"] == 50.0
    assert "| 2016 | 2 |" in build_loans.render_report(summary)
