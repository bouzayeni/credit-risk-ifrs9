"""Profile the raw accepted-loans file in chunks and write docs/phase0_profile.json.

Reads the gzip CSV in chunks so it runs on a machine with little free memory. Records, for
every column, its non-null count and a few examples, plus the distributions that decide the
modelling design: loan status, term, grade, issue year, and status by issue year.

Usage (from the repo root):
    uv run python -m credit_risk.data.profile_raw
"""

import json
import logging
import sys
from collections import Counter, defaultdict

import pandas as pd

from credit_risk.config import REPO_ROOT, get_settings
from credit_risk.data.sources import ACCEPTED_FILE

log = logging.getLogger("profile_raw")

CHUNK_ROWS = 100_000
OUTPUT = REPO_ROOT / "docs" / "phase0_profile.json"
NUMERIC_SUMMARY = (
    "loan_amnt", "funded_amnt", "int_rate", "installment", "annual_inc", "dti",
    "total_pymnt", "total_rec_prncp", "recoveries", "collection_recovery_fee", "out_prncp",
)  # fmt: skip


def profile(path) -> dict:
    rows = 0
    non_null: Counter = Counter()
    examples: dict[str, list] = defaultdict(list)
    dtypes: dict[str, str] = {}
    status, term, grade, year = Counter(), Counter(), Counter(), Counter()
    status_by_year: dict[str, Counter] = defaultdict(Counter)
    sums, mins, maxs = Counter(), {}, {}
    issue_min = issue_max = None

    reader = pd.read_csv(path, compression="gzip", chunksize=CHUNK_ROWS, low_memory=False)
    for chunk in reader:
        # The file ends with summary lines ("Total amount funded in policy code ...") that
        # have no loan id or status; count and drop them rather than treat them as loans.
        chunk = chunk[chunk["loan_status"].notna()]
        rows += len(chunk)
        non_null.update(chunk.notna().sum().to_dict())
        for col in chunk.columns:
            dtypes.setdefault(col, str(chunk[col].dtype))
            if len(examples[col]) < 3:
                vals = chunk[col].dropna().astype(str).unique()[: 3 - len(examples[col])]
                examples[col].extend(vals.tolist())

        issue = pd.to_datetime(chunk["issue_d"], format="%b-%Y", errors="coerce")
        lo, hi = issue.min(), issue.max()
        issue_min = lo if issue_min is None or lo < issue_min else issue_min
        issue_max = hi if issue_max is None or hi > issue_max else issue_max

        status.update(chunk["loan_status"].value_counts().to_dict())
        term.update(chunk["term"].str.strip().value_counts().to_dict())
        grade.update(chunk["grade"].value_counts().to_dict())
        years = issue.dt.year.astype("Int64").astype(str)
        year.update(years.value_counts().to_dict())
        for (y, s), n in chunk.groupby([years, chunk["loan_status"]]).size().items():
            status_by_year[y][s] += int(n)

        for col in NUMERIC_SUMMARY:
            s = pd.to_numeric(chunk[col], errors="coerce")
            sums[col] += float(s.sum())
            mins[col] = min(mins.get(col, float("inf")), float(s.min()))
            maxs[col] = max(maxs.get(col, float("-inf")), float(s.max()))
        log.info("profiled %s rows", f"{rows:,}")

    columns = {
        col: {
            "dtype": dtypes[col],
            "non_null": int(non_null[col]),
            "null_pct": round(100 * (1 - non_null[col] / rows), 2),
            "examples": examples[col],
        }
        for col in dtypes
    }
    return {
        "file": ACCEPTED_FILE,
        "rows": rows,
        "n_columns": len(columns),
        "issue_date_range": [str(issue_min.date()), str(issue_max.date())],
        "loan_status": dict(status.most_common()),
        "term": dict(term.most_common()),
        "grade": dict(sorted(grade.items())),
        "issue_year": dict(sorted(year.items())),
        "status_by_issue_year": {
            y: dict(c.most_common()) for y, c in sorted(status_by_year.items())
        },
        "numeric_summary": {
            col: {"min": mins[col], "max": maxs[col], "mean": sums[col] / rows}
            for col in NUMERIC_SUMMARY
        },
        "columns": columns,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    path = get_settings().raw_path / ACCEPTED_FILE
    result = profile(path)
    OUTPUT.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    log.info("wrote %s (%s rows, %s columns)", OUTPUT, result["rows"], result["n_columns"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
