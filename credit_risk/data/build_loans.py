"""Build the clean loan-level dataset (Parquet) and the Phase 1 report.

Reads the raw gzip CSV in chunks (low memory), keeps only the columns listed in
credit_risk.data.columns, applies the cleaning rules and the default definition from
credit_risk.data.clean, and writes data/processed/loans.parquet. Then writes
docs/phase1_report.md and docs/phase1_summary.json from that file.

Usage (from the repo root):
    uv run python -m credit_risk.data.build_loans
    uv run python -m credit_risk.data.build_loans --report-only
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from credit_risk.config import REPO_ROOT, get_settings
from credit_risk.data.clean import (
    LAST_COMPLETE_VINTAGE_12M,
    SNAPSHOT_DATE,
    add_target,
    clean,
    drop_footer_and_old_policy,
)
from credit_risk.data.columns import RAW_COLUMNS
from credit_risk.data.sources import ACCEPTED_FILE

log = logging.getLogger("build_loans")

CHUNK_ROWS = 200_000
OUTPUT_NAME = "loans.parquet"
REPORT_MD = REPO_ROOT / "docs" / "phase1_report.md"
REPORT_JSON = REPO_ROOT / "docs" / "phase1_summary.json"


def build(raw_file: Path, output: Path, chunk_rows: int = CHUNK_ROWS) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".parquet.tmp")
    counts: Counter = Counter()
    writer = None
    schema = None
    try:
        reader = pd.read_csv(raw_file, usecols=list(RAW_COLUMNS), chunksize=chunk_rows,
                             low_memory=False)  # fmt: skip
        for raw in reader:
            counts["raw_lines"] += len(raw)
            kept, dropped = drop_footer_and_old_policy(raw)
            counts.update(dropped)
            if kept.empty:  # e.g. a final chunk holding only footer lines
                continue
            df, cleaned = clean(kept)
            counts.update(cleaned)
            df = add_target(df)
            table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(tmp, schema, compression="zstd")
            writer.write_table(table)
            counts["loans"] += len(df)
            log.info("written %s loans", f"{counts['loans']:,}")
    finally:
        if writer is not None:
            writer.close()
    tmp.replace(output)  # only a complete file ever has the final name
    return dict(counts)


def rate_table(df: pd.DataFrame, by: str, flags: list[str]) -> pd.DataFrame:
    g = df.groupby(by, observed=True)
    out = g.size().rename("loans").to_frame()
    for flag in flags:
        out[flag] = (100 * g[flag].mean()).round(2)
    return out


def summarise(parquet: Path, build_counts: dict) -> dict:
    cols = ["issue_d", "loan_status", "grade", "term", "last_pymnt_d", "is_default",
            "is_default_strict", "is_closed", "eligible_12m", "default_12m",
            "default_12m_strict", "months_to_default", "default_month_by_amount",
            "default_month_by_date"]  # fmt: skip
    df = pd.read_parquet(parquet, columns=cols)
    df["vintage"] = df["issue_d"].dt.year

    current_last_pay = df.loc[df["loan_status"] == "Current", "last_pymnt_d"].value_counts()
    elig = df[df["eligible_12m"]]
    defaults = df[df["is_default"]]
    by_amount, by_date = defaults["default_month_by_amount"], defaults["default_month_by_date"]
    both = by_amount.notna() & by_date.notna()

    by_vintage = rate_table(elig, "vintage", ["default_12m", "default_12m_strict"])
    lifetime = rate_table(df[df["vintage"] <= 2015], "vintage", ["is_default", "is_closed"])
    by_grade = rate_table(elig, "grade", ["default_12m", "default_12m_strict"])
    by_term = rate_table(elig, "term", ["default_12m"])
    mtd = defaults["months_to_default"]

    return {
        "build_counts": build_counts,
        "snapshot_date": str(SNAPSHOT_DATE.date()),
        "current_loans_last_payment_month": {
            str(k.date()): int(v) for k, v in current_last_pay.head(3).items()
        },
        "last_complete_vintage_12m": str(LAST_COMPLETE_VINTAGE_12M.date()),
        "loans": len(df),
        "eligible_12m": int(df["eligible_12m"].sum()),
        "default_12m_rate_pct": round(100 * elig["default_12m"].mean(), 2),
        "default_12m_strict_rate_pct": round(100 * elig["default_12m_strict"].mean(), 2),
        "defaults_total": len(defaults),
        "default_month_estimates": {
            "both_available": int(both.sum()),
            "equal": int((by_amount == by_date)[both].sum()),
            "amount_earlier": int((by_amount < by_date)[both].sum()),
            "date_earlier": int((by_date < by_amount)[both].sum()),
            "median_abs_gap_months": float((by_amount - by_date)[both].abs().median()),
        },
        "months_to_default_quantiles": {
            str(q): float(v) for q, v in mtd.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).items()
        },
        "share_of_defaults_within_12m_pct": round(100 * (mtd <= 12).mean(), 2),
        "by_vintage_12m": by_vintage.reset_index().to_dict(orient="records"),
        "lifetime_by_vintage": lifetime.reset_index().to_dict(orient="records"),
        "by_grade_12m": by_grade.reset_index().to_dict(orient="records"),
        "by_term_12m": by_term.reset_index().to_dict(orient="records"),
    }


def md_table(records: list[dict], headers: dict[str, str]) -> str:
    lines = ["| " + " | ".join(headers.values()) + " |", "|" + "---|" * len(headers)]
    for r in records:
        cells = [f"{r[k]:,}" if k == "loans" else str(r[k]) for k in headers]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_report(s: dict) -> str:
    c = s["build_counts"]
    e = s["default_month_estimates"]
    q = s["months_to_default_quantiles"]
    rates = {"default_12m": "12m default % (main)", "default_12m_strict": "12m default % (strict)"}
    last_pay = ", ".join(f"{k} ({v:,})" for k, v in s["current_loans_last_payment_month"].items())
    term_table = md_table(
        s["by_term_12m"],
        {"term": "Term (months)", "loans": "Loans", "default_12m": "12m default %"},
    )
    lifetime_table = md_table(
        s["lifetime_by_vintage"],
        {
            "vintage": "Vintage",
            "loans": "Loans",
            "is_default": "Defaulted %",
            "is_closed": "Closed %",
        },
    )
    return f"""# Phase 1 report: clean loan dataset and default definition

Generated by `python -m credit_risk.data.build_loans` from `{ACCEPTED_FILE}`; numbers below
are computed, not estimated. Machine-readable copy: [phase1_summary.json](phase1_summary.json).

## Rows kept

| Step | Rows |
|---|---|
| Lines in the raw file | {c["raw_lines"]:,} |
| Footer lines (no loan status) | -{c["footer_lines"]:,} |
| "Does not meet the credit policy" loans (pre-2010 policy) | -{c["old_policy"]:,} |
| **Loans in `loans.parquet`** | **{c["loans"]:,}** |

Cleaning changes (values set to missing, rows kept): `dti` below 0 or 999 placeholder
{c["dti_set_missing"]:,}; `annual_inc` of 0 {c["annual_inc_zero_set_missing"]:,}; latest FICO
reported as 0 {c["last_fico_range_low_zero_set_missing"]:,}. Incomes above 1M are kept
({c["annual_inc_above_1m_kept"]:,} loans); binning will cap their influence.

## Snapshot date

"Current" loans last paid in {last_pay},
so statuses are as of **{s["snapshot_date"]}**. This is the IFRS 9 reporting date.

## Default definition

- **Main:** status Charged Off, Default or Late (31-120 days).
- **Strict (sensitivity):** Charged Off or Default only.
- **Target `default_12m`:** default whose first missed payment falls within 12 months of
  issue. Only loans issued up to **{s["last_complete_vintage_12m"]}** are eligible
  (12 months plus 3 months for a missed payment to show as 31+ days late), which is
  {s["eligible_12m"]:,} loans.

**12-month default rate:** {s["default_12m_rate_pct"]}% (main), {s["default_12m_strict_rate_pct"]}%
(strict).

### Dating the default

The month of default is the first missed payment, estimated from money repaid
(instalments paid) and from `last_pymnt_d`, taking the earlier. Of
{e["both_available"]:,} defaults with both estimates, the two agree for {e["equal"]:,};
the repaid amount is earlier for {e["amount_earlier"]:,} and the last payment date is earlier
for {e["date_earlier"]:,}. Median gap: {e["median_abs_gap_months"]} months. A later last
payment date can come from partial payments after the first missed one or from payments after
charge-off; the data cannot tell these apart, so the earlier estimate is used.

Months from issue to default, all defaults: 10% {q["0.1"]}, 25% {q["0.25"]}, median
{q["0.5"]}, 75% {q["0.75"]}, 90% {q["0.9"]}. Only {s["share_of_defaults_within_12m_pct"]}% of
defaults happen in the first 12 months, which is why IFRS 9 stage 2 needs lifetime PD curves
(Phase 4), not just the 12-month PD.

## 12-month default rate by vintage (eligible loans)

{md_table(s["by_vintage_12m"], {"vintage": "Vintage", "loans": "Loans", **rates})}

## 12-month default rate by Lending Club grade

{md_table(s["by_grade_12m"], {"grade": "Grade", "loans": "Loans", **rates})}

The rate rises steadily from A to G: LC's grade ranks risk well, and is the benchmark the
scorecard (built without it) has to approach.

## 12-month default rate by term

{term_table}

## Lifetime outcome of older vintages (status at snapshot)

{lifetime_table}

Vintages up to 2012 are fully closed. From 2013 on, some loans are still open, so their
lifetime default rate is a lower bound.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report-only", action="store_true", help="Reuse the existing Parquet.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    settings = get_settings()
    output = settings.processed_path / OUTPUT_NAME
    counts_file = output.with_suffix(".counts.json")
    if args.report_only:
        counts = json.loads(counts_file.read_text(encoding="utf-8"))
    else:
        counts = build(settings.raw_path / ACCEPTED_FILE, output)
        counts_file.write_text(json.dumps(counts, indent=2), encoding="utf-8")

    summary = summarise(output, counts)
    REPORT_JSON.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    REPORT_MD.write_text(render_report(summary), encoding="utf-8")
    log.info("wrote %s and %s", REPORT_MD.name, REPORT_JSON.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
