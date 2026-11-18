"""Load loans.parquet into the PostgreSQL star schema (dw) and create the BI views (bi).

Runs as credit_etl in ONE transaction: drop and recreate the tables, COPY the dimensions and
the fact in batches, create the views, then reconcile row counts and the 12-month default
count against the Parquet file before committing. A failure at any step rolls back, so
readers never see a half-loaded warehouse.

Usage (from the repo root, after credit_risk.warehouse.setup_db):
    uv run python -m credit_risk.warehouse.load
"""

import io
import logging
import sys
import time

import pandas as pd
import psycopg
import pyarrow.parquet as pq
from psycopg import sql

from credit_risk.config import REPO_ROOT, get_db_settings, get_settings
from credit_risk.data.build_loans import OUTPUT_NAME
from credit_risk.warehouse import dimensions as d

log = logging.getLogger("load")

SQL_DIR = REPO_ROOT / "warehouse" / "sql"
BATCH_ROWS = 200_000


def copy_frame(cur: psycopg.Cursor, table: str, df: pd.DataFrame) -> None:
    """COPY a frame as CSV. Empty unquoted fields are NULL; booleans are written true/false."""
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, na_rep="")
    schema, name = table.split(".")
    stmt = sql.SQL("COPY {}.{} ({}) FROM STDIN WITH (FORMAT csv)").format(
        sql.Identifier(schema),
        sql.Identifier(name),
        sql.SQL(", ").join(sql.Identifier(c) for c in df.columns),
    )
    with cur.copy(stmt) as copy:
        copy.write(buf.getvalue())


def build_dimensions(parquet) -> dict[str, pd.DataFrame]:
    cols = ["issue_d", "last_pymnt_d", "months_to_default", "sub_grade", "purpose", "addr_state",
            "loan_status"]  # fmt: skip
    src = pd.read_parquet(parquet, columns=cols)
    last_default = (
        src["issue_d"].dt.to_period("M") + src["months_to_default"].fillna(0).astype(int)
    ).max()
    last = max(src["issue_d"].max(), src["last_pymnt_d"].max(), last_default.to_timestamp())
    return {
        "month": d.dim_month(src["issue_d"].min(), last),
        "grade": d.dim_grade(src["sub_grade"]),
        "purpose": d.dim_purpose(src["purpose"]),
        "state": d.dim_state(src["addr_state"]),
        "status": d.dim_status(src["loan_status"]),
    }


def expected_totals(parquet) -> dict[str, int]:
    df = pd.read_parquet(parquet, columns=["eligible_12m", "default_12m", "is_default"])
    return {
        "loans": len(df),
        "defaults": int(df["is_default"].sum()),
        "defaults_12m_eligible": int((df["eligible_12m"] & df["default_12m"]).sum()),
    }


def reconcile(cur: psycopg.Cursor, expected: dict[str, int]) -> dict[str, int]:
    cur.execute("SELECT count(*), count(*) FILTER (WHERE is_default) FROM dw.fact_loan")
    loans, defaults = cur.fetchone()
    cur.execute("SELECT coalesce(sum(defaults_12m), 0) FROM bi.vintage_default_12m")
    (view_defaults,) = cur.fetchone()
    actual = {"loans": loans, "defaults": defaults, "defaults_12m_eligible": int(view_defaults)}
    if actual != expected:
        raise RuntimeError(f"Reconciliation failed: warehouse {actual} vs parquet {expected}")
    return actual


def run(parquet) -> dict[str, int]:
    s = get_db_settings()
    dims = build_dimensions(parquet)
    expected = expected_totals(parquet)
    started = time.perf_counter()
    with psycopg.connect(**s.conninfo("etl")) as conn, conn.cursor() as cur:
        cur.execute((SQL_DIR / "10_star_schema.sql").read_text(encoding="utf-8"))
        for name, frame in dims.items():
            copy_frame(cur, f"dw.dim_{name}", frame)
            log.info("dim_%s: %s rows", name, len(frame))

        loaded = 0
        for batch in pq.ParquetFile(parquet).iter_batches(
            batch_size=BATCH_ROWS, columns=list(d.SOURCE_COLUMNS)
        ):
            fact = d.fact_rows(batch.to_pandas(), dims)
            copy_frame(cur, "dw.fact_loan", fact)
            loaded += len(fact)
            log.info("fact_loan: %s rows", f"{loaded:,}")

        cur.execute((SQL_DIR / "20_bi_views.sql").read_text(encoding="utf-8"))
        cur.execute("ANALYZE dw.fact_loan")
        result = reconcile(cur, expected)
        conn.commit()
    log.info("committed in %.0fs; reconciled %s", time.perf_counter() - started, result)
    return result


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(get_settings().processed_path / OUTPUT_NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
