"""Checks against the loaded PostgreSQL warehouse. Skipped when the database is unreachable.

Run after `python -m credit_risk.warehouse.load`:  uv run pytest -m db
"""

import json

import psycopg
import pytest

from credit_risk.config import REPO_ROOT, get_db_settings

pytestmark = pytest.mark.db


@pytest.fixture(scope="module")
def ro():
    try:
        conn = psycopg.connect(**get_db_settings().conninfo("ro"))
    except psycopg.OperationalError as exc:
        pytest.skip(f"warehouse not reachable: {exc}")
    yield conn
    conn.close()


def scalar(conn, query: str):
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchone()[0]


def test_matches_the_phase1_report(ro):
    summary = json.loads((REPO_ROOT / "docs" / "phase1_summary.json").read_text())
    assert scalar(ro, "SELECT count(*) FROM dw.fact_loan") == summary["loans"]
    rate = scalar(
        ro, "SELECT round(100.0 * sum(defaults_12m) / sum(loans), 2) FROM bi.vintage_default_12m"
    )
    assert float(rate) == summary["default_12m_rate_pct"]
    by_grade = {r["grade"]: r["default_12m"] for r in summary["by_grade_12m"]}
    with ro.cursor() as cur:
        cur.execute(
            "SELECT grade, round(100.0 * sum(defaults_12m) / sum(loans), 2) "
            "FROM bi.vintage_default_12m GROUP BY grade"
        )
        assert {g: float(v) for g, v in cur.fetchall()} == by_grade


def test_every_fact_row_has_its_dimensions(ro):
    orphans = scalar(
        ro,
        """SELECT count(*) FROM dw.fact_loan f
           LEFT JOIN dw.dim_month m ON m.month_key = f.issue_month_key
           LEFT JOIN dw.dim_grade g ON g.grade_key = f.grade_key
           LEFT JOIN dw.dim_status s ON s.status_key = f.status_key
           WHERE m.month_key IS NULL OR g.grade_key IS NULL OR s.status_key IS NULL""",
    )
    assert orphans == 0


def test_open_book_is_only_open_statuses(ro):
    with ro.cursor() as cur:
        cur.execute("SELECT DISTINCT loan_status FROM bi.open_book")
        statuses = {r[0] for r in cur.fetchall()}
    assert statuses <= {"Current", "In Grace Period", "Late (16-30 days)",
                        "Late (31-120 days)", "Default"}  # fmt: skip
    assert "Current" in statuses


def test_read_only_role_cannot_write(ro):
    with ro.cursor() as cur, pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
        cur.execute("DELETE FROM dw.dim_status")
    ro.rollback()

    # Read-only mode is only a default the session can switch off; the real guard is that
    # the role has no write privileges. Prove that with read-only mode turned off.
    ro.autocommit = True
    try:
        with ro.cursor() as cur:
            cur.execute("SET default_transaction_read_only = off")
            for stmt in ("DELETE FROM dw.dim_status", "CREATE TABLE public.x (i int)",
                         "CREATE TABLE dw.x (i int)"):  # fmt: skip
                with pytest.raises(psycopg.errors.InsufficientPrivilege):
                    cur.execute(stmt)
            cur.execute("RESET default_transaction_read_only")
    finally:
        ro.autocommit = False
