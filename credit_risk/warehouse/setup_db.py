"""Create the credit_dw database, its tablespace, roles, schemas and grants.

Runs as the PostgreSQL superuser (from .env). Idempotent: existing objects are kept and role
passwords are re-synced with .env, so it is safe to re-run.

- credit_etl owns the dw (star schema) and bi (reporting views) schemas and loads data.
- credit_ro can only read dw and bi, every transaction is read-only, and queries time out.

Usage (from the repo root):
    uv run python -m credit_risk.warehouse.setup_db
"""

import logging
import sys

import psycopg
from psycopg import sql

from credit_risk.config import get_db_settings

log = logging.getLogger("setup_db")

TABLESPACE = "credit_k"
SCHEMAS = ("dw", "bi")
RO_STATEMENT_TIMEOUT = "120s"


def ensure_role(cur: psycopg.Cursor, name: str, password: str) -> None:
    cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (name,))
    verb = "ALTER" if cur.fetchone() else "CREATE"
    cur.execute(
        sql.SQL("{} ROLE {} LOGIN PASSWORD {}").format(
            sql.SQL(verb), sql.Identifier(name), sql.Literal(password)
        )
    )
    log.info("%s role %s", verb.lower() + "d" if verb == "CREATE" else "updated", name)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    s = get_db_settings()
    etl, ro, db = s.credit_etl_user, s.credit_ro_user, s.credit_db_name
    if not s.credit_etl_password.get_secret_value() or not s.credit_ro_password.get_secret_value():
        raise SystemExit("Set CREDIT_ETL_PASSWORD and CREDIT_RO_PASSWORD in .env")

    with psycopg.connect(**s.conninfo("super", "postgres"), autocommit=True) as conn:
        cur = conn.cursor()
        ensure_role(cur, etl, s.credit_etl_password.get_secret_value())
        ensure_role(cur, ro, s.credit_ro_password.get_secret_value())

        tablespace = sql.SQL("")
        if s.credit_db_tablespace_dir:
            cur.execute("SELECT 1 FROM pg_tablespace WHERE spcname = %s", (TABLESPACE,))
            if not cur.fetchone():
                cur.execute(
                    sql.SQL("CREATE TABLESPACE {} LOCATION {}").format(
                        sql.Identifier(TABLESPACE), sql.Literal(s.credit_db_tablespace_dir)
                    )
                )
                log.info("created tablespace %s at %s", TABLESPACE, s.credit_db_tablespace_dir)
            tablespace = sql.SQL(" TABLESPACE {}").format(sql.Identifier(TABLESPACE))

        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db,))
        if not cur.fetchone():
            cur.execute(
                sql.SQL("CREATE DATABASE {} OWNER {}{}").format(
                    sql.Identifier(db), sql.Identifier(etl), tablespace
                )
            )
            log.info("created database %s", db)

    with psycopg.connect(**s.conninfo("super", db), autocommit=True) as conn:
        cur = conn.cursor()
        ident = sql.Identifier
        statements = [
            sql.SQL("REVOKE ALL ON DATABASE {} FROM PUBLIC").format(ident(db)),
            sql.SQL("GRANT CONNECT ON DATABASE {} TO {}, {}").format(
                ident(db), ident(etl), ident(ro)
            ),
            sql.SQL("REVOKE CREATE ON SCHEMA public FROM PUBLIC"),
        ]
        for schema in SCHEMAS:
            statements += [
                sql.SQL("CREATE SCHEMA IF NOT EXISTS {} AUTHORIZATION {}").format(
                    ident(schema), ident(etl)
                ),
                sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(ident(schema), ident(ro)),
                sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA {} TO {}").format(
                    ident(schema), ident(ro)
                ),
                sql.SQL(
                    "ALTER DEFAULT PRIVILEGES FOR ROLE {} IN SCHEMA {} GRANT SELECT ON TABLES TO {}"
                ).format(ident(etl), ident(schema), ident(ro)),
            ]
        statements += [
            sql.SQL("ALTER ROLE {} IN DATABASE {} SET statement_timeout = {}").format(
                ident(ro), ident(db), sql.Literal(RO_STATEMENT_TIMEOUT)
            ),
            sql.SQL("ALTER ROLE {} IN DATABASE {} SET default_transaction_read_only = on").format(
                ident(ro), ident(db)
            ),
        ]
        for stmt in statements:
            cur.execute(stmt)
    log.info("database %s ready: schemas %s, owner %s, read-only %s", db, SCHEMAS, etl, ro)
    return 0


if __name__ == "__main__":
    sys.exit(main())
