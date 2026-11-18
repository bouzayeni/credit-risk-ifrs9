"""Load the model results (Phases 4-7) into the warehouse for Power BI.

Runs as credit_etl in ONE transaction: recreate the results tables, COPY the rating scale,
loan-level ECL, PD curves, scenarios and monitoring, create the bi views, then reconcile
against the phase summaries (row counts, total ECL, ECL by stage) before committing.

Usage (from the repo root, after credit_risk.warehouse.load and the model phases):
    uv run python -m credit_risk.warehouse.load_results
"""

import json
import logging
import sys

import numpy as np
import pandas as pd
import psycopg

from credit_risk.config import REPO_ROOT, get_db_settings, get_settings
from credit_risk.ecl import scenarios
from credit_risk.pd.calibration import GRADE_BOUNDS, GRADES
from credit_risk.warehouse.load import SQL_DIR, copy_frame

log = logging.getLogger("load_results")

DOCS = REPO_ROOT / "docs"
MODELS = REPO_ROOT / "models"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rating_frame(cal: dict, lgd: dict, ecl_summary: dict) -> pd.DataFrame:
    scen = {r["grade"]: r for r in ecl_summary["scenario_pds"]}
    lower = (0.0, *GRADE_BOUNDS)
    upper = (*GRADE_BOUNDS, None)
    return pd.DataFrame({
        "rating": GRADES, "rating_rank": range(1, len(GRADES) + 1),
        "pd_lower": lower, "pd_upper": upper,
        "pd_12m": [cal["grade_pd"][g] for g in GRADES],
        "pd_12m_upside": [scen[g]["pd_upside"] for g in GRADES],
        "pd_12m_downside": [scen[g]["pd_downside"] for g in GRADES],
        "lgd": [lgd["lgd_by_rating"][g] for g in GRADES],
    })  # fmt: skip


def ecl_frame(processed) -> pd.DataFrame:
    e = pd.read_parquet(processed / "ecl.parquet")
    money = [
        "out_prncp",
        "ecl",
        "ecl_upside",
        "ecl_base",
        "ecl_downside",
        "ecl_12m_base",
        "ecl_lifetime_base",
    ]
    e[money] = e[money].round(2)
    return e.rename(columns={"id": "loan_id", "out_prncp": "gross_carrying_amt"})[[
        "loan_id", "stage", "rating_orig", "rating_now", "sicr_delinquent", "sicr_hardship", "sicr_pd",
        "pd_orig", "pd_now", "fico_change", "gross_carrying_amt", "ecl", "ecl_upside", "ecl_base",
        "ecl_downside", "ecl_12m_base", "ecl_lifetime_base", "pd_12m_fwd_base", "pd_lifetime_fwd_base",
    ]]  # fmt: skip


def curve_frame(processed) -> pd.DataFrame:
    c = pd.read_parquet(processed / "pd_term_structure.parquet")
    return c.rename(columns={"term": "term_months", "hazard_default_scaled": "hazard_default_anchored"})[
        ["rating", "term_months", "mob", "hazard_default_anchored", "hazard_repaid", "marginal_pd",
         "cumulative_pd", "survival", "borrowed"]
    ].rename(columns={"hazard_default_anchored": "hazard_default"})  # fmt: skip


def scenario_frame(ecl_summary: dict) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(ecl_summary["scenarios"]):
        rows.append({"scenario": r["scenario"], "sort_order": i + 1, "factor_shift": r["k"],
                     "weight": r["weight"], "ecl": round(r["ecl"], 2)})  # fmt: skip
    return pd.DataFrame(rows)


def monitoring_frames(mon: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    q = pd.DataFrame(mon["quarters"])
    q["quarter_start"] = pd.PeriodIndex(q["quarter"], freq="Q").start_time.date
    cols = {
        "quarter": "issue_quarter", "quarter_start": "quarter_start", "loans": "loans", "mean_score": "mean_score",
        "psi_score": "psi_score", "psi_status": "psi_status", "max_csi": "max_csi", "max_csi_feature": "max_csi_feature",
        "gini": "gini", "gini_se": "gini_se", "gini_status": "gini_status",
        "cal_dev_observed": "observed_12m", "cal_dev_predicted": "scorecard_pd", "cal_dev_ratio": "scorecard_ratio",
        "cal_dev_status": "scorecard_status", "cal_prod_predicted": "production_pd",
        "cal_prod_ratio": "production_ratio", "cal_prod_status": "production_status", "gini_6m": "gini_6m",
        "early_observed": "early_observed", "early_predicted": "early_expected", "early_ratio": "early_ratio",
        "early_status": "early_status",
    }  # fmt: skip
    quarters = q.reindex(columns=list(cols)).rename(columns=cols)
    csi = pd.DataFrame(mon["csi"]).rename(columns={"quarter": "issue_quarter"})
    return quarters, csi[["issue_quarter", "feature", "csi", "status"]]


def reconcile(cur: psycopg.Cursor, ecl_summary: dict, frames: dict) -> dict:
    cur.execute("SELECT count(*), sum(ecl), sum(gross_carrying_amt) FROM dw.fact_ecl")
    n, ecl, gca = cur.fetchone()
    tol = 0.01 * n  # rounding to cents
    if n != ecl_summary["loans"]:
        raise RuntimeError(f"fact_ecl rows {n} != {ecl_summary['loans']}")
    if abs(float(ecl) - ecl_summary["ecl"]) > tol or abs(float(gca) - ecl_summary["gca"]) > tol:
        raise RuntimeError(f"ECL {ecl} / GCA {gca} do not match the Phase 6 summary")
    cur.execute("SELECT stage, sum(ecl) FROM dw.fact_ecl GROUP BY stage")
    by_stage = {int(s): float(v) for s, v in cur.fetchall()}
    for r in ecl_summary["stages"]:
        if str(r["stage"]) != "Total" and abs(by_stage[int(r["stage"])] - r["ecl"]) > tol:
            raise RuntimeError(f"stage {r['stage']} ECL does not match")
    counts = {}
    for table, frame in frames.items():
        cur.execute(f"SELECT count(*) FROM {table}")  # noqa: S608 - fixed table names
        counts[table] = cur.fetchone()[0]
        if counts[table] != len(frame):
            raise RuntimeError(f"{table}: {counts[table]} rows loaded, {len(frame)} expected")
    cur.execute("SELECT count(*) FROM bi.ecl_loan")
    if cur.fetchone()[0] != n:
        raise RuntimeError("bi.ecl_loan lost rows in its joins")
    return {"ecl": float(ecl), "gca": float(gca), **counts}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    processed = get_settings().processed_path
    ecl_summary = read_json(DOCS / "phase6_summary.json")
    quarters, csi = monitoring_frames(read_json(DOCS / "phase7_summary.json"))
    frames = {
        "dw.dim_rating": rating_frame(read_json(MODELS / "pd_calibration.json"), read_json(MODELS / "lgd.json"), ecl_summary),
        "dw.fact_ecl": ecl_frame(processed),
        "dw.fact_pd_curve": curve_frame(processed),
        "dw.ecl_scenario": scenario_frame(ecl_summary),
        "dw.monitoring_quarter": quarters,
        "dw.monitoring_feature": csi,
    }  # fmt: skip
    assert set(scenarios.SCENARIOS) <= set(frames["dw.ecl_scenario"]["scenario"])
    with psycopg.connect(**get_db_settings().conninfo("etl")) as conn, conn.cursor() as cur:
        cur.execute((SQL_DIR / "30_results.sql").read_text(encoding="utf-8"))
        for table, frame in frames.items():
            copy_frame(cur, table, frame.replace({np.nan: None}))
            log.info("%s: %s rows", table, f"{len(frame):,}")
        cur.execute((SQL_DIR / "40_results_views.sql").read_text(encoding="utf-8"))
        cur.execute("ANALYZE dw.fact_ecl")
        result = reconcile(cur, ecl_summary, frames)
        conn.commit()
    log.info("committed; reconciled %s", result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
