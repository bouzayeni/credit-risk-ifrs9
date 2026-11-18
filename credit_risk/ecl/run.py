"""Phase 6: IFRS 9 expected credit loss of the open book at 2019-03-31.

Usage (from the repo root, after credit_risk.pd.run and credit_risk.loss.run):
    uv run python -m credit_risk.ecl.run
"""

import json
import logging
import sys

import numpy as np
import pandas as pd

from credit_risk.config import REPO_ROOT, get_settings
from credit_risk.data.build_loans import OUTPUT_NAME
from credit_risk.ecl import engine, scenarios, staging
from credit_risk.ecl.report import render
from credit_risk.loss.run import OPEN_STATUSES
from credit_risk.pd.calibration import GRADES, Calibration, assign_grade

log = logging.getLogger("ecl")

MODELS = REPO_ROOT / "models"
REPORT_MD = REPO_ROOT / "docs" / "phase6_ecl.md"
REPORT_JSON = REPO_ROOT / "docs" / "phase6_summary.json"
RHO_YEARS = range(2010, 2018)
SICR_SENSITIVITY = (1.5, 2.0, 3.0)
BACKTEST_VINTAGES = ((2014, 36), (2015, 36), (2013, 60))


def load_specs():
    read = lambda name: json.loads((MODELS / name).read_text())  # noqa: E731
    cal = read("pd_calibration.json")
    return (
        read("scorecard.json"),
        Calibration(cal["intercept"], cal["slope"]),
        cal,
        read("lgd.json"),
    )


def load_book() -> pd.DataFrame:
    processed = get_settings().processed_path
    cols = [
        "id",
        "loan_status",
        "term",
        "int_rate",
        "installment",
        "out_prncp",
        "months_observed",
        "fico",
        "last_fico",
        "hardship_flag",
        "issue_d",
    ]
    book = pd.read_parquet(processed / OUTPUT_NAME, columns=cols, filters=[("out_prncp", ">", 0)])
    book = book[book["loan_status"].isin(OPEN_STATUSES)]
    ratings = pd.read_parquet(processed / "ratings.parquet", columns=["id", "pd_12m", "rating"])
    book = book.merge(ratings, on="id", validate="one_to_one").reset_index(drop=True)
    return book.rename(columns={"pd_12m": "pd_orig", "rating": "rating_orig"})


def systematic_risk() -> dict:
    processed = get_settings().processed_path
    loans = pd.read_parquet(processed / OUTPUT_NAME, columns=["id", "issue_d", "default_12m"])
    loans = loans[loans["issue_d"].dt.year.isin(RHO_YEARS)]
    ratings = pd.read_parquet(processed / "ratings.parquet", columns=["id", "rating"])
    loans = loans.merge(ratings, on="id")
    return scenarios.estimate_rho(loans["rating"], loans["issue_d"].dt.year, loans["default_12m"])


def rerate(book, scorecard, cal, extrapolate: bool) -> tuple[np.ndarray, np.ndarray]:
    pd_now = staging.current_pd(book["pd_orig"], book["fico"], book["last_fico"], scorecard, cal,
                                extrapolate)  # fmt: skip
    return pd_now, assign_grade(pd_now)


def project_all(book, rating_now, grade_pd, rho, lgd_spec, hd, hp, rolls=None) -> dict[str, dict]:
    """ECL of every loan as if performing, per scenario (stage 3 is handled separately).

    Loans now 1-30 days late (`rolls`: probability of reaching 31+ days by bucket) default next
    month with that probability, and otherwise follow their rating's curve.
    """
    g_idx = pd.Series(rating_now).map({g: i for i, g in enumerate(GRADES)}).to_numpy()
    base_pd = pd.Series(rating_now).map(grade_pd).to_numpy()
    lgd = pd.Series(rating_now).map(lgd_spec["lgd_by_rating"]).to_numpy()
    args = dict(
        hd_table=hd, hp_table=hp, grade_idx=g_idx, term=book["term"].to_numpy(dtype=int),
        mob=book["months_observed"].to_numpy(dtype=int),
        balance=book["out_prncp"].to_numpy(dtype=float), rate_pct=book["int_rate"].to_numpy(dtype=float),
        installment=book["installment"].to_numpy(dtype=float), lgd=lgd, shock_months=scenarios.SHOCK_MONTHS,
    )  # fmt: skip
    out = {}
    for name, (k, _w) in scenarios.SCENARIOS.items():
        mult = scenarios.scenario_pd(base_pd, rho, k) / base_pd
        out[name] = engine.project(multiplier=mult, **args)
    if rolls:
        roll = book["loan_status"].map(rolls).fillna(0.0).astype(float).to_numpy()
        rate = book["int_rate"].to_numpy(dtype=float) / 1200
        imminent = roll * lgd * book["out_prncp"].to_numpy(dtype=float) / (1 + rate)
        for p in out.values():
            for c in ("ecl_12m", "ecl_lifetime"):
                p[c] = imminent + (1 - roll) * p[c]
            for c in ("pd_12m_fwd", "pd_lifetime_fwd"):
                p[c] = roll + (1 - roll) * p[c]
    return out


def vintage_backtest(cal_spec, lgd_by_rating_undisc, hd, hp) -> list[dict]:
    """Lifetime loss predicted at origination vs realised, for fully matured vintages.

    Each loan is projected from month 0 with its origination rating and contractual
    schedule, base scenario, undiscounted, with undiscounted LGD (realised losses are not
    discounted either). Realised loss = principal at default - net recoveries.
    """
    processed = get_settings().processed_path
    cols = ["id", "issue_d", "term", "loan_status", "funded_amnt", "int_rate", "installment",
            "total_rec_prncp", "recoveries", "collection_recovery_fee", "default_12m"]  # fmt: skip
    loans = pd.read_parquet(processed / OUTPUT_NAME, columns=cols)
    ratings = pd.read_parquet(processed / "ratings.parquet", columns=["id", "rating"])
    rows = []
    for year, term in BACKTEST_VINTAGES:
        v = loans[(loans["issue_d"].dt.year == year) & (loans["term"] == term)].merge(
            ratings, on="id"
        )
        g_idx = v["rating"].map({g: i for i, g in enumerate(GRADES)}).to_numpy()
        res = engine.project(hd, hp, g_idx, v["term"].to_numpy(dtype=int), np.zeros(len(v), dtype=int),
                             v["funded_amnt"].to_numpy(dtype=float), v["int_rate"].to_numpy(dtype=float),
                             v["installment"].to_numpy(dtype=float), v["rating"].map(lgd_by_rating_undisc).to_numpy(),
                             np.ones(len(v)), 0, discount=False)  # fmt: skip
        bad = v["loan_status"].isin(["Charged Off", "Default", "Late (31-120 days)"])
        net_rec = (v["recoveries"] - v["collection_recovery_fee"]).clip(lower=0)
        loss = ((v["funded_amnt"] - v["total_rec_prncp"]) - net_rec).clip(lower=0).where(bad, 0.0)
        funded = v["funded_amnt"].sum()
        rows.append({"vintage": f"{year}, {term} months", "loans": int(len(v)),
                     "pd_12m_predicted": float(res["pd_12m_fwd"].mean()), "default_12m": float(v["default_12m"].mean()),
                     "pd_predicted": float(res["pd_lifetime_fwd"].mean()), "default_rate": float(bad.mean()),
                     "loss_predicted": float(res["ecl_lifetime"].sum() / funded),
                     "loss_realised": float(loss.sum() / funded)})  # fmt: skip
    return rows


def combine(book, stage, proj, lgd_spec, rating_now) -> pd.DataFrame:
    """Per-loan ECL by scenario and probability-weighted, given the stages."""
    stage3_loss = (
        book["out_prncp"].to_numpy()
        * pd.Series(rating_now).map(lgd_spec["lgd_by_rating"]).to_numpy()
    )
    out = pd.DataFrame(index=book.index)
    weighted = np.zeros(len(book))
    for name, (_k, w) in scenarios.SCENARIOS.items():
        p = proj[name]
        ecl = np.where(
            stage == 3, stage3_loss, np.where(stage == 2, p["ecl_lifetime"], p["ecl_12m"])
        )
        out[f"ecl_{name}"] = ecl
        weighted += w * ecl
    out["ecl"] = weighted
    return out


def stage_table(book, stage, ecl) -> pd.DataFrame:
    t = pd.DataFrame({"stage": stage, "gca": book["out_prncp"].to_numpy(), "ecl": ecl})
    g = t.groupby("stage").agg(loans=("gca", "size"), gca=("gca", "sum"), ecl=("ecl", "sum"))
    g.loc["Total"] = g.sum()
    g["coverage"] = g["ecl"] / g["gca"]
    g["share_gca"] = g["gca"] / g.loc["Total", "gca"]
    return g.reset_index()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    scorecard, cal, cal_spec, lgd_spec = load_specs()
    grade_pd = cal_spec["grade_pd"]
    curves = pd.read_parquet(get_settings().processed_path / "pd_term_structure.parquet")
    hd, hp = engine.hazard_tables(curves)
    book = load_book()
    log.info("open book: %s loans, %.1fM", len(book), book["out_prncp"].sum() / 1e6)

    sys_risk = systematic_risk()
    rho = max(sys_risk["rho"], scenarios.RHO_FLOOR)
    log.info("rho estimated %.4f, used %.4f", sys_risk["rho"], rho)
    rolls = staging.roll_rates(book["loan_status"])
    log.info("roll rates to 31+ days: %s", rolls)

    pd_now, rating_now = rerate(book, scorecard, cal, extrapolate=True)
    stages = staging.assign_stage(
        book["loan_status"], book["hardship_flag"], book["pd_orig"], pd_now
    )
    stage = stages["stage"].to_numpy()
    proj = project_all(book, rating_now, grade_pd, rho, lgd_spec, hd, hp, rolls)
    ecl = combine(book, stage, proj, lgd_spec, rating_now)
    main_tbl = stage_table(book, stage, ecl["ecl"].to_numpy())
    log.info(
        "ECL %.1fM, coverage %.2f%%",
        ecl["ecl"].sum() / 1e6,
        100 * ecl["ecl"].sum() / book["out_prncp"].sum(),
    )

    # Scenarios on their own (same stages)
    scen = []
    for name, (k, w) in scenarios.SCENARIOS.items():
        scen.append({"scenario": name, "k": k, "weight": w, "ecl": float(ecl[f"ecl_{name}"].sum())})
    scen.append(
        {
            "scenario": "probability-weighted",
            "k": None,
            "weight": 1.0,
            "ecl": float(ecl["ecl"].sum()),
        }
    )
    multipliers = pd.DataFrame({
        "grade": GRADES, "pd_base": [grade_pd[g] for g in GRADES],
        **{f"pd_{n}": scenarios.scenario_pd([grade_pd[g] for g in GRADES], rho, k) for n, (k, _) in scenarios.SCENARIOS.items()},
    })  # fmt: skip

    # Sensitivities
    sens = []
    for m in SICR_SENSITIVITY:
        st = staging.assign_stage(
            book["loan_status"], book["hardship_flag"], book["pd_orig"], pd_now, m
        )["stage"].to_numpy()
        e = combine(book, st, proj, lgd_spec, rating_now)["ecl"].sum()
        sens.append(
            {
                "case": f"SICR at {m:g}x origination PD",
                "stage2_loans": int((st == 2).sum()),
                "ecl": float(e),
            }
        )
    all1 = np.where(stage == 3, 3, 1)
    all2 = np.where(stage == 3, 3, 2)
    sens.append(
        {
            "case": "No SICR: every performing loan on 12-month ECL",
            "stage2_loans": 0,
            "ecl": float(combine(book, all1, proj, lgd_spec, rating_now)["ecl"].sum()),
        }
    )
    sens.append(
        {
            "case": "Every performing loan on lifetime ECL",
            "stage2_loans": int((all2 == 2).sum()),
            "ecl": float(combine(book, all2, proj, lgd_spec, rating_now)["ecl"].sum()),
        }
    )
    pd_flat, rating_flat = rerate(book, scorecard, cal, extrapolate=False)
    st_flat = staging.assign_stage(
        book["loan_status"], book["hardship_flag"], book["pd_orig"], pd_flat
    )["stage"].to_numpy()
    proj_flat = project_all(book, rating_flat, grade_pd, rho, lgd_spec, hd, hp, rolls)
    sens.append(
        {
            "case": "No FICO extrapolation below the scorecard's lowest bin",
            "stage2_loans": int((st_flat == 2).sum()),
            "ecl": float(combine(book, st_flat, proj_flat, lgd_spec, rating_flat)["ecl"].sum()),
        }
    )
    del proj_flat
    for label, kwargs in (
        ("Delinquent loans on the rating curve only (no roll rates)", dict(rho=rho, rolls=None)),
        (f"Systematic risk as estimated (rho {100 * sys_risk['rho']:.1f}% instead of {100 * rho:.0f}%)",
         dict(rho=sys_risk["rho"], rolls=rolls)),
    ):  # fmt: skip
        alt = project_all(book, rating_now, grade_pd, lgd_spec=lgd_spec, hd=hd, hp=hp, **kwargs)
        sens.append({"case": label, "stage2_loans": int((stage == 2).sum()),
                     "ecl": float(combine(book, stage, alt, lgd_spec, rating_now)["ecl"].sum())})  # fmt: skip
        del alt

    lgd_undisc = json.loads(
        (REPO_ROOT / "docs" / "phase5_summary.json").read_text(encoding="utf-8")
    )
    lgd_undisc = {r["rating"]: r["lgd_undiscounted"] for r in lgd_undisc["by_rating"]}
    backtest = vintage_backtest(cal_spec, lgd_undisc, hd, hp)
    log.info("backtest: %s", backtest)

    # Breakdowns
    detail = book[["id", "loan_status", "term", "out_prncp", "rating_orig", "pd_orig"]].copy()
    detail["pd_now"], detail["rating_now"], detail["stage"] = pd_now, rating_now, stage
    detail[["sicr_delinquent", "sicr_hardship", "sicr_pd"]] = stages[
        ["sicr_delinquent", "sicr_hardship", "sicr_pd"]
    ]
    detail["fico_change"] = (book["last_fico"] - book["fico"]).where(book["last_fico"] >= 300)
    for c in ("ecl_12m", "ecl_lifetime", "pd_12m_fwd", "pd_lifetime_fwd"):
        detail[f"{c}_base"] = proj["base"][c]
    detail = pd.concat([detail, ecl], axis=1)
    detail.to_parquet(get_settings().processed_path / "ecl.parquet", index=False)

    s2 = detail[detail["stage"] == 2]
    drivers = {
        "stage2_loans": int(len(s2)),
        "delinquent": int(s2["sicr_delinquent"].sum()), "hardship": int(s2["sicr_hardship"].sum()),
        "pd_test": int(s2["sicr_pd"].sum()),
        "pd_test_only": int((s2["sicr_pd"] & ~s2["sicr_delinquent"] & ~s2["sicr_hardship"]).sum()),
        "delinquent_also_pd": int((s2["sicr_delinquent"] & s2["sicr_pd"]).sum()),
    }  # fmt: skip
    by_rating = (
        detail.groupby(["rating_now", "stage"]).agg(loans=("id", "size"), gca=("out_prncp", "sum"), ecl=("ecl", "sum"))
        .reset_index()
    )  # fmt: skip
    mig = pd.crosstab(detail["rating_orig"], detail["rating_now"]).reindex(
        index=GRADES, columns=GRADES, fill_value=0
    )
    gi = np.arange(len(GRADES))
    moved = pd.DataFrame({
        "rating_orig": GRADES, "loans": mig.sum(axis=1).to_numpy(),
        "better": [mig.iloc[i, :i].sum() / max(mig.iloc[i].sum(), 1) for i in gi],
        "same": [mig.iloc[i, i] / max(mig.iloc[i].sum(), 1) for i in gi],
        "worse_1_2": [mig.iloc[i, i + 1 : i + 3].sum() / max(mig.iloc[i].sum(), 1) for i in gi],
        "worse_3plus": [mig.iloc[i, i + 3 :].sum() / max(mig.iloc[i].sum(), 1) for i in gi],
    })  # fmt: skip
    fico_bands = pd.cut(detail["fico_change"], [-1000, -80, -60, -40, -20, 0, 20, 1000],
                        labels=["<= -80", "-79 to -60", "-59 to -40", "-39 to -20", "-19 to 0", "+1 to +20", "> +20"])  # fmt: skip
    by_fico = (
        detail.assign(band=fico_bands, delinquent=detail["loan_status"].ne("Current"),
                      pd_ratio=detail["pd_now"] / detail["pd_orig"])
        .groupby("band", observed=True)
        .agg(loans=("id", "size"), delinquent=("delinquent", "mean"), pd_ratio=("pd_ratio", "median"),
             stage2=("stage", lambda s: (s == 2).mean()))
        .reset_index()
    )  # fmt: skip

    summary = {
        "snapshot": "2019-03-31", "loans": int(len(book)), "gca": float(book["out_prncp"].sum()),
        "ecl": float(ecl["ecl"].sum()), "coverage": float(ecl["ecl"].sum() / book["out_prncp"].sum()),
        "stages": main_tbl.to_dict("records"), "stage2_drivers": drivers,
        "rho": rho, "rho_estimated": sys_risk["rho"], "rho_floor": scenarios.RHO_FLOOR,
        "roll_rates": rolls, "backtest": backtest, "factor_sd": sys_risk["factor_sd"], "rho_by_grade": sys_risk["by_grade"].to_dict("records"),
        "scenarios": scen, "scenario_pds": multipliers.to_dict("records"), "shock_months": scenarios.SHOCK_MONTHS,
        "sensitivities": sens, "fico_slope": staging.FicoWoe(next(f for f in scorecard["features"] if f["name"] == "fico")).slope,
        "by_rating": by_rating.to_dict("records"), "migration": moved.to_dict("records"),
        "by_fico_change": by_fico.assign(band=by_fico["band"].astype(str)).to_dict("records"),
        "pd_ratio_median_stage1": float((detail.loc[detail["stage"] == 1, "pd_now"] / detail.loc[detail["stage"] == 1, "pd_orig"]).median()),
        "mean_lifetime_pd_stage2_base": float(s2["pd_lifetime_fwd_base"].mean()),
        "mean_12m_pd_stage1_base": float(detail.loc[detail["stage"] == 1, "pd_12m_fwd_base"].mean()),
    }  # fmt: skip
    REPORT_JSON.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    REPORT_MD.write_text(render(summary), encoding="utf-8")
    log.info("wrote %s", REPORT_MD.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
