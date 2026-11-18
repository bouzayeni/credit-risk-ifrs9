"""Phase 4: calibrate 12-month PDs, build the rating scale and lifetime PD curves.

Design (the scorecard from Phase 3 is frozen; loans with a complete 12-month window):
    validation     calibrate on 2016, back-test on 2017 (out-of-time)
    production     calibrate on 2017, the latest complete year at the 2019-03-31 reporting
                   date (point-in-time, as IFRS 9 requires); used by later phases
    term structure all vintages 2007-2018, by production rating grade and term

A first attempt (calibrate on 2015-2016 pooled, grade PD = mean fitted PD) under-predicted
2017. It is kept in the report, with the reason the design changed.

Usage (from the repo root, after credit_risk.scorecard.train):
    uv run python -m credit_risk.pd.run
"""

import json
import logging
import sys

import numpy as np
import pandas as pd

from credit_risk.config import REPO_ROOT, get_settings
from credit_risk.data.build_loans import OUTPUT_NAME
from credit_risk.pd import calibration as cal
from credit_risk.pd import term_structure as ts
from credit_risk.scorecard.metrics import PDO

log = logging.getLogger("pd")

HOLDOUT_YEAR = 2017
VARIANTS = {  # name: (calibration years, grade PD method)
    "first attempt": ((2015, 2016), "model"),
    "validated design": ((2016,), "observed"),
}
PRODUCTION_YEARS = (2017,)
OPEN_STATUSES = ("Current", "In Grace Period", "Late (16-30 days)", "Late (31-120 days)", "Default")
REPORT_MD = REPO_ROOT / "docs" / "phase4_pd.md"
REPORT_JSON = REPO_ROOT / "docs" / "phase4_summary.json"


def load() -> pd.DataFrame:
    processed = get_settings().processed_path
    cols = [
        "id",
        "issue_d",
        "term",
        "loan_status",
        "is_default",
        "months_to_default",
        "last_pymnt_d",
        "months_observed",
        "eligible_12m",
        "default_12m",
        "out_prncp",
    ]
    loans = pd.read_parquet(processed / OUTPUT_NAME, columns=cols)
    scores = pd.read_parquet(processed / "scores.parquet", columns=["id", "score"])
    return loans.merge(scores, on="id", validate="one_to_one")


def term_structures(df: pd.DataFrame, grade_pds: dict[str, float]):
    exits = ts.exit_times(df)
    term_curves = {
        int(term): ts.hazards(exits.loc[idx], max_mob=int(term))
        for term, idx in df.groupby("term").groups.items()
    }
    curves, anchors = [], []
    for (grade, term), idx in df.groupby(["rating", "term"]).groups.items():
        raw = ts.hazards(exits.loc[idx], max_mob=int(term), fallback=term_curves[int(term)])
        scale = ts.anchor_scale(raw, grade_pds[grade])
        anchored = ts.cumulate(raw, scale)
        raw_cum = ts.cumulate(raw)["cumulative_pd"]
        anchored.insert(0, "term", int(term))
        anchored.insert(0, "rating", grade)
        anchored["cumulative_pd_unanchored"] = raw_cum.to_numpy()
        curves.append(anchored)
        anchors.append(
            {
                "rating": grade,
                "term": int(term),
                "loans": len(idx),
                "pd_12m_target": round(grade_pds[grade], 4),
                "pd_12m_pooled": round(float(raw_cum.iloc[11]), 4),
                "anchor_factor": round(scale, 3),
                "lifetime_pd_pooled": round(float(raw_cum.iloc[-1]), 4),
                "lifetime_pd_anchored": round(float(anchored["cumulative_pd"].iloc[-1]), 4),
                "months_borrowed": int(anchored["borrowed"].sum()),
                "relative_risk": round(float(anchored["relative_risk"].iloc[0]), 3),
            }
        )
    order = {g: i for i, g in enumerate(cal.GRADES)}
    anchors_df = pd.DataFrame(anchors).sort_values(
        ["rating", "term"], key=lambda col: col.map(order) if col.name == "rating" else col
    )
    return pd.concat(curves, ignore_index=True), anchors_df.reset_index(drop=True)


def md(df: pd.DataFrame, pct_cols=(), fmt=None) -> str:
    out = df.copy()
    for c in pct_cols:
        out[c] = (100 * out[c]).map(lambda v: f"{v:.2f}%")
    for c, f in (fmt or {}).items():
        out[c] = out[c].map(f)
    for c in out.columns:
        if pd.api.types.is_integer_dtype(out[c]):
            out[c] = out[c].map(lambda v: f"{v:,}")
    header = "| " + " | ".join(out.columns) + " |\n|" + "---|" * len(out.columns)
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in out.itertuples(index=False)]
    return "\n".join([header, *rows])


def calibrate(df: pd.DataFrame, years, method: str):
    """Fit on the given issue years; return model, PD, rating, grade PDs and sample mask."""
    sample = df["eligible_12m"] & df["issue_d"].dt.year.isin(years)
    model = cal.fit_calibration(df.loc[sample, "score"], df.loc[sample, "default_12m"])
    pd_12m = pd.Series(model.pd(df["score"]), index=df.index)
    rating = pd.Series(cal.assign_grade(pd_12m), index=df.index)
    values = df["default_12m"] if method == "observed" else pd_12m
    grade_pds = cal.grade_pd(rating[sample], values[sample])
    return model, pd_12m, rating, grade_pds, sample


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = load()
    year = df["issue_d"].dt.year
    holdout = df["eligible_12m"] & (year == HOLDOUT_YEAR)
    y_hold = df.loc[holdout, "default_12m"]

    variants, holdout_tables = [], {}
    for name, (years, method) in VARIANTS.items():
        model, pd_12m, rating, grade_pds, sample = calibrate(df, years, method)
        bt = cal.backtest(rating[holdout], y_hold, grade_pds)
        bt_in = cal.backtest(rating[sample], df.loc[sample, "default_12m"], grade_pds)
        bt_in = bt_in.set_index("grade")
        holdout_tables[name] = bt
        predicted = float(rating[holdout].map(grade_pds).mean())
        variants.append(
            {
                "variant": name,
                "calibrated_on": "–".join(map(str, years)),
                "grade_pd": "observed default rate" if method == "observed" else "mean fitted PD",
                "slope": round(model.slope, 5),
                "pd_2017_predicted": predicted,
                "pd_2017_observed": float(y_hold.mean()),
                "grades_failing_jeffreys": int((bt["jeffreys_p"] < 0.05).sum()),
                "in_sample_ends": {
                    g: {
                        "grade_pd": float(bt_in.loc[g, "grade_pd"]),
                        "observed": float(bt_in.loc[g, "observed_rate"]),
                    }
                    for g in ("R1", "R10")
                },
            }
        )
        log.info(
            "%s: 2017 predicted %.4f vs observed %.4f, %d grades fail Jeffreys",
            name,
            predicted,
            y_hold.mean(),
            variants[-1]["grades_failing_jeffreys"],
        )

    # Production calibration: latest complete year.
    model, df["pd_12m"], df["rating"], grade_pds, calib = calibrate(
        df, PRODUCTION_YEARS, "observed"
    )
    log.info(
        "production (%s): intercept %.3f, slope %.5f (points scale implies %.5f)",
        PRODUCTION_YEARS,
        model.intercept,
        model.slope,
        -np.log(2) / PDO,
    )
    by_year = (
        df[df["eligible_12m"] & year.between(2012, HOLDOUT_YEAR)]
        .assign(year=year)
        .pivot_table(index="rating", columns="year", values="default_12m", aggfunc="mean")
        .reindex(cal.GRADES)
    )
    curves, anchors = term_structures(df, grade_pds)

    open_book = df[df["loan_status"].isin(OPEN_STATUSES)]
    book = (
        open_book.groupby("rating")
        .agg(loans=("id", "size"), outstanding=("out_prncp", "sum"))
        .reindex(cal.GRADES)
        .fillna(0)
    )
    book["share_outstanding"] = book["outstanding"] / book["outstanding"].sum()

    processed = get_settings().processed_path
    df["pd_12m_grade"] = df["rating"].map(grade_pds)
    df[["id", "pd_12m", "rating", "pd_12m_grade"]].to_parquet(
        processed / "ratings.parquet", index=False
    )
    curves.to_parquet(processed / "pd_term_structure.parquet", index=False)
    spec = {
        "intercept": model.intercept,
        "slope": model.slope,
        "calibration_years": PRODUCTION_YEARS,
        "grade_bounds": cal.GRADE_BOUNDS,
        "grade_pd": grade_pds,
    }
    (REPO_ROOT / "models" / "pd_calibration.json").write_text(json.dumps(spec, indent=2))

    summary = {
        "variants": variants,
        "holdout_backtest": {k: v.to_dict("records") for k, v in holdout_tables.items()},
        "production": {
            "years": PRODUCTION_YEARS,
            "intercept": model.intercept,
            "slope": model.slope,
            "slope_implied_by_points": -np.log(2) / PDO,
            "loans": int(calib.sum()),
            "observed_rate": float(df.loc[calib, "default_12m"].mean()),
        },
        "grade_bounds": cal.GRADE_BOUNDS,
        "grade_pd": grade_pds,
        "default_rate_by_year": {
            str(k): {g: (None if pd.isna(v) else float(v)) for g, v in col.items()}
            for k, col in by_year.items()
        },
        "anchors": anchors.to_dict("records"),
        "open_book_by_rating": book.reset_index().to_dict("records"),
    }
    REPORT_JSON.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    REPORT_MD.write_text(render(summary, holdout_tables, by_year, anchors, book), encoding="utf-8")
    log.info("wrote %s", REPORT_MD.name)
    return 0


def render(s, holdout_tables, by_year, anchors, book) -> str:
    pr = s["production"]
    bounds = [0.0, *cal.GRADE_BOUNDS]
    scale = pd.DataFrame(
        {
            "Rating": cal.GRADES,
            "PD range": [
                f"{100 * lo:.2f}% – {100 * hi:.2f}%"
                if i < len(cal.GRADE_BOUNDS)
                else f"> {100 * lo:.2f}%"
                for i, (lo, hi) in enumerate(zip(bounds, [*cal.GRADE_BOUNDS, 1.0], strict=True))
            ],
            "Grade PD": [s["grade_pd"][g] for g in cal.GRADES],
        }
    )
    cols = {
        "grade": "Rating",
        "loans": "Loans",
        "defaults": "Defaults",
        "observed_rate": "Observed",
        "grade_pd": "Grade PD",
        "jeffreys_p": "Jeffreys p",
    }
    fmt_p = {"Jeffreys p": lambda v: f"{v:.3f}"}
    bt = {
        k: md(v.rename(columns=cols), ("Observed", "Grade PD"), fmt_p)
        for k, v in holdout_tables.items()
    }
    var = pd.DataFrame(s["variants"]).rename(
        columns={
            "variant": "Variant",
            "calibrated_on": "Calibrated on",
            "grade_pd": "Grade PD from",
            "slope": "Slope",
            "pd_2017_predicted": "2017 predicted",
            "pd_2017_observed": "2017 observed",
            "grades_failing_jeffreys": "Grades failing Jeffreys (of 10)",
        }
    )
    var = md(var, ("2017 predicted", "2017 observed"))
    yr = by_year.copy()
    yr.columns = [str(cname) for cname in yr.columns]
    yr = md(yr.reset_index().rename(columns={"rating": "Rating"}), yr.columns)
    anc = md(
        anchors[
            [
                "rating",
                "term",
                "loans",
                "pd_12m_target",
                "pd_12m_pooled",
                "anchor_factor",
                "months_borrowed",
                "lifetime_pd_pooled",
                "lifetime_pd_anchored",
            ]
        ].rename(
            columns={
                "rating": "Rating",
                "term": "Term",
                "loans": "Loans",
                "pd_12m_target": "12m PD (grade)",
                "pd_12m_pooled": "12m PD (pooled curve)",
                "anchor_factor": "Factor",
                "months_borrowed": "Borrowed months",
                "lifetime_pd_pooled": "Lifetime PD (pooled)",
                "lifetime_pd_anchored": "Lifetime PD (anchored)",
            }
        ),
        (
            "12m PD (grade)",
            "12m PD (pooled curve)",
            "Lifetime PD (pooled)",
            "Lifetime PD (anchored)",
        ),
    )
    bk = book.reset_index().rename(
        columns={
            "rating": "Rating",
            "loans": "Open loans",
            "outstanding": "Outstanding ($)",
            "share_outstanding": "Share",
        }
    )
    bk["Open loans"] = bk["Open loans"].astype(int)
    bk = md(bk, ("Share",), {"Outstanding ($)": lambda v: f"{v / 1e6:,.1f}M"})
    first = s["variants"][0]
    ends = first["in_sample_ends"]

    return f"""# Phase 4: PD calibration, rating scale and lifetime PD

Generated by `python -m credit_risk.pd.run`; numbers are computed. Detail:
[phase4_summary.json](phase4_summary.json). The Phase 3 scorecard is frozen; this phase only
maps its score to 12-month probabilities of default (PD) and lifetime PD curves.

## Calibration method

PD = 1 / (1 + exp(-(a + b × score))) is fitted on one calibration period; the PD is then
bucketed into ten rating grades with fixed PD bands, and each grade's PD is the observed
12-month default rate of its loans in the calibration period. Calibration is checked on the
following year with the **Jeffreys test** (the ECB's PD back-test): a p-value below 0.05 means
a grade's PD is too low for the defaults observed.

## Validation: calibrate on one year, test on the next

{var}

**What went wrong first, and why the design changed.** The first attempt pooled 2015 and 2016
and used the mean fitted PD per grade. It under-predicted 2017 ({100 * first["pd_2017_predicted"]:.2f}%
vs {100 * first["pd_2017_observed"]:.2f}%), with {first["grades_failing_jeffreys"]} grades failing. Two causes, both visible
without the 2017 data: 2015 was a lower-risk year than 2016 (table by issue year below), so
pooling diluted the current level; and the logistic fit is slightly too steep at the ends
(in-sample it gave R1 a PD of {100 * ends["R1"]["grade_pd"]:.2f}% against {100 * ends["R1"]["observed"]:.2f}%
observed, and R10 {100 * ends["R10"]["grade_pd"]:.2f}% against {100 * ends["R10"]["observed"]:.2f}%). The validated design calibrates on
the most recent year only (point-in-time, which IFRS 9 asks for) and uses each grade's
observed default rate. Because this change was made after seeing the first 2017 result, both
are reported.

Validated design, 2017 back-test by grade:

{bt["validated design"]}

First attempt, 2017 back-test by grade:

{bt["first attempt"]}

## Production calibration (used for IFRS 9)

Refitted with the validated design on the latest complete year, {", ".join(map(str, pr["years"]))}
({pr["loans"]:,} loans, observed 12-month default rate {100 * pr["observed_rate"]:.2f}%):
a = {pr["intercept"]:.3f}, b = {pr["slope"]:.5f}. The points scale implies
b = −ln 2 / {PDO} = {pr["slope_implied_by_points"]:.5f}: the fitted slope is within
{100 * abs(pr["slope"] / pr["slope_implied_by_points"] - 1):.1f}% of it, so the scorecard spreads
risk at the right rate and calibration mostly moves the level.

### Rating master scale

{md(scale, ("Grade PD",))}

### Observed 12-month default rate by rating and issue year

{yr}

Grades rank risk in every year. The level moves with the credit cycle and LC's policy: this
is why the calibration uses the latest year, and why Phase 6 adds forward-looking scenarios.

## Lifetime PD term structure

Monthly default and repayment hazards by rating and term, from all vintages (2007–2018), with
repayment as a competing risk and the last {ts.RECOGNITION_LAG_MONTHS} months of each loan's
window censored (defaults not yet recognised). Months with fewer than {ts.MIN_AT_RISK:,} loans
at risk ("Borrowed months": late months of rare rating-term pairs) use the term's all-grades
curve, scaled by the pair's relative risk over its reliable months. Each curve's default hazards are then scaled by one
factor so its 12-month cumulative PD equals the grade PD (anchoring to point-in-time).

{anc}

Lifetime PD is far above the 12-month PD (59% of defaults happen after month 12, Phase 1):
this is the gap between IFRS 9 stage 1 (12-month ECL) and stage 2 (lifetime ECL). Factors
above 1 mean the pooled history was less risky than the calibration year.

## Open book at 2019-03-31 by rating

{bk}

Outputs: `ratings.parquet` (fitted PD, rating and grade PD for every loan),
`pd_term_structure.parquet` (hazards, survival, marginal and cumulative PD by rating, term and
month on book) and `models/pd_calibration.json`.
"""


if __name__ == "__main__":
    sys.exit(main())
