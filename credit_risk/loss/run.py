"""Phase 5: LGD from recoveries and EAD from amortisation schedules.

Usage (from the repo root, after credit_risk.pd.run):
    uv run python -m credit_risk.loss.run
"""

import json
import logging
import sys

import numpy as np
import pandas as pd

from credit_risk.config import REPO_ROOT, get_settings
from credit_risk.data.build_loans import OUTPUT_NAME
from credit_risk.loss import ead, lgd
from credit_risk.pd.calibration import GRADES

log = logging.getLogger("loss")

OPEN_STATUSES = ("Current", "In Grace Period", "Late (16-30 days)", "Late (31-120 days)", "Default")
PLAIN_INT_COLS = ("Default year", "Term", "Months from 2019-03-31")  # no thousands separator
REPORT_MD = REPO_ROOT / "docs" / "phase5_lgd_ead.md"
REPORT_JSON = REPO_ROOT / "docs" / "phase5_summary.json"


def load() -> pd.DataFrame:
    processed = get_settings().processed_path
    cols = [
        "id",
        "issue_d",
        "loan_status",
        "term",
        "funded_amnt",
        "int_rate",
        "installment",
        "total_rec_prncp",
        "recoveries",
        "collection_recovery_fee",
        "months_to_default",
        "out_prncp",
        "months_observed",
        "last_pymnt_d",
        "debt_settlement_flag",
    ]
    loans = pd.read_parquet(processed / OUTPUT_NAME, columns=cols)
    ratings = pd.read_parquet(processed / "ratings.parquet", columns=["id", "rating"])
    return loans.merge(ratings, on="id", validate="one_to_one")


def ead_validation(df: pd.DataFrame) -> dict:
    """Realised EAD of charged-off loans vs the schedule from origination.

    The default month is the first missed payment, so the scheduled balance is the one after
    months_to_default - 1 payments.
    """
    co = df[df["loan_status"].eq("Charged Off")].copy()
    co["ead_actual"] = co["funded_amnt"] - co["total_rec_prncp"]
    co = co[co["ead_actual"] > 0]
    co["ead_scheduled"] = ead.balance_after(
        co["funded_amnt"], co["int_rate"], co["installment"], co["months_to_default"] - 1
    )
    co = co[co["ead_scheduled"] > 0]
    ratio = co["ead_actual"] / co["ead_scheduled"]
    by_term = {
        int(t): {
            "loans": int(len(g)),
            "weighted_ratio": float(g["ead_actual"].sum() / g["ead_scheduled"].sum()),
            "median_ratio": float((g["ead_actual"] / g["ead_scheduled"]).median()),
        }
        for t, g in co.groupby("term")
    }
    return {
        "loans": int(len(co)),
        "weighted_ratio": float(co["ead_actual"].sum() / co["ead_scheduled"].sum()),
        "median_ratio": float(ratio.median()),
        "share_within_5pct": float((ratio.sub(1).abs() <= 0.05).mean()),
        "share_below_95pct": float((ratio < 0.95).mean()),
        "share_above_105pct": float((ratio > 1.05).mean()),
        "by_term": by_term,
    }


def open_book_runoff(df: pd.DataFrame) -> dict:
    """Contractual run-off of the open book from the reporting date."""
    book = df[df["loan_status"].isin(OPEN_STATUSES) & (df["out_prncp"] > 0)]
    sched = ead.schedule(book["out_prncp"], book["int_rate"], book["installment"], 61)
    total = sched.sum(axis=0)
    # Contractual balance today if every payment since issue had been made exactly on time.
    paid = (book["months_observed"]).clip(lower=0)
    contractual_now = ead.balance_after(
        book["funded_amnt"], book["int_rate"], book["installment"], paid
    )
    current = book["loan_status"].eq("Current").to_numpy()
    return {
        "loans": int(len(book)),
        "outstanding_now": float(total[0]),
        "outstanding_after_months": {str(m): float(total[m]) for m in (12, 24, 36, 48, 60)},
        "current_loans_actual_vs_contractual": float(
            book["out_prncp"].to_numpy()[current].sum() / contractual_now[current].sum()
        ),
    }


def md(df: pd.DataFrame, pct_cols=(), money_cols=()) -> str:
    out = df.copy()
    for c in pct_cols:
        out[c] = (100 * out[c]).map(lambda v: f"{v:.1f}%")
    for c in money_cols:
        out[c] = out[c].map(lambda v: f"{v / 1e6:,.1f}M")
    for c in out.columns:
        if pd.api.types.is_integer_dtype(out[c]) and c not in PLAIN_INT_COLS:
            out[c] = out[c].map(lambda v: f"{v:,}")
    header = "| " + " | ".join(map(str, out.columns)) + " |\n|" + "---|" * len(out.columns)
    return "\n".join(
        [header, *("| " + " | ".join(map(str, r)) + " |" for r in out.itertuples(index=False))]
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = load()

    # LGD
    co = lgd.realised(df)
    timing = lgd.recovery_timing(co)
    co["discount_factor"] = lgd.discount_factor(co["int_rate"], timing)
    co["lgd"] = 1 - co["recovery_rate"] * co["discount_factor"]
    done = co[co["workout_complete"]].copy()
    done["default_year"] = lgd.default_period(done).dt.year
    done["months_to_default_band"] = pd.cut(
        done["months_to_default"],
        [0, 6, 12, 24, 36, 60],
        labels=["1-6", "7-12", "13-24", "25-36", "37-60"],
    )
    by_rating = lgd.segment_lgd(done, "rating")
    by_rating["rating"] = pd.Categorical(by_rating["rating"], GRADES, ordered=True)
    by_rating = by_rating.sort_values("rating")
    pooled = float(np.average(done["lgd"], weights=done["ead_at_default"]))
    lgd_by_rating = {
        str(r.rating): (float(r.lgd) if r.defaults >= lgd.MIN_SEGMENT_DEFAULTS else pooled)
        for r in by_rating.itertuples()
    }
    by_term = lgd.segment_lgd(done, "term")
    by_year = lgd.segment_lgd(done, "default_year")
    by_mtd = lgd.segment_lgd(done, "months_to_default_band")
    by_settle = lgd.segment_lgd(done, "debt_settlement_flag")
    mean_month = float((timing.index.to_numpy() * timing.to_numpy()).sum())

    # EAD
    ead_val = ead_validation(df)
    runoff = open_book_runoff(df)

    spec = {
        "lgd_by_rating": lgd_by_rating,
        "lgd_pooled": pooled,
        "ratings_on_pooled": [
            str(r.rating) for r in by_rating.itertuples() if r.defaults < lgd.MIN_SEGMENT_DEFAULTS
        ],
        "workout_months": lgd.WORKOUT_MONTHS,
        "recovery_timing": {int(k): float(v) for k, v in timing.items()},
    }
    (REPO_ROOT / "models" / "lgd.json").write_text(json.dumps(spec, indent=2))

    summary = {
        "charged_off": int(len(co)),
        "complete_workouts": int(len(done)),
        "lgd_pooled": pooled,
        "ratings_on_pooled": [
            str(r.rating) for r in by_rating.itertuples() if r.defaults < lgd.MIN_SEGMENT_DEFAULTS
        ],
        "lgd_pooled_undiscounted": float(
            np.average(1 - done["recovery_rate"], weights=done["ead_at_default"])
        ),
        "zero_recovery_share": float((done["recovery_rate"] == 0).mean()),
        "recovery_mean_month": mean_month,
        "by_rating": by_rating.assign(rating=by_rating["rating"].astype(str)).to_dict("records"),
        "by_term": by_term.to_dict("records"),
        "by_default_year": by_year.to_dict("records"),
        "by_months_to_default": by_mtd.assign(
            months_to_default_band=by_mtd["months_to_default_band"].astype(str)
        ).to_dict("records"),
        "by_settlement": by_settle.to_dict("records"),
        "recovery_rate_by_months_since_default": {
            int(k): float(v)
            for k, v in co.groupby("months_since_default")["recovery_rate"].mean().loc[:36].items()
        },
        "ead_validation": ead_val,
        "open_book_runoff": runoff,
    }
    REPORT_JSON.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    REPORT_MD.write_text(
        render(summary, by_rating, by_term, by_year, by_mtd, by_settle, co), encoding="utf-8"
    )
    log.info(
        "LGD pooled %.4f (undiscounted %.4f) on %s complete workouts; EAD actual/scheduled %.3f",
        pooled,
        summary["lgd_pooled_undiscounted"],
        len(done),
        ead_val["weighted_ratio"],
    )
    return 0


def render(s, by_rating, by_term, by_year, by_mtd, by_settle, co) -> str:
    cols = {
        "defaults": "Defaults",
        "ead": "EAD at default",
        "recovery_rate": "Net recovery",
        "lgd_undiscounted": "LGD (undiscounted)",
        "lgd": "LGD (discounted)",
    }
    pct = ("Net recovery", "LGD (undiscounted)", "LGD (discounted)")

    def seg(df, key, label):
        return md(
            df.rename(columns={key: label, **cols})[[label, *cols.values()]],
            pct,
            ("EAD at default",),
        )

    curve = pd.Series(s["recovery_rate_by_months_since_default"])
    picks = [m for m in (3, 6, 9, 12, 15, 18, 24, 30, 36) if m in curve.index]
    curve_tbl = md(
        pd.DataFrame(
            {
                "Months since first missed payment": picks,
                "Mean net recovery so far": [curve[m] for m in picks],
            }
        ),
        ("Mean net recovery so far",),
    )
    e, r = s["ead_validation"], s["open_book_runoff"]
    term_rows = pd.DataFrame(
        [
            {
                "Term": t,
                "Loans": v["loans"],
                "Actual / scheduled (EAD-weighted)": f"{v['weighted_ratio']:.3f}",
                "Median ratio": f"{v['median_ratio']:.3f}",
            }
            for t, v in e["by_term"].items()
        ]
    )
    runoff_rows = pd.DataFrame(
        [
            {"Months from 2019-03-31": int(m), "Contractual balance": v}
            for m, v in {"0": r["outstanding_now"], **r["outstanding_after_months"]}.items()
        ]
    )
    return f"""# Phase 5: loss given default (LGD) and exposure at default (EAD)

Generated by `python -m credit_risk.loss.run`; numbers are computed. Detail:
[phase5_summary.json](phase5_summary.json).

## LGD

**Data.** {s["charged_off"]:,} charged-off loans. LGD uses the {s["complete_workouts"]:,} whose first
missed payment is at least {lgd.WORKOUT_MONTHS} months before the snapshot, because recoveries
arrive late (table below): younger defaults would overstate LGD.

**Definition.** LGD = 1 − net recovery rate × discount factor, where net recovery =
recoveries − collection fees, as a share of the principal outstanding at default, and the
discount factor brings recoveries back to the default date at the loan's own interest rate
(IFRS 9 discounts at the effective interest rate). Segment values are EAD-weighted.

**Recovery timing.** The data has no recovery dates (for loans with recoveries, `last_pymnt_d` is
still the month before default, so recovery receipts are not dated). Timing is inferred
from the cross-section, as the mean net recovery of loans observed t months after default:

{curve_tbl}

Recoveries build up from about month 6 (charge-off at about 150 days past due, then
collection or debt sale) and level off by month 18–24. Past month 18 the
curve moves by about a point either way: each month is a different default cohort, and
2016 defaults recovered less than 2015 or 2017 ones (default-year table below). Mean
recovery month used for discounting: {s["recovery_mean_month"]:.1f}.

**Result: EAD-weighted LGD {100 * s["lgd_pooled"]:.1f}%** ({100 * s["lgd_pooled_undiscounted"]:.1f}% before
discounting). {100 * s["zero_recovery_share"]:.1f}% of defaults recover nothing. High LGD is typical of
unsecured consumer loans whose bad debt is sold for a few cents per dollar.

### By rating (used for ECL)

{seg(by_rating, "rating", "Rating")}

LGD barely moves with the rating (the discounted spread is mostly the higher interest rate
of riskier loans): the rating predicts *whether* a borrower defaults, not how much is
recovered afterwards. ECL uses the rating-level LGD where a rating has at least
{lgd.MIN_SEGMENT_DEFAULTS:,} completed workouts and the pooled LGD otherwise ({", ".join(s["ratings_on_pooled"]) or "none"}).

### Other cuts considered

By term:

{seg(by_term, "term", "Term")}

By months from issue to default (net recovery rises from early to late defaults, but the
late groups are small):

{seg(by_mtd, "months_to_default_band", "Months to default")}

By default year (older years are small samples):

{seg(by_year, "default_year", "Default year")}

Debt settlement agreed with the borrower:

{seg(by_settle, "debt_settlement_flag", "Settlement")}

Term, months to default and default year move LGD by a few points at most on the large
segments, so the rating segmentation is kept. Debt settlement does matter (net recovery about
three times higher), but it is agreed *after* default, so it cannot segment the LGD of loans
that have not defaulted yet; the pooled LGD already contains the historical settlement mix.
Stage 3 loans (31+ days late or flagged default) get the same LGD with no cure: the data
cannot identify loans that were late and recovered, so cure rates cannot be estimated, and
assuming none is conservative.

## EAD

**Model.** Lending Club loans are fixed-rate and fully amortising. The EAD of an open loan in
a future month is its contractual balance then, rolled forward from today's outstanding
principal with the original instalment and rate. Prepayment is not applied here: it is a
competing risk inside the lifetime PD curves (Phase 4), and applying it twice would understate
losses. EAD is principal only (accrued interest, at most about one month, is left out).

**Consistency check on {e["loans"]:,} charged-off loans.** Scheduled balance at the first missed
payment (from origination) vs the principal actually outstanding at default. This is not an
independent validation: the default month partly comes from total paid ÷ instalment (Phase 1),
so close agreement is expected. It confirms that the schedule formula, rates and instalments
are consistent with the data:

{md(term_rows)}

Overall: actual / scheduled = {e["weighted_ratio"]:.3f} (EAD-weighted), median {e["median_ratio"]:.3f};
{100 * e["share_within_5pct"]:.1f}% of loans within ±5%, {100 * e["share_below_95pct"]:.1f}% below
(paid ahead of schedule before defaulting), {100 * e["share_above_105pct"]:.1f}% above (paid less than due
before the first fully missed instalment).

**Open book run-off.** {r["loans"]:,} open loans with principal outstanding (Phase 4 counted
every loan with an open status); for loans still current, today's principal is
{100 * r["current_loans_actual_vs_contractual"]:.1f}% of the contractual balance they would have if every
payment had been made exactly on schedule (the gap is partial prepayment). Contractual balance
of the open book from the reporting date:

{md(runoff_rows, (), ("Contractual balance",))}
"""


if __name__ == "__main__":
    sys.exit(main())
