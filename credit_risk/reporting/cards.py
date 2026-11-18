"""Phase 9: model cards and the model risk register, generated from the phase summaries.

Every number comes from docs/phase*_summary.json, so the cards cannot disagree with the
phase reports they summarise.

Usage (from the repo root, after all phases):
    uv run python -m credit_risk.reporting.cards
"""

import json
import logging
import math
import sys

from credit_risk.config import REPO_ROOT

log = logging.getLogger("cards")

DOCS = REPO_ROOT / "docs"


def load(name: str) -> dict:
    return json.loads((DOCS / f"{name}.json").read_text(encoding="utf-8"))


def ok(v) -> bool:
    """A metric that exists for this quarter (JSON keeps missing ones as NaN)."""
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def pct(v, d=1) -> str:
    return f"{100 * v:.{d}f}%"


def money(v) -> str:
    return f"${v / 1e6:,.1f}M"


def sens(s6: dict, prefix: str) -> dict:
    return next(r for r in s6["sensitivities"] if r["case"].startswith(prefix))


def change(s6: dict, prefix: str) -> str:
    return f"{100 * (sens(s6, prefix)['ecl'] / s6['ecl'] - 1):+.1f}%"


def scorecard_card(s1, s3, s7) -> str:
    sp, bm = s3["splits"], s3["benchmarks_test"]
    feats = "\n".join(
        f"| {f['feature']} | {f['iv']:.3f} | {f['coef']:.3f} |" for f in s3["final_features"]
    )
    q = s7["quarters"]
    gmin = min((r for r in q if ok(r.get("gini"))), key=lambda r: r["gini"])
    return f"""## 1. Application scorecard

| | |
|---|---|
| Purpose | Rank applicants by risk of default within 12 months, at origination |
| Owner / version | Project author; `models/scorecard.json` (Phase 3) |
| Target | `default_12m`: first missed payment within 12 months of issue, of a loan charged off, in default or 31–120 days late at 2019-03-31 ({s1["default_12m_rate_pct"]:.2f}% of {s1["eligible_12m"]:,} loans with a full 12-month window) |
| Population | Booked Lending Club loans; no rejected applications |
| Development | Train {sp["train"]["loans"]:,} loans (2007–2014), validation {sp["valid"]["loans"]:,} (2015), out-of-time test {sp["test"]["loans"]:,} (2016–2017) |
| Method | Monotonic WoE binning (bins at least 5% of loans), IV and correlation filters, forward selection on validation Gini, logistic regression, points scale 600 at odds 50:1, 20 points to double the odds |
| Report | [phase3_scorecard.md](phase3_scorecard.md) |

**Features** ({len(s3["final_features"])}):

| Feature | IV | Coefficient (on WoE) |
|---|---|---|
{feats}

**Performance** (Gini / KS):

| Sample | Gini | KS |
|---|---|---|
| Train | {sp["train"]["gini"]:.3f} | {sp["train"]["ks"]:.3f} |
| Validation (2015) | {sp["valid"]["gini"]:.3f} | {sp["valid"]["ks"]:.3f} |
| Out-of-time test (2016–2017) | {sp["test"]["gini"]:.3f} (95% CI {sp["test"]["gini_ci95"][0]:.3f}–{sp["test"]["gini_ci95"][1]:.3f}) | {sp["test"]["ks"]:.3f} |
| Benchmark: XGBoost on the same candidates | {bm["xgboost"]["gini"]:.3f} | {bm["xgboost"]["ks"]:.3f} |
| Benchmark: Lending Club sub-grade | {bm["lc_sub_grade"]["gini"]:.3f} | {bm["lc_sub_grade"]["ks"]:.3f} |

**Use and limits.** An interpretable scorecard that gives up about
{bm["xgboost"]["gini"] - sp["test"]["gini"]:.2f} Gini against gradient boosting and
{bm["lc_sub_grade"]["gini"] - sp["test"]["gini"]:.2f} against Lending Club's own sub-grade. Its points-scale PD is not used
for provisioning: it under-predicted 2016–2017 defaults by up to {max(r["cal_dev_ratio"] for r in q if ok(r.get("cal_dev_ratio"))):.2f}×
(Phase 7), so PDs come from the Phase 4 calibration. Monitoring: Gini at its lowest
{gmin["gini"]:.3f} ({gmin["quarter"]}), still above the amber line; population shift from 2017Q4.
"""


def pd_card(s4, s6, s7) -> str:
    first, final = s4["variants"]
    prod = s4["production"]
    gp = s4["grade_pd"]
    anchors = s4["anchors"]
    borrowed = [a for a in anchors if a["months_borrowed"] > 0]
    bt = s6["backtest"]
    early = [r for r in s7["quarters"] if ok(r.get("early_ratio"))][-1]
    return f"""## 2. PD model: calibration, master scale and lifetime curves

| | |
|---|---|
| Purpose | 12-month point-in-time PD for each loan, rating grade, and lifetime PD for IFRS 9 |
| Version | `models/pd_calibration.json`, `processed/pd_term_structure.parquet` (Phase 4) |
| Calibration | Logistic fit of 12-month default on the frozen score, on the latest complete year ({", ".join(map(str, prod["years"]))}, {prod["loans"]:,} loans): intercept {prod["intercept"]:.3f}, slope {prod["slope"]:.5f} (the points scale implies {prod["slope_implied_by_points"]:.5f}) |
| Master scale | R1–R10 with fixed PD bounds; grade PD = observed rate of the calibration year, from {pct(gp["R1"], 2)} (R1) to {pct(gp["R10"], 2)} (R10) |
| Lifetime PD | Discrete-time competing risks (default vs repayment) by rating and term, 3-month recognition lag, anchored to the grade PD; {len(borrowed)} of {len(anchors)} rating-term curves borrow the term's shape in thin months |
| Report | [phase4_pd.md](phase4_pd.md) |

**Validation** (calibrate on one year, predict the next):

| Design | Calibrated on | 2017 predicted | 2017 observed | Grades failing Jeffreys (5%) |
|---|---|---|---|---|
| First attempt | {first["calibrated_on"]} | {pct(first["pd_2017_predicted"], 2)} | {pct(first["pd_2017_observed"], 2)} | {first["grades_failing_jeffreys"]} of 10 |
| Validated design | {final["calibrated_on"]} | {pct(final["pd_2017_predicted"], 2)} | {pct(final["pd_2017_observed"], 2)} | {final["grades_failing_jeffreys"]} of 10 |

The design was changed after the first 2017 result; both are reported.

**Lifetime back-test** (whole chain from origination, Phase 6):
{"; ".join(f"{r['vintage']}: lifetime PD {pct(r['pd_predicted'])} predicted vs {pct(r['default_rate'])} actual" for r in bt)}.
The over-prediction comes mostly from the first year, where the 2017 level is applied to
lower-risk vintages.

**Use and limits.** Point-in-time on 2017 loans. Older vintages defaulted less, and 2018
loans may default more (6-month defaults of {early["quarter"]} loans at {early["early_ratio"]:.2f}× the curve, Phase 7):
recalibrate when 2018 12-month outcomes are complete.
"""


def lgd_ead_card(s5) -> str:
    r = s5["by_rating"]
    lo = min(r, key=lambda x: x["lgd"])
    hi = max(r, key=lambda x: x["lgd"])
    st = {str(x["debt_settlement_flag"]): x for x in s5["by_settlement"]}
    e, ro = s5["ead_validation"], s5["open_book_runoff"]
    return f"""## 3. LGD model

| | |
|---|---|
| Purpose | Share of exposure lost after default, for IFRS 9 |
| Version | `models/lgd.json` (Phase 5) |
| Data | {s5["complete_workouts"]:,} of {s5["charged_off"]:,} charged-off loans, those defaulted at least 18 months before the snapshot (complete recovery workout) |
| Method | LGD = 1 − net recovery rate × discount factor; net recovery = recoveries − collection fees, over principal at default; recoveries discounted at the loan rate with a timing profile inferred from the cross-section (mean recovery month {s5["recovery_mean_month"]:.1f}); segment by rating, EAD-weighted |
| Result | {pct(s5["lgd_pooled"])} pooled ({pct(s5["lgd_pooled_undiscounted"])} undiscounted); {pct(s5["zero_recovery_share"])} of defaults recover nothing; by rating {pct(lo["lgd"])} ({lo["rating"]}) to {pct(hi["lgd"])} ({hi["rating"]}); pooled value used for {", ".join(s5["ratings_on_pooled"]) or "no rating"} (under 1,000 workouts) |
| Report | [phase5_lgd_ead.md](phase5_lgd_ead.md) |

**Use and limits.** LGD barely varies by rating, term or default year. Debt settlement does
matter (LGD {pct(st["True"]["lgd"])} with a settlement, {pct(st["False"]["lgd"])} without), but it is agreed after default, so it cannot
segment future losses. Recovery dates are not in the data. No cure is modelled for loans
31+ days late, which is conservative.

## 4. EAD model

| | |
|---|---|
| Purpose | Exposure in each future month, for IFRS 9 |
| Method | Contractual amortisation of today's principal at the original rate and instalment; prepayment left to the competing-risk PD curves (not applied twice); principal only |
| Check | On {e["loans"]:,} charged-off loans, actual / scheduled balance at default = {e["weighted_ratio"]:.3f} (EAD-weighted), {pct(e["share_within_5pct"])} within ±5%. Partly expected by construction, since the default month comes partly from payments made: a consistency check, not an independent validation |
| Open book | {ro["loans"]:,} loans; current loans at {pct(ro["current_loans_actual_vs_contractual"])} of their contractual balance (partial prepayment) |
| Report | [phase5_lgd_ead.md](phase5_lgd_ead.md) |
"""


def ecl_card(s6) -> str:
    st = {str(r["stage"]): r for r in s6["stages"]}
    sc = {r["scenario"]: r for r in s6["scenarios"]}
    rr = s6["roll_rates"]
    stage_rows = "\n".join(
        f"| {k} | {int(r['loans']):,} | {money(r['gca'])} | {money(r['ecl'])} | {pct(r['coverage'], 2)} |"
        for k, r in st.items()
    )
    sens_rows = "\n".join(
        f"| {r['case']} | {money(r['ecl'])} | {100 * (r['ecl'] / s6['ecl'] - 1):+.1f}% |"
        for r in s6["sensitivities"]
    )
    bt = "; ".join(
        f"{r['vintage']} {r['loss_predicted'] / r['loss_realised']:.2f}×" for r in s6["backtest"]
    )
    return f"""## 5. IFRS 9 expected credit loss

| | |
|---|---|
| Purpose | Loss allowance for the open book at {s6["snapshot"]} |
| Staging | Stage 3: 31+ days past due or default. Stage 2: 1–30 days past due, hardship plan, or current 12-month PD at least 2× origination (current PD = scorecard re-applied with March 2019 FICO). Stage 1: the rest |
| Engine | Monthly marginal PD (Phase 4 curves from the current month on book) × LGD × contractual EAD × discount at the loan rate; 12 months for stage 1, remaining life for stage 2, LGD × balance for stage 3 |
| Delinquent loans | Roll rates to 31+ days from steady-state bucket stocks: grace period {pct(rr["In Grace Period"])}, 16–30 days {pct(rr["Late (16-30 days)"])} |
| Scenarios | Single-factor (Vasicek) shifts of the 12-month PD for 24 months; rho {pct(s6["rho"], 0)} (estimated {pct(s6["rho_estimated"], 1)} on a recession-free period, floored at the Basel retail minimum); upside / base / downside weighted {pct(sc["upside"]["weight"], 0)} / {pct(sc["base"]["weight"], 0)} / {pct(sc["downside"]["weight"], 0)} |
| Report | [phase6_ecl.md](phase6_ecl.md) |

**Result: {money(s6["ecl"])} on {money(s6["gca"])} ({s6["loans"]:,} loans), coverage {pct(s6["coverage"], 2)}.**

| Stage | Loans | Gross carrying amount | ECL | Coverage |
|---|---|---|---|---|
{stage_rows}

Scenarios: upside {money(sc["upside"]["ecl"])}, base {money(sc["base"]["ecl"])}, downside {money(sc["downside"]["ecl"])}.
Back-test of lifetime loss on matured vintages (predicted / actual): {bt}.

**Sensitivities:**

| Case | ECL | Change |
|---|---|---|
{sens_rows}
"""


def register(s1, s3, s4, s6, s7) -> str:
    bm, sp = s3["benchmarks_test"], s3["splits"]
    first, final = s4["variants"]
    bt = s6["backtest"]
    ratios = [r["loss_predicted"] / r["loss_realised"] for r in bt]
    early = [r for r in s7["quarters"] if ok(r.get("early_ratio"))][-1]
    rows = [
        ("Data", "One snapshot (March 2019), no payment history: cures and delinquency transitions cannot be observed",
         f"Roll rates inferred from bucket stocks; ECL {change(s6, 'Delinquent')} without them", "Phase 6", "Use monthly performance tapes if available; back-test roll rates"),
        ("Data", "Booked loans only, no rejected applications",
         "Scorecard performance on the full applicant population is unknown; approval-policy changes appear as population drift", "Phases 3, 7", "Reject inference if application data becomes available"),
        ("Data", "Four bureau features are not reported before 2012",
         f"{pct(s7['missing_share_full_dev'])} of the 2007–2014 development sample sits in their MISSING bin; a false CSI above 0.25 if used as the stability reference", "Phases 3, 7", "Stability measured against 2013–2014; redevelop on 2016+ data"),
        ("Definition", "Default includes loans 31–120 days late, some of which cure",
         f"12-month default rate {s1['default_12m_rate_pct']:.2f}% vs {s1['default_12m_strict_rate_pct']:.2f}% with charged-off/default only", "Phase 1", "No action: immaterial"),
        ("Scorecard", "Interpretable logistic scorecard ranks less well than gradient boosting and the lender's grade",
         f"Test Gini {sp['test']['gini']:.3f} vs XGBoost {bm['xgboost']['gini']:.3f} and LC sub-grade {bm['lc_sub_grade']['gini']:.3f}", "Phase 3", "Accepted for transparency; XGBoost and sub-grade reported as benchmarks"),
        ("PD", "Calibration level is point-in-time on 2017 loans",
         f"Lifetime loss over-predicted {min(ratios):.2f}–{max(ratios):.2f}× on older matured vintages; {early['quarter']} early defaults {early['early_ratio']:.2f}× expected", "Phases 4, 6, 7", "Recalibrate on 2018 when 12-month outcomes are complete"),
        ("PD", "Calibration design chosen after seeing the 2017 hold-out",
         f"First design {pct(first['pd_2017_predicted'], 2)} vs validated {pct(final['pd_2017_predicted'], 2)} predicted ({pct(final['pd_2017_observed'], 2)} observed)", "Phase 4", "Both designs reported; confirm on 2018 outcomes"),
        ("PD", "Thin late months of rare rating-term pairs",
         "Those months borrow the term's all-grades curve scaled to the pair's relative risk", "Phase 4", "Tested; revisit with more seasoned data"),
        ("LGD", "No recovery dates in the data",
         "Recovery timing (and so discounting) inferred from the cross-section", "Phase 5", "Use dated recovery cash flows if available"),
        ("LGD / ECL", "No cure for stage 3; LGD measured on charged-off loans only",
         "Stage 3 coverage is conservative", "Phases 5, 6", "Estimate cure rates from performance history"),
        ("Staging", "Current PD updates FICO only, with WoE extrapolated below the scorecard's FICO range",
         f"ECL {change(s6, 'No FICO')} without extrapolation; the PD test then hardly triggers", "Phase 6", "Build a behavioural score on payment history"),
        ("Scenarios", "Systematic risk floored at a regulatory value; scenario weights and shifts are judgement",
         f"ECL {change(s6, 'Systematic')} with the estimated rho", "Phase 6", "Link scenarios to macro forecasts once a downturn is in the data"),
        ("ECL", "Effective interest rate ignores fees; EAD ignores accrued interest",
         "Small: about one month of interest", "Phases 5, 6", "Add fee amortisation if fee data is available"),
        ("Monitoring", "Thresholds are industry conventions; z-tests treat loans as independent",
         "Significance is overstated, so a status also needs a material gap", "Phase 7", "Calibrate thresholds on the model's own history"),
        ("Reporting", "Power BI project generated as text and checked by tests, not yet opened in Power BI Desktop",
         "A layout error would only show on opening", "Phase 8", "Open, refresh and fix before use"),
        ("Governance", "Developed and validated by the same person",
         "No independent challenge", "All", "Independent review before any real use"),
    ]  # fmt: skip
    body = "\n".join(
        f"| {i + 1} | {a} | {b} | {c} | {d} | {e} |" for i, (a, b, c, d, e) in enumerate(rows)
    )
    return f"""# Model risk register

Generated by `python -m credit_risk.reporting.cards` from the phase summaries.

| # | Area | Limitation | Impact (quantified where possible) | Where | Mitigation / next step |
|---|---|---|---|---|---|
{body}
"""


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    s1, s3, s4, s5, s6, s7 = (load(f"phase{i}_summary") for i in (1, 3, 4, 5, 6, 7))
    cards = f"""# Model cards

Generated by `python -m credit_risk.reporting.cards` from the phase summaries; each card
links to its full report. Limitations across all models: [model_risk_register.md](model_risk_register.md).

{scorecard_card(s1, s3, s7)}
{pd_card(s4, s6, s7)}
{lgd_ead_card(s5)}
{ecl_card(s6)}"""
    (DOCS / "model_cards.md").write_text(cards, encoding="utf-8")
    (DOCS / "model_risk_register.md").write_text(register(s1, s3, s4, s6, s7), encoding="utf-8")
    log.info("wrote model_cards.md and model_risk_register.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
