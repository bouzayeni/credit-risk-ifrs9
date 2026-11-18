"""Lifetime PD term structure by rating grade and term, from all vintages.

Discrete-time competing risks, month on book (MOB) t = 1, 2, ...:

- a loan leaves the risk set by defaulting (first missed payment at MOB months_to_default),
  by repaying in full (MOB of its last payment), or by censoring (still open at the end of
  its observation window);
- the observation window ends RECOGNITION_LAG months before the snapshot, because a payment
  missed in the last months does not yet show as 31+ days late. Events after that are treated
  as censored, so recent vintages do not understate defaults;
- hazards: h_d(t) = defaults at t / at risk at t, h_p(t) = repayments at t / at risk at t,
  where "at risk at t" = loans whose exit MOB is t or later. Months with too few loans at risk
  borrow the shape of the term's all-grades curve, scaled to the group's relative risk;
- survival S(t) = prod over s <= t of (1 - h_d(s) - h_p(s)); marginal (unconditional) PD
  in month t = S(t-1) * h_d(t); cumulative PD = running sum of marginal PDs.

Prepayment matters: a loan repaid early can no longer default, so lifetime PD is lower than
compounding the default hazard alone would suggest.

Hazards come from pooled vintages (closer to through-the-cycle). To make the curve agree with
the point-in-time 12-month PD of the rating scale, each curve's default hazards are scaled by
one factor so that its 12-month cumulative PD equals the grade PD (anchoring).
"""

import numpy as np
import pandas as pd

from credit_risk.data.clean import RECOGNITION_LAG_MONTHS

MIN_AT_RISK = 1_000  # below this, a month's hazards are too noisy to use on their own
HORIZON_12M = 12


def exit_times(loans: pd.DataFrame) -> pd.DataFrame:
    """Exit MOB and exit type ('default', 'repaid', 'censored') of each loan.

    Expects issue_d, last_pymnt_d, loan_status, is_default, months_to_default, months_observed.
    """
    window = (loans["months_observed"] - RECOGNITION_LAG_MONTHS).clip(lower=0).astype(float)
    paid_mob = (
        (loans["last_pymnt_d"].dt.year - loans["issue_d"].dt.year) * 12
        + (loans["last_pymnt_d"].dt.month - loans["issue_d"].dt.month)
    ).astype(float)

    event_time = np.where(
        loans["is_default"],
        loans["months_to_default"].astype(float),
        np.where(loans["loan_status"].eq("Fully Paid"), paid_mob.clip(lower=1), np.inf),
    )
    event_type = np.where(
        loans["is_default"],
        "default",
        np.where(loans["loan_status"].eq("Fully Paid"), "repaid", "censored"),
    )
    observed = event_time <= window
    return pd.DataFrame(
        {
            "exit_mob": np.where(observed, event_time, window).astype(int),
            "exit_type": np.where(observed, event_type, "censored"),
        },
        index=loans.index,
    )


def hazards(
    exits: pd.DataFrame, max_mob: int, fallback: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Default and repayment hazards for MOB 1..max_mob from one group's exits.

    Months with fewer than MIN_AT_RISK loans at risk are too noisy to use. With a `fallback`
    curve (e.g. the same term, all grades), those months borrow its shape: the fallback default
    hazard times this group's relative risk over its reliable months (actual defaults divided by
    the defaults the fallback would predict there), and the fallback repayment hazard. Without
    one, the last reliable month is carried forward.
    """
    mob = exits["exit_mob"].clip(upper=max_mob + 1).to_numpy()
    n = max_mob + 2
    leaving = np.bincount(mob, minlength=n)
    at_risk = leaving[::-1].cumsum()[::-1]  # exit MOB >= t
    d = np.bincount(mob[exits["exit_type"].eq("default").to_numpy()], minlength=n)
    p = np.bincount(mob[exits["exit_type"].eq("repaid").to_numpy()], minlength=n)
    t = np.arange(1, max_mob + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        h_d = np.where(at_risk[t] > 0, d[t] / at_risk[t], np.nan)
        h_p = np.where(at_risk[t] > 0, p[t] / at_risk[t], np.nan)
    out = pd.DataFrame(
        {
            "mob": t,
            "at_risk": at_risk[t],
            "defaults": d[t],
            "repaid": p[t],
            "hazard_default": h_d,
            "hazard_repaid": h_p,
        }
    )
    thin = (out["at_risk"] < MIN_AT_RISK).to_numpy()
    out["borrowed"] = thin
    out["relative_risk"] = 1.0
    if not thin.any():
        return out
    if fallback is not None:
        fb_d = fallback["hazard_default"].to_numpy()[: len(out)]
        fb_p = fallback["hazard_repaid"].to_numpy()[: len(out)]
        expected = float((out["at_risk"].to_numpy()[~thin] * fb_d[~thin]).sum())
        rel = float(out["defaults"].to_numpy()[~thin].sum() / expected) if expected > 0 else 1.0
        out.loc[thin, "hazard_default"] = rel * fb_d[thin]
        out.loc[thin, "hazard_repaid"] = fb_p[thin]
        out["relative_risk"] = rel
    else:
        out.loc[thin, ["hazard_default", "hazard_repaid"]] = np.nan
        out[["hazard_default", "hazard_repaid"]] = out[["hazard_default", "hazard_repaid"]].ffill()
    return out


def cumulate(curve: pd.DataFrame, default_scale: float = 1.0) -> pd.DataFrame:
    """Survival, marginal and cumulative PD from hazards (default hazards scaled)."""
    out = curve.copy()
    h_d = (out["hazard_default"] * default_scale).clip(upper=1.0)
    h_p = out["hazard_repaid"].clip(upper=1.0 - h_d)
    survival = (1 - h_d - h_p).cumprod()
    prev = np.concatenate([[1.0], survival.to_numpy()[:-1]])
    out["hazard_default_scaled"] = h_d
    out["survival"] = survival
    out["marginal_pd"] = prev * h_d
    out["cumulative_pd"] = out["marginal_pd"].cumsum()
    return out


def anchor_scale(curve: pd.DataFrame, target_pd_12m: float) -> float:
    """Factor on default hazards making the 12-month cumulative PD equal the target."""
    lo, hi = 0.0, 20.0
    for _ in range(60):  # bisection; cumulative PD rises with the factor
        mid = (lo + hi) / 2
        pd12 = cumulate(curve, mid)["cumulative_pd"].iloc[HORIZON_12M - 1]
        lo, hi = (mid, hi) if pd12 < target_pd_12m else (lo, mid)
    return (lo + hi) / 2
