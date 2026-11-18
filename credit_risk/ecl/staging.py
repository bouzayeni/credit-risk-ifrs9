"""IFRS 9 staging: credit-impaired, significant increase in credit risk (SICR), performing.

Current PD. The frozen application scorecard is re-applied with the borrower's FICO as of
the reporting date (`last_fico`, pulled monthly by Lending Club) in place of the FICO at
application; the other features are as at origination. Only the FICO points move:

    score_now = score_orig - factor * beta_fico * (WoE(fico_now) - WoE(fico_orig))
    PD_now    = calibration(score_now)

The scorecard never saw FICO below Lending Club's minimum (about 660), so its lowest FICO bin
would treat 660 and 560 alike. Below that bin, WoE is extrapolated along the straight line
fitted to the bin WoEs (FICO floored at 500). This is a model assumption; the report shows
the effect of switching it off.

Stages (first match wins):
  3  credit-impaired: 31+ days past due or in default (the Phase 1 default definition).
  2  SICR: 1-30 days past due (grace period or 16-30 days late), on a hardship plan
     (forbearance), or PD_now >= SICR_MULTIPLE x PD at origination.
  1  everything else.
The PD test compares 12-month PDs as a proxy for lifetime PDs (allowed by IFRS 9 B5.5.13
when the pattern of default risk over time does not differ materially); both come from the
same calibration, so their ratio reflects the change in the borrower, not in the economy.
"""

import numpy as np
import pandas as pd

from credit_risk.pd.calibration import Calibration

SICR_MULTIPLE = 2.0
FICO_FLOOR = 500.0
STAGE3_STATUSES = ("Late (31-120 days)", "Default")
DELINQUENT_STATUSES = ("In Grace Period", "Late (16-30 days)")
# Months a loan spends in each delinquency bucket if it does not cure (15, 15 and 90 days).
BUCKET_MONTHS = {"In Grace Period": 0.5, "Late (16-30 days)": 0.5, "Late (31-120 days)": 3.0}


def roll_rates(status: pd.Series) -> dict[str, float]:
    """Probability that a loan now 1-30 days late reaches 31+ days (the default definition).

    The data is one snapshot with no payment history, so transitions come from stocks: in a
    steady state, the monthly flow into a bucket is its stock divided by the months a loan
    spends there, and every loan reaching 31+ days passed through the earlier buckets. The
    roll rate from a bucket is the flow into 31+ days over the flow into that bucket.
    Cures shorten the stay in every bucket, which understates each inflow by an unknown
    amount, so the error can go either way; the report shows ECL without this adjustment.
    """
    counts = status.value_counts()
    flow = {b: counts.get(b, 0) / m for b, m in BUCKET_MONTHS.items()}
    into_default = flow["Late (31-120 days)"]
    return {b: float(min(into_default / flow[b], 1.0)) for b in DELINQUENT_STATUSES if flow[b] > 0}


class FicoWoe:
    """WoE of a FICO score from the scorecard's FICO bins, with linear extrapolation below."""

    def __init__(self, feature: dict, extrapolate: bool = True):
        self.edges = np.asarray(feature["edges"], dtype=float)
        woes = [b["woe"] for b in feature["bins"] if b["bin"] != "MISSING"]
        self.woe = np.asarray(woes[: len(self.edges)], dtype=float)
        self.extrapolate = extrapolate
        # Representative FICO of each closed bin (lower, upper]: values step by 5, so the
        # mean of lower+5 .. upper. The lowest bin holds the minimum, its upper edge.
        lower = np.concatenate([[self.edges[0] - 5], self.edges[:-1]])
        mid = (lower + 5 + self.edges) / 2
        closed = np.isfinite(self.edges)
        self.slope = float(np.polyfit(mid[closed], self.woe[closed], 1)[0])

    def __call__(self, fico) -> np.ndarray:
        x = np.asarray(fico, dtype=float)
        idx = np.minimum(np.searchsorted(self.edges, x, side="left"), len(self.edges) - 1)
        out = self.woe[idx]
        if self.extrapolate:
            below = x < self.edges[0]
            floored = np.maximum(x, FICO_FLOOR)
            out = np.where(below, self.woe[0] + self.slope * (floored - self.edges[0]), out)
        return out


def current_pd(pd_orig, fico_orig, fico_now, scorecard: dict, calibration: Calibration,
               extrapolate: bool = True) -> np.ndarray:  # fmt: skip
    """12-month PD with today's FICO; loans without an updated FICO keep their original PD."""
    feature = next(f for f in scorecard["features"] if f["name"] == "fico")
    woe = FicoWoe(feature, extrapolate)
    factor = scorecard["scale"]["pdo"] / np.log(2)
    f_orig = np.asarray(fico_orig, dtype=float)
    f_now = np.asarray(fico_now, dtype=float)
    f_now = np.where(np.isnan(f_now) | (f_now < 300), f_orig, f_now)
    delta_points = -factor * feature["coef"] * (woe(f_now) - woe(f_orig))
    p = np.asarray(pd_orig, dtype=float)
    logit = np.log(p / (1 - p)) + calibration.slope * delta_points
    return 1 / (1 + np.exp(-logit))


def assign_stage(status: pd.Series, hardship: pd.Series, pd_orig, pd_now,
                 multiple: float = SICR_MULTIPLE) -> pd.DataFrame:  # fmt: skip
    """Stage and the reason for it. Reasons for stage 2 are flagged separately (they overlap)."""
    impaired = status.isin(STAGE3_STATUSES).to_numpy()
    delinquent = status.isin(DELINQUENT_STATUSES).to_numpy()
    forborne = hardship.fillna(False).astype(bool).to_numpy()
    pd_up = np.asarray(pd_now) >= multiple * np.asarray(pd_orig)
    stage = np.where(impaired, 3, np.where(delinquent | forborne | pd_up, 2, 1))
    return pd.DataFrame(
        {
            "stage": stage,
            "sicr_delinquent": delinquent & ~impaired,
            "sicr_hardship": forborne & ~impaired,
            "sicr_pd": pd_up & ~impaired,
        },
        index=status.index,
    )
