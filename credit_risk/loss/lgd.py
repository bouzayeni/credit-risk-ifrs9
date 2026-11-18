"""Loss given default (LGD) from charged-off loans with a complete recovery workout.

- Exposure at default (realised) = funded amount - principal repaid before default.
- Net recovery = recoveries - collection fees, capped to [0, EAD].
- Workout complete: the first missed payment is at least WORKOUT_MONTHS before the snapshot.
  Recoveries arrive months after charge-off; younger defaults would overstate LGD.
- Discounting (IFRS 9 measures losses at present value, at the effective interest rate): the
  data has no recovery dates, so timing comes from the cross-section, as the mean net
  recovery rate by months since default. Its increments say what share of recoveries arrives
  in each month; each loan's recoveries are discounted with that profile at its own rate.
  The profile mixes vintages, so it is an approximation.

LGD = 1 - net recovery rate x discount factor. Segment LGD = EAD-weighted average.
"""

import numpy as np
import pandas as pd

from credit_risk.data.clean import SNAPSHOT_DATE

WORKOUT_MONTHS = 18
TIMING_HORIZON = 36  # months since default covered by the recovery-timing profile
MIN_SEGMENT_DEFAULTS = 1_000


def default_period(df: pd.DataFrame) -> pd.Series:
    return df["issue_d"].dt.to_period("M") + df["months_to_default"].astype(int)


def months_since_default(df: pd.DataFrame) -> pd.Series:
    snap = SNAPSHOT_DATE.to_period("M")
    dp = default_period(df)
    return (snap.year * 12 + snap.month) - (dp.dt.year * 12 + dp.dt.month)


def realised(df: pd.DataFrame) -> pd.DataFrame:
    """Charged-off loans with positive exposure: EAD, net recovery rate, months since default."""
    co = df[df["loan_status"].eq("Charged Off")].copy()
    co["ead_at_default"] = co["funded_amnt"] - co["total_rec_prncp"]
    co = co[co["ead_at_default"] > 0]
    net = (co["recoveries"] - co["collection_recovery_fee"]).clip(lower=0)
    co["recovery_rate"] = (net / co["ead_at_default"]).clip(upper=1.0)
    co["months_since_default"] = months_since_default(co)
    co["workout_complete"] = co["months_since_default"] >= WORKOUT_MONTHS
    return co


def recovery_timing(co: pd.DataFrame) -> pd.Series:
    """Share of eventual recoveries received in each month after default (index 1..H).

    Cumulative profile = mean recovery rate of loans t months after default, made
    non-decreasing and scaled to its plateau (mean over months H/2..H).
    """
    by_month = co.groupby("months_since_default")["recovery_rate"].mean()
    curve = by_month.reindex(range(0, TIMING_HORIZON + 1)).interpolate().fillna(0.0).cummax()
    plateau = curve.loc[TIMING_HORIZON // 2 :].mean()
    cum = (curve / plateau).clip(upper=1.0)
    cum.iloc[-1] = 1.0
    shares = cum.diff().fillna(cum.iloc[0]).clip(lower=0)
    shares = shares.loc[1:]  # month 0 recoveries are treated as month 1
    shares.iloc[0] += float(cum.iloc[0])
    return shares / shares.sum()


def discount_factor(annual_rate_pct, timing: pd.Series) -> np.ndarray:
    """Expected discount factor of recoveries given the timing profile, at each loan's rate."""
    r = np.asarray(annual_rate_pct, dtype=float)[:, None] / 1200.0
    t = timing.index.to_numpy(dtype=float)[None, :]
    return ((1 + r) ** (-t) * timing.to_numpy()[None, :]).sum(axis=1)


def segment_lgd(co: pd.DataFrame, by: str) -> pd.DataFrame:
    g = co.groupby(by, observed=True)
    out = pd.DataFrame(
        {
            "defaults": g.size(),
            "ead": g["ead_at_default"].sum(),
            "recovery_rate": g.apply(
                lambda x: np.average(x["recovery_rate"], weights=x["ead_at_default"]),
                include_groups=False,
            ),
            "lgd": g.apply(
                lambda x: np.average(x["lgd"], weights=x["ead_at_default"]), include_groups=False
            ),
            "lgd_undiscounted": g.apply(
                lambda x: np.average(1 - x["recovery_rate"], weights=x["ead_at_default"]),
                include_groups=False,
            ),
        }
    )
    return out.reset_index()
