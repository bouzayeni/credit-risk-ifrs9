"""Monthly ECL projection for performing loans (stages 1 and 2).

For a loan at month on book m with R contractual months left, month j = 1..R ahead:

    marginal PD_j = h_d(m + j) * S_(j-1),   S_j = prod_(i <= j) (1 - h_d(m + i) - h_p(m + i))
    EAD_j         = contractual balance after j - 1 more payments
    loss_j        = marginal PD_j * LGD * EAD_j / (1 + r)^j

h_d and h_p are the Phase 4 default and repayment hazards of the loan's rating and term
(anchored to the point-in-time grade PD), so the projection is conditional on the loan being
alive today, and prepayment reduces expected losses without being applied to EAD again.
LGD is already discounted to the default date; r is the loan's monthly contract rate (its
effective interest rate, ignoring fees). Scenario multipliers scale h_d for the first
`shock_months` months.

12-month ECL sums j = 1..min(12, R); lifetime ECL sums j = 1..R.
"""

import numpy as np
import pandas as pd

from credit_risk.loss import ead
from credit_risk.pd.calibration import GRADES

TERMS = (36, 60)
HORIZON = 60
CHUNK = 50_000


def hazard_tables(curves: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Arrays [grade, term, mob] of anchored default and repayment hazards (mob 0..HORIZON)."""
    hd = np.zeros((len(GRADES), len(TERMS), HORIZON + 1))
    hp = np.zeros_like(hd)
    g_idx = {g: i for i, g in enumerate(GRADES)}
    t_idx = {t: i for i, t in enumerate(TERMS)}
    for r in curves.itertuples(index=False):
        gi, ti, m = g_idx[r.rating], t_idx[int(r.term)], int(r.mob)
        hd[gi, ti, m] = r.hazard_default_scaled
        hp[gi, ti, m] = r.hazard_repaid
    return hd, hp


def project(
    hd_table,
    hp_table,
    grade_idx,
    term,
    mob,
    balance,
    rate_pct,
    installment,
    lgd,
    multiplier,
    shock_months: int,
    discount: bool = True,
) -> dict[str, np.ndarray]:
    """12-month and lifetime ECL and PD for each loan (arrays of equal length)."""
    out = {k: [] for k in ("ecl_12m", "ecl_lifetime", "pd_12m_fwd", "pd_lifetime_fwd")}
    j = np.arange(1, HORIZON + 1)[None, :]
    term_idx_all = np.searchsorted(np.asarray(TERMS), np.asarray(term))
    for s in range(0, len(grade_idx), CHUNK):
        sl = slice(s, s + CHUNK)
        g, ti = np.asarray(grade_idx[sl])[:, None], term_idx_all[sl][:, None]
        t, m = np.asarray(term[sl])[:, None], np.asarray(mob[sl])[:, None]
        remaining = np.maximum(t - m, 1)
        alive = j <= remaining
        mob_idx = np.clip(m + j, 1, t)
        h_p = np.where(alive, hp_table[g, ti, mob_idx], 0.0)
        mult = np.where(j <= shock_months, np.asarray(multiplier[sl])[:, None], 1.0)
        h_d = np.where(alive, np.minimum(hd_table[g, ti, mob_idx] * mult, 1 - h_p), 0.0)
        surv = np.cumprod(1 - h_d - h_p, axis=1)
        surv_prev = np.concatenate([np.ones((surv.shape[0], 1)), surv[:, :-1]], axis=1)
        marginal = h_d * surv_prev
        exposure = ead.schedule(balance[sl], rate_pct[sl], installment[sl], HORIZON)
        r = np.asarray(rate_pct[sl], dtype=float)[:, None] / 1200.0
        df = (1 + r) ** (-j) if discount else 1.0
        loss = marginal * exposure * np.asarray(lgd[sl])[:, None] * df
        out["ecl_12m"].append(loss[:, :12].sum(axis=1))
        out["ecl_lifetime"].append(loss.sum(axis=1))
        out["pd_12m_fwd"].append(marginal[:, :12].sum(axis=1))
        out["pd_lifetime_fwd"].append(marginal.sum(axis=1))
    return {k: np.concatenate(v) for k, v in out.items()}
