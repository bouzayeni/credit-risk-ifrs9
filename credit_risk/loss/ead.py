"""Exposure at default (EAD) from the contractual amortisation schedule.

Lending Club loans are fixed-rate and fully amortising, so with monthly rate r and
instalment I, the balance after k more payments is

    B_k = B_0 (1 + r)^k - I ((1 + r)^k - 1) / r,   floored at 0.

For IFRS 9 the EAD of an open loan in a future month is its contractual balance then.
Prepayment is not modelled here: it is already a competing risk in the lifetime PD curves
(Phase 4), whose marginal PDs are unconditional, so applying it again would double count.
EAD is principal only; accrued interest (at most about one month) is left out.
"""

import numpy as np


def balance_after(balance, annual_rate_pct, installment, k) -> np.ndarray:
    """Contractual principal outstanding after k more monthly payments (broadcasts)."""
    b0 = np.asarray(balance, dtype=float)
    r = np.asarray(annual_rate_pct, dtype=float) / 1200.0
    inst = np.asarray(installment, dtype=float)
    k = np.asarray(k, dtype=float)
    growth = (1 + r) ** k
    with np.errstate(divide="ignore", invalid="ignore"):
        paid = np.where(r > 0, inst * (growth - 1) / r, inst * k)
    return np.maximum(b0 * growth - paid, 0.0)


def schedule(balance, annual_rate_pct, installment, months: int) -> np.ndarray:
    """Balances at the start of each of the next `months` months (column 0 = today)."""
    k = np.arange(months)[None, :]
    return balance_after(
        np.asarray(balance, dtype=float)[:, None],
        np.asarray(annual_rate_pct, dtype=float)[:, None],
        np.asarray(installment, dtype=float)[:, None],
        k,
    )
