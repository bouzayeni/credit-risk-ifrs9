"""Forward-looking scenarios through a single-factor (Vasicek) model.

Lending Club's history cannot support a regression of default rates on macro variables:
2010-2017 had falling unemployment while LC's default rates rose as it loosened credit, so a
fitted link would have the wrong sign. Instead, the systematic risk is measured directly from
the year-to-year swings of default rates within each rating grade, and scenarios are shifts
of that systematic factor:

    probit(PD_scenario) = probit(PD_base) + k * sqrt(rho / (1 - rho))

where k is the scenario's adverse shift in standard deviations of the factor. In the Vasicek
model, probit(default rate of year t) varies across years with variance rho / (1 - rho)
(plus binomial noise, removed below), which gives rho.

Year-to-year swings in LC's grades also contain underwriting changes, so rho is, if anything,
overstated; the report compares it with the Basel range for retail exposures.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

# name: (adverse shift k in factor standard deviations, probability weight)
SCENARIOS = {"upside": (-1.0, 0.30), "base": (0.0, 0.50), "downside": (2.0, 0.20)}
SHOCK_MONTHS = 24  # scenario PDs apply for two years, then hazards revert to the base curve
MIN_CELL_LOANS = 500
# Lower end of the Basel IRB asset correlation for other retail exposures. The estimation
# period (2010-2017) has no recession, so the estimate is floored here.
RHO_FLOOR = 0.03


def estimate_rho(grade: pd.Series, year: pd.Series, default: pd.Series) -> dict:
    """Asset correlation from the across-year variance of probit default rates per grade."""
    cells = (
        pd.DataFrame({"grade": grade, "year": year, "d": default.astype(float)})
        .groupby(["grade", "year"], observed=True)["d"]
        .agg(["mean", "size"])
    )
    cells = cells[(cells["size"] >= MIN_CELL_LOANS) & (cells["mean"] > 0) & (cells["mean"] < 1)]
    p, n = cells["mean"], cells["size"]
    z = pd.Series(norm.ppf(p), index=cells.index)
    noise = p * (1 - p) / n / norm.pdf(z) ** 2  # delta method: sampling variance of probit(p)
    rows = []
    for g, zg in z.groupby(level="grade"):
        if len(zg) < 4:
            continue
        excess = max(float(zg.var(ddof=1) - noise.loc[g].mean()), 0.0)
        rows.append({"grade": g, "years": len(zg), "probit_var": float(zg.var(ddof=1)),
                     "noise_var": float(noise.loc[g].mean()), "excess_var": excess})  # fmt: skip
    table = pd.DataFrame(rows)
    v = float(np.average(table["excess_var"], weights=table["years"]))
    return {"rho": v / (1 + v), "factor_sd": float(np.sqrt(v)), "by_grade": table}


def scenario_pd(pd_base, rho: float, k: float) -> np.ndarray:
    shift = k * np.sqrt(rho / (1 - rho))
    return norm.cdf(norm.ppf(np.asarray(pd_base, dtype=float)) + shift)
