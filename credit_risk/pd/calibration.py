"""12-month PD calibration and the rating master scale.

Calibration maps the frozen scorecard's points to a 12-month probability of default with a
logistic fit, PD = 1 / (1 + exp(-(a + b * score))), on recent vintages. If the scorecard's
points scale is right, b is close to -ln(2) / PDO and only the level a moves.

The master scale buckets PDs into rating grades with fixed, geometrically spaced PD
boundaries, so a grade means the same risk whatever the period. Each grade's PD is the
observed 12-month default rate of the calibration loans in it.

The Jeffreys test (used by the ECB for PD back-testing) checks each grade: with D defaults out
of N loans, the posterior of the true default rate is Beta(D + 0.5, N - D + 0.5); the p-value
is the posterior probability that the true rate is at or below the grade PD. A small p-value
means the grade PD is too low.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression

# Upper PD bounds of grades R1..R9; R10 is everything above the last bound.
GRADE_BOUNDS = (0.015, 0.0225, 0.0325, 0.045, 0.06, 0.08, 0.105, 0.14, 0.19)
GRADES = tuple(f"R{i}" for i in range(1, len(GRADE_BOUNDS) + 2))


@dataclass(frozen=True)
class Calibration:
    intercept: float
    slope: float

    def pd(self, score) -> np.ndarray:
        return 1 / (1 + np.exp(-(self.intercept + self.slope * np.asarray(score, dtype=float))))


def fit_calibration(score: pd.Series, default: pd.Series) -> Calibration:
    model = LogisticRegression(C=1e6, max_iter=500).fit(
        score.to_frame("score"), default.astype(int)
    )
    return Calibration(float(model.intercept_[0]), float(model.coef_[0][0]))


def assign_grade(pd_values) -> np.ndarray:
    """Rating grade of each PD (upper bounds inclusive)."""
    idx = np.searchsorted(np.asarray(GRADE_BOUNDS), np.asarray(pd_values), side="left")
    return np.asarray(GRADES, dtype=object)[idx]


def grade_pd(grades: pd.Series, values: pd.Series) -> dict[str, float]:
    """PD of each grade = mean of `values` over its loans.

    Pass default flags to get each grade's observed default rate (the method used), or model
    PDs to get the mean fitted PD. Grades are forced to be non-decreasing (a riskier grade never
    gets a lower PD); empty grades get their band midpoint.
    """
    means = pd.Series(values.to_numpy(dtype=float), index=grades.to_numpy()).groupby(level=0).mean()
    bounds = (0.0, *GRADE_BOUNDS, 0.30)
    raw = [
        float(means[g]) if g in means.index else (bounds[i] + bounds[i + 1]) / 2
        for i, g in enumerate(GRADES)
    ]
    return dict(zip(GRADES, np.maximum.accumulate(raw).tolist(), strict=True))


def jeffreys_p_value(defaults: int, loans: int, pd_grade: float) -> float:
    return float(beta.cdf(pd_grade, defaults + 0.5, loans - defaults + 0.5))


def backtest(grades: pd.Series, default: pd.Series, grade_pds: dict[str, float]) -> pd.DataFrame:
    """Observed vs predicted default rate by grade, with the Jeffreys test."""
    df = pd.DataFrame({"grade": grades.to_numpy(), "default": default.astype(int).to_numpy()})
    g = df.groupby("grade")["default"].agg(loans="count", defaults="sum").reindex(GRADES)
    g = g.dropna().astype(int)
    g["observed_rate"] = g["defaults"] / g["loans"]
    g["grade_pd"] = [grade_pds[i] for i in g.index]
    g["jeffreys_p"] = [jeffreys_p_value(d, n, p) for d, n, p in
                       zip(g["defaults"], g["loans"], g["grade_pd"], strict=True)]  # fmt: skip
    return g.rename_axis("grade").reset_index()
