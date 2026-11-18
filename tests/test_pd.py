import numpy as np
import pandas as pd
import pytest

from credit_risk.pd import calibration as c
from credit_risk.pd import term_structure as ts


def test_calibration_recovers_a_known_logistic_relation():
    rng = np.random.default_rng(0)
    score = pd.Series(rng.normal(580, 20, 200_000))
    true = c.Calibration(intercept=17.0, slope=-0.0347)
    default = pd.Series(rng.uniform(size=len(score)) < true.pd(score))
    fit = c.fit_calibration(score, default)
    assert fit.slope == pytest.approx(true.slope, rel=0.05)
    assert fit.pd(580) == pytest.approx(true.pd(580), rel=0.05)


def test_grades_follow_bounds_inclusively():
    grades = c.assign_grade([0.001, 0.015, 0.0151, 0.19, 0.5])
    assert list(grades) == ["R1", "R1", "R2", "R9", "R10"]
    assert len(c.GRADES) == len(c.GRADE_BOUNDS) + 1


def test_jeffreys_flags_an_underestimated_grade():
    assert c.jeffreys_p_value(defaults=150, loans=1000, pd_grade=0.10) < 0.001  # 15% vs 10%
    assert c.jeffreys_p_value(defaults=100, loans=1000, pd_grade=0.10) == pytest.approx(
        0.5, abs=0.05
    )
    assert c.jeffreys_p_value(defaults=50, loans=1000, pd_grade=0.10) > 0.999


def loans(rows):
    df = pd.DataFrame(rows)
    for col in ("issue_d", "last_pymnt_d"):
        df[col] = pd.to_datetime(df[col])
    return df


def test_exit_times_and_recognition_lag():
    df = loans(
        [
            # defaulted at MOB 5, well inside the window
            {
                "issue_d": "2016-01-01",
                "last_pymnt_d": "2016-05-01",
                "loan_status": "Charged Off",
                "is_default": True,
                "months_to_default": 5,
                "months_observed": 38,
            },
            # repaid at MOB 10
            {
                "issue_d": "2016-01-01",
                "last_pymnt_d": "2016-11-01",
                "loan_status": "Fully Paid",
                "is_default": False,
                "months_to_default": np.nan,
                "months_observed": 38,
            },
            # still current: censored 3 months before the snapshot
            {
                "issue_d": "2018-01-01",
                "last_pymnt_d": "2019-03-01",
                "loan_status": "Current",
                "is_default": False,
                "months_to_default": np.nan,
                "months_observed": 14,
            },
            # default in the last 3 months: not yet fully recognised, so censored
            {
                "issue_d": "2018-01-01",
                "last_pymnt_d": "2018-12-01",
                "loan_status": "Late (31-120 days)",
                "is_default": True,
                "months_to_default": 13,
                "months_observed": 14,
            },
        ]
    )
    e = ts.exit_times(df)
    assert e["exit_mob"].tolist() == [5, 10, 11, 11]
    assert e["exit_type"].tolist() == ["default", "repaid", "censored", "censored"]


def test_hazards_and_competing_risks():
    # 100 loans: 10 default at MOB 1, 20 repay at MOB 2, the rest censored at MOB 3
    exits = pd.DataFrame(
        {
            "exit_mob": [1] * 10 + [2] * 20 + [3] * 70,
            "exit_type": ["default"] * 10 + ["repaid"] * 20 + ["censored"] * 70,
        }
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ts, "MIN_AT_RISK", 1)
        h = ts.hazards(exits, max_mob=3)
    assert h["at_risk"].tolist() == [100, 90, 70]
    assert h["hazard_default"].tolist() == [0.1, 0.0, 0.0]
    assert h["hazard_repaid"].tolist() == pytest.approx([0.0, 20 / 90, 0.0])
    cum = ts.cumulate(h)
    assert cum["survival"].tolist() == pytest.approx([0.9, 0.7, 0.7])
    assert cum["cumulative_pd"].iloc[-1] == pytest.approx(0.1)


def test_thin_months_carry_the_last_reliable_hazard_without_fallback():
    exits = pd.DataFrame(
        {
            "exit_mob": [1] * 5 + [2] * 5 + [9] * 3,
            "exit_type": ["default"] * 5 + ["default"] * 5 + ["censored"] * 3,
        }
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ts, "MIN_AT_RISK", 5)
        h = ts.hazards(exits, max_mob=3)
    assert h["borrowed"].tolist() == [False, False, True]
    assert h["hazard_default"].iloc[2] == h["hazard_default"].iloc[1]


def test_thin_months_borrow_the_fallback_shape_scaled_to_relative_risk():
    # 200 loans at risk in month 1 with 4 defaults (2%); the rest are censored at month 2, so
    # months 2-3 are thin. Fallback: 1% default and 3% repaid hazard every month.
    exits = pd.DataFrame(
        {
            "exit_mob": [1] * 4 + [2] * 196,
            "exit_type": ["default"] * 4 + ["censored"] * 196,
        }
    )
    fallback = pd.DataFrame(
        {"mob": [1, 2, 3], "hazard_default": [0.01] * 3, "hazard_repaid": [0.03] * 3}
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ts, "MIN_AT_RISK", 197)
        h = ts.hazards(exits, max_mob=3, fallback=fallback)
    assert h["borrowed"].tolist() == [False, True, True]
    assert h["relative_risk"].iloc[0] == pytest.approx(2.0)  # 4 actual vs 2 expected
    assert h["hazard_default"].tolist() == pytest.approx([0.02, 0.02, 0.02])
    assert h["hazard_repaid"].tolist()[1:] == pytest.approx([0.03, 0.03])


def test_anchoring_hits_the_target_12m_pd():
    curve = pd.DataFrame({"mob": range(1, 37), "hazard_default": 0.004, "hazard_repaid": 0.02})
    scale = ts.anchor_scale(curve, target_pd_12m=0.06)
    anchored = ts.cumulate(curve, scale)
    assert anchored["cumulative_pd"].iloc[11] == pytest.approx(0.06, abs=1e-6)
    assert scale > 1
    assert anchored["cumulative_pd"].is_monotonic_increasing


def test_grade_pd_is_observed_rate_and_never_decreases():
    grades = pd.Series(["R1"] * 100 + ["R2"] * 100 + ["R3"] * 100)
    default = pd.Series([1] * 2 + [0] * 98 + [1] * 5 + [0] * 95 + [1] * 4 + [0] * 96)
    pds = c.grade_pd(grades, default)
    assert pds["R1"] == 0.02 and pds["R2"] == 0.05
    assert pds["R3"] == 0.05  # observed 4% < R2: raised to keep the scale monotonic
    assert pds["R10"] > pds["R9"] > pds["R3"]  # empty grades: band midpoints
