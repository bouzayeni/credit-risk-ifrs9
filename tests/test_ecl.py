import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from credit_risk.ecl import engine, scenarios, staging
from credit_risk.pd.calibration import GRADES, Calibration

FICO = {
    "name": "fico",
    "coef": -0.69,
    "edges": [662.0, 672.0, 692.0, float("inf")],
    "bins": [
        {"bin": "<= 662", "woe": -0.4},
        {"bin": "(662, 672]", "woe": -0.2},
        {"bin": "(672, 692]", "woe": 0.0},
        {"bin": "> 692", "woe": 0.5},
    ],
}
SCORECARD = {"scale": {"pdo": 20}, "features": [FICO]}


def test_fico_woe_lookup_and_extrapolation():
    w = staging.FicoWoe(FICO)
    assert w([662, 663, 690, 800]).tolist() == [-0.4, -0.2, 0.0, 0.5]
    # Closed bins: representative FICO 662, 669.5, 684.5 -> WoE -0.4, -0.2, 0.0
    slope = np.polyfit([662, 669.5, 684.5], [-0.4, -0.2, 0.0], 1)[0]
    assert w.slope == pytest.approx(slope)
    assert w([612])[0] == pytest.approx(-0.4 - 50 * slope)
    assert w([400])[0] == pytest.approx(w([staging.FICO_FLOOR])[0])  # floored
    assert staging.FicoWoe(FICO, extrapolate=False)([612])[0] == -0.4


def test_current_pd_moves_only_with_fico():
    cal = Calibration(intercept=10.0, slope=-0.02)
    pd_orig = np.array([0.05, 0.05, 0.05, 0.05])
    now = staging.current_pd(pd_orig, [700, 700, 700, 700], [700, 680, np.nan, 0], SCORECARD, cal)
    assert now[[0, 2, 3]] == pytest.approx([0.05] * 3)  # same FICO / missing / LC's 0 code
    # 700 -> 680: WoE 0.5 -> 0.0, points change = factor * coef * 0.5 (negative)
    factor = 20 / np.log(2)
    logit = np.log(0.05 / 0.95) + cal.slope * (-factor * -0.69 * (0.0 - 0.5))
    assert now[1] == pytest.approx(1 / (1 + np.exp(-logit)))
    assert now[1] > 0.05


def test_stage_assignment():
    status = pd.Series(["Current", "Current", "Current", "In Grace Period", "Late (31-120 days)",
                        "Late (16-30 days)", "Default"])  # fmt: skip
    hardship = pd.Series([False, True, False, False, True, False, False])
    pd_orig = np.full(7, 0.05)
    pd_now = np.array([0.09, 0.05, 0.10, 0.05, 0.30, 0.05, 0.05])
    s = staging.assign_stage(status, hardship, pd_orig, pd_now)
    assert s["stage"].tolist() == [1, 2, 2, 2, 3, 2, 3]
    assert s["sicr_hardship"].tolist() == [False, True, False, False, False, False, False]
    assert s["sicr_pd"].tolist() == [False, False, True, False, False, False, False]


def test_scenario_pd_shift():
    assert scenarios.scenario_pd([0.05], 0.04, 0.0)[0] == pytest.approx(0.05)
    up, down = (
        scenarios.scenario_pd([0.05, 0.05], 0.04, -1.0)[0],
        scenarios.scenario_pd([0.05], 0.04, 2.0)[0],
    )
    assert up < 0.05 < down
    assert down == pytest.approx(norm.cdf(norm.ppf(0.05) + 2 * np.sqrt(0.04 / 0.96)))


def test_rho_recovered_from_simulated_vasicek_years():
    rng = np.random.default_rng(0)
    rho, rows = 0.03, []
    z = rng.standard_normal(40)
    z = (z - z.mean()) / z.std(ddof=1)  # unit variance, so only the estimator is tested
    for grade, pd_g in [("A", 0.03), ("B", 0.08)]:
        for year, zt in enumerate(z):
            p = norm.cdf((norm.ppf(pd_g) - np.sqrt(rho) * zt) / np.sqrt(1 - rho))
            n = 20_000
            d = np.zeros(n)
            d[: rng.binomial(n, p)] = 1
            rows.append(pd.DataFrame({"grade": grade, "year": year, "d": d}))
    df = pd.concat(rows, ignore_index=True)
    est = scenarios.estimate_rho(df["grade"], df["year"], df["d"])
    assert est["rho"] == pytest.approx(rho, rel=0.15)


def flat_tables(h_d, h_p):
    hd = np.full((len(GRADES), len(engine.TERMS), engine.HORIZON + 1), h_d)
    hp = np.full_like(hd, h_p)
    return hd, hp


def test_projection_constant_hazard_closed_form():
    hd, hp = flat_tables(0.01, 0.0)
    # Interest-free, no instalment: balance stays 1,000; LGD 1; no discounting.
    one = (np.array([0]), np.array([36]), np.array([0]), np.array([1000.0]), np.array([0.0]))
    res = engine.project(hd, hp, *one, np.array([0.0]), np.array([1.0]), np.array([1.0]), 24)
    assert res["pd_12m_fwd"][0] == pytest.approx(1 - 0.99**12)
    assert res["pd_lifetime_fwd"][0] == pytest.approx(1 - 0.99**36)
    assert res["ecl_12m"][0] == pytest.approx(1000 * (1 - 0.99**12))


def test_projection_respects_remaining_term_prepayment_and_shock():
    hd, hp = flat_tables(0.01, 0.0)
    args = (np.array([0]), np.array([36]), np.array([30]), np.array([1000.0]), np.array([0.0]),
            np.array([0.0]), np.array([1.0]))  # fmt: skip
    res = engine.project(hd, hp, *args, np.array([1.0]), 24)
    assert res["pd_lifetime_fwd"][0] == pytest.approx(1 - 0.99**6)  # 6 months left
    shocked = engine.project(hd, hp, *args, np.array([2.0]), 24)
    assert shocked["pd_lifetime_fwd"][0] == pytest.approx(1 - 0.98**6)
    hd2, hp2 = flat_tables(0.01, 0.05)
    with_prepay = engine.project(hd2, hp2, *args, np.array([1.0]), 24)
    assert with_prepay["pd_lifetime_fwd"][0] < res["pd_lifetime_fwd"][0]


def test_roll_rates_from_bucket_stocks():
    # Steady state: 100 loans/month enter grace, 60 reach 16-30 days, 45 reach 31+ days.
    # Stocks = flow x months in bucket: 50 in grace, 30 in 16-30, 135 in 31-120.
    status = pd.Series(
        ["In Grace Period"] * 50 + ["Late (16-30 days)"] * 30 + ["Late (31-120 days)"] * 135
        + ["Current"] * 1000
    )  # fmt: skip
    rolls = staging.roll_rates(status)
    assert rolls["In Grace Period"] == pytest.approx(0.45)
    assert rolls["Late (16-30 days)"] == pytest.approx(0.75)
