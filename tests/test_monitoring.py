import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from credit_risk.monitoring import metrics as mm


def test_psi_known_value_and_zero_for_identical():
    e = pd.Series({"a": 50, "b": 50})
    a = pd.Series({"a": 0.6, "b": 0.4})
    expected = (0.6 - 0.5) * np.log(0.6 / 0.5) + (0.4 - 0.5) * np.log(0.4 / 0.5)
    assert mm.psi(e, a) == pytest.approx(expected)
    assert mm.psi(e, e) == pytest.approx(0.0)
    assert mm.psi(e, pd.Series({"a": 10})) > 1  # an emptied bin is floored, not infinite


def test_status_thresholds():
    assert [mm.psi_status(v) for v in (0.05, 0.10, 0.2, 0.25)] == ["green", "amber", "amber", "red"]
    assert [mm.gini_status(g, 0.4) for g in (0.37, 0.35, 0.31)] == ["green", "amber", "red"]


def test_gini_and_standard_error_against_bootstrap():
    rng = np.random.default_rng(1)
    risk = rng.normal(size=20_000)
    y = rng.random(20_000) < 1 / (1 + np.exp(-(risk * 0.8 - 3)))
    g, se = mm.gini_with_se(y, risk)
    assert g == pytest.approx(2 * roc_auc_score(y, risk) - 1)
    boot = []
    for _ in range(200):
        i = rng.integers(0, len(y), len(y))
        boot.append(2 * roc_auc_score(y[i], risk[i]) - 1)
    assert se == pytest.approx(np.std(boot), rel=0.2)


def test_calibration_needs_significance_and_materiality():
    pds = np.full(100_000, 0.05)
    on_target = mm.calibration(np.r_[np.ones(5_000), np.zeros(95_000)], pds)
    assert on_target["status"] == "green" and on_target["ratio"] == pytest.approx(1.0)
    high = mm.calibration(np.r_[np.ones(6_500), np.zeros(93_500)], pds)  # +30%
    assert high["status"] == "red"
    amber = mm.calibration(np.r_[np.ones(5_750), np.zeros(94_250)], pds)  # +15%
    assert amber["status"] == "amber"
    small = mm.calibration(np.r_[np.ones(13), np.zeros(87)], np.full(100, 0.10))  # +30%, n=100
    assert small["status"] == "green"  # not significant


def test_score_from_spec_numeric_categorical_missing():
    spec = {"features": [
        {"name": "x", "kind": "numeric", "iv": 0.1, "edges": [1.0, float("inf")], "levels": {},
         "bins": [{"bin": "<= 1", "woe": 0.1, "points": 30},
                  {"bin": "> 1", "woe": -0.1, "points": 20},
                  {"bin": "MISSING", "woe": 0.0, "points": 25}]},
        {"name": "c", "kind": "categorical", "iv": 0.1, "edges": [], "levels": {"A": "A", "B": "B"},
         "bins": [{"bin": "A", "woe": 0.2, "points": 40}, {"bin": "B", "woe": -0.2, "points": 10},
                  {"bin": "RARE (<1% each)", "woe": 0.0, "points": 25}]},
    ]}  # fmt: skip
    df = pd.DataFrame({"x": [0.5, 3.0, np.nan], "c": ["A", "B", "Z"]})
    total, labels = mm.score(df, spec)
    assert total.tolist() == [70, 30, 50]
    assert labels["x"].tolist() == ["<= 1", "> 1", "MISSING"]
    assert labels["c"].tolist() == ["A", "B", "RARE (<1% each)"]
