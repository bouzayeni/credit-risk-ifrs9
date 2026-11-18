import numpy as np
import pandas as pd
import pytest

from credit_risk.scorecard import binning as b
from credit_risk.scorecard import metrics as m


def synthetic(n=20_000, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 100, n)
    p_bad = 1 / (1 + np.exp(-(x - 70) / 10))  # risk rises with x
    y = (rng.uniform(size=n) < p_bad).astype(int)
    xs = pd.Series(x)
    xs[rng.uniform(size=n) < 0.03] = np.nan
    return xs, pd.Series(y)


def test_woe_sign_and_iv():
    woe, iv = b.woe_iv(np.array([10, 90]), np.array([90, 10]))
    assert woe[0] > 0 > woe[1]  # first bin mostly goods: safer than average
    assert iv > 1


def test_monotonic_merge_pools_violators():
    edges = [1, 2, 3, np.inf]
    kept = b.monotonic_merge(edges, bads=[10, 30, 20, 40], counts=[100] * 4, increasing=True)
    assert kept == [1, 3, np.inf]  # bins 2 and 3 (30%, 20%) pooled into 25%


def test_merge_small_joins_closest_neighbour():
    kept = b.merge_small([1, 2, 3, np.inf], bads=[10, 1, 40, 50], counts=[100, 5, 100, 100],
                         min_count=50)  # fmt: skip
    assert kept == [2, 3, np.inf]  # tiny bin (20%) joins bin 1 (10%), not bin 3 (40%)


def test_numeric_bins_are_monotonic_large_and_keep_missing():
    x, y = synthetic()
    fb = b.fit("x", x, y)
    t = fb.table()
    numeric = t[t["bin"] != b.MISSING]
    assert fb.increasing
    assert numeric["bad_rate"].is_monotonic_increasing
    assert (numeric["share"] * len(x) >= b.MIN_BIN_SHARE * x.notna().sum() - 1).all()
    assert t["bin"].iloc[-1] == b.MISSING
    assert numeric["woe"].is_monotonic_decreasing
    assert fb.iv > 0.3


def test_numeric_assignment_uses_upper_inclusive_edges():
    x, y = synthetic()
    fb = b.fit("x", x, y)
    edge = fb.edges[0]
    labels = fb.assign(pd.Series([edge, edge + 1e-9, np.nan, 1e9]))
    assert labels[0] == fb.labels[0] and labels[1] == fb.labels[1]
    assert labels[2] == b.MISSING and labels[3] == fb.labels[len(fb.edges) - 1]


def test_categorical_pools_rare_and_unseen_levels():
    x = pd.Series(["RENT"] * 600 + ["OWN"] * 395 + ["NONE"] * 5)
    y = pd.Series([1] * 120 + [0] * 480 + [1] * 20 + [0] * 375 + [1] * 5)
    fb = b.fit("home", x, y)
    assert set(fb.labels) == {"RENT", "OWN", b.OTHER}
    assert list(fb.assign(pd.Series(["NONE", "MARS", "OWN"]))) == [b.OTHER, b.OTHER, "OWN"]
    woe = dict(zip(fb.labels, fb.woe, strict=True))
    assert woe["OWN"] > woe["RENT"] > woe[b.OTHER]


def test_transform_gives_zero_for_bins_unseen_in_training():
    x = pd.Series(np.arange(1000, dtype=float))
    y = pd.Series((np.arange(1000) > 700).astype(int))
    fb = b.fit("x", x, y)  # no missing values in training
    assert fb.transform(pd.Series([np.nan]))[0] == 0.0


def test_gini_and_ks():
    y = np.array([0, 0, 1, 1])
    assert m.gini(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert m.gini(y, [0.9, 0.8, 0.2, 0.1]) == -1.0
    assert m.ks(y, [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_score_scaling():
    # odds 50:1 good:bad -> 600 points; +20 points doubles the odds
    log_odds_bad = np.log(1 / 50)
    assert m.score_from_log_odds_bad(log_odds_bad) == pytest.approx(600)
    assert m.score_from_log_odds_bad(np.log(1 / 100)) == pytest.approx(620)
    assert m.pd_from_score(600) == pytest.approx(1 / 51)
