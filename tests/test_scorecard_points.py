import numpy as np
import pandas as pd

from credit_risk.scorecard import binning, train
from credit_risk.scorecard import metrics as m


def synthetic(n=30_000, seed=3):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 1, n)
    b = rng.uniform(0, 1, n)
    home = rng.choice(["RENT", "OWN", "MORTGAGE"], n)
    log_odds = -3 + 2 * a - 1.5 * b + np.where(home == "RENT", 0.4, 0.0)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-log_odds))).astype(int)
    return pd.DataFrame({"a": a, "b": b, "home": home, "y": y})


def test_points_reproduce_the_logistic_regression_score():
    df = synthetic()
    feats = ["a", "b", "home"]
    bins = {f: binning.fit(f, df[f], df["y"]) for f in feats}
    x = np.column_stack([bins[f].transform(df[f]) for f in feats])
    res, final, removed = train.final_logit(
        x, df["y"].to_numpy(), feats, {f: i for i, f in enumerate(feats)}
    )
    assert final == feats and removed == []
    assert (res.params[1:] < 0).all()  # WoE inputs: safer bins must lower the default odds

    table = train.points_table(bins, final, res.params[1:], res.params[0])
    points = train.score(df, bins, table)
    exact = m.score_from_log_odds_bad(res.params[0] + x @ res.params[1:])
    assert np.abs(points - exact).max() <= 0.5 * len(feats) + 1e-9  # only rounding differs
    assert m.gini(df["y"], -points) > 0.2


def test_final_logit_drops_a_wrong_sign_feature():
    # In-sample p-values are optimistic when WoE is fit on the same data (even noise can look
    # significant), so noise is screened out by forward selection on validation data instead.
    # What final_logit must guarantee: no term acts against its own binning.
    df = synthetic()
    feats = ["a", "b"]
    bins = {f: binning.fit(f, df[f], df["y"]) for f in feats}
    x = np.column_stack([bins["a"].transform(df["a"]), -bins["b"].transform(df["b"])])
    _, final, removed = train.final_logit(x, df["y"].to_numpy(), feats, {"a": 0, "b": 1})
    assert final == ["a"]
    assert removed[0]["feature"] == "b" and removed[0]["coef"] > 0
