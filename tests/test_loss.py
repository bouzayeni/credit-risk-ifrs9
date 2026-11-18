import numpy as np
import pandas as pd
import pytest

from credit_risk.loss import ead, lgd


def annuity(principal, annual_pct, n):
    r = annual_pct / 1200
    return principal * r / (1 - (1 + r) ** -n)


def test_balance_amortises_to_zero_at_maturity():
    inst = annuity(10_000, 12.0, 36)
    assert ead.balance_after(10_000, 12.0, inst, 0) == pytest.approx(10_000)
    assert ead.balance_after(10_000, 12.0, inst, 36) == pytest.approx(0, abs=1e-6)
    assert ead.balance_after(10_000, 12.0, inst, 48) == 0  # floored, never negative
    # After one payment: interest 100, principal part = instalment - 100
    assert ead.balance_after(10_000, 12.0, inst, 1) == pytest.approx(10_000 - (inst - 100))


def test_schedule_shape_and_monotonic():
    inst = annuity(np.array([10_000, 5_000]), 10.0, 60)
    s = ead.schedule([10_000, 5_000], [10.0, 10.0], inst, 61)
    assert s.shape == (2, 61)
    assert (np.diff(s, axis=1) <= 1e-9).all()
    assert s[:, -1] == pytest.approx([0, 0], abs=1e-6)


def charged_off(months_since, recovery, ead_amt=1000.0, n=1):
    """Rows whose default month is `months_since` months before the March 2019 snapshot."""
    default_period = pd.Period("2019-03", "M") - months_since
    issue = (default_period - 5).to_timestamp()
    return pd.DataFrame(
        {
            "loan_status": ["Charged Off"] * n,
            "issue_d": [issue] * n,
            "months_to_default": [5] * n,
            "funded_amnt": [ead_amt + 500] * n,
            "total_rec_prncp": [500.0] * n,
            "recoveries": [recovery] * n,
            "collection_recovery_fee": [0.0] * n,
        }
    )


def test_realised_recovery_rate_and_workout_flag():
    df = pd.concat([charged_off(24, 150.0), charged_off(6, 0.0), charged_off(30, 5_000.0)])
    co = lgd.realised(df)
    assert co["recovery_rate"].tolist() == [0.15, 0.0, 1.0]  # capped at the exposure
    assert co["months_since_default"].tolist() == [24, 6, 30]
    assert co["workout_complete"].tolist() == [True, False, True]


def test_recovery_timing_follows_the_cross_section():
    # Recoveries only appear from month 10 on (10% of exposure), nothing before.
    rows = [charged_off(m, 100.0 if m >= 10 else 0.0, n=20) for m in range(1, 37)]
    co = lgd.realised(pd.concat(rows))
    timing = lgd.recovery_timing(co)
    assert timing.sum() == pytest.approx(1.0)
    assert timing.loc[:9].sum() == pytest.approx(0.0, abs=1e-9)
    assert timing.loc[10] == pytest.approx(1.0)


def test_discount_factor():
    timing = pd.Series([1.0], index=[12])
    df = lgd.discount_factor([12.0, 0.0], timing)
    assert df[0] == pytest.approx(1.01**-12)
    assert df[1] == pytest.approx(1.0)
