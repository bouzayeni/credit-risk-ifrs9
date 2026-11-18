"""Monitoring metrics and traffic-light thresholds.

- Population stability index (PSI) of the score and characteristic stability index (CSI) of
  each feature: sum over bins of (actual% - expected%) * ln(actual% / expected%), against the
  development population. Usual thresholds: below 0.10 stable, 0.10-0.25 some shift, above
  0.25 major shift.
- Discrimination: Gini by period with the Hanley-McNeil standard error, compared with the
  development Gini. Amber below 90% of it, red below 80%.
- Calibration: observed default rate vs mean predicted PD. The z-test treats loans as
  independent, which overstates significance when defaults are correlated, so a status also
  needs a material gap: amber if the ratio is outside 0.9-1.1 and significant, red outside
  0.8-1.2 and significant.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from credit_risk.scorecard.binning import FeatureBins

PSI_AMBER, PSI_RED = 0.10, 0.25
GINI_AMBER, GINI_RED = 0.90, 0.80  # share of the development Gini
CAL_AMBER, CAL_RED = 0.10, 0.20  # relative gap between observed and predicted
ALPHA = 0.05
EPS = 1e-4  # floor on bin shares, so empty bins do not give infinite PSI


def psi(expected: pd.Series, actual: pd.Series) -> float:
    """PSI between two distributions given as counts or shares over the same bins."""
    bins = expected.index.union(actual.index)
    e = expected.reindex(bins, fill_value=0).astype(float)
    a = actual.reindex(bins, fill_value=0).astype(float)
    e = (e / e.sum()).clip(lower=EPS)
    a = (a / a.sum()).clip(lower=EPS)
    return float(((a - e) * np.log(a / e)).sum())


def psi_status(v: float) -> str:
    return "red" if v >= PSI_RED else "amber" if v >= PSI_AMBER else "green"


def gini_with_se(y, risk) -> tuple[float, float]:
    """Gini = 2 AUC - 1 and its standard error (Hanley and McNeil, 1982)."""
    y = np.asarray(y, dtype=bool)
    auc = roc_auc_score(y, risk)
    n1, n0 = y.sum(), (~y).sum()
    q1, q2 = auc / (2 - auc), 2 * auc**2 / (1 + auc)
    var = (auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n0 - 1) * (q2 - auc**2)) / (n1 * n0)
    return 2 * auc - 1, 2 * float(np.sqrt(var))


def gini_status(g: float, reference: float) -> str:
    r = g / reference
    return "red" if r < GINI_RED else "amber" if r < GINI_AMBER else "green"


def calibration(defaults, pds) -> dict:
    d = np.asarray(defaults, dtype=float)
    p = np.asarray(pds, dtype=float)
    observed, predicted = d.mean(), p.mean()
    se = np.sqrt((p * (1 - p)).sum()) / len(p)
    z = (observed - predicted) / se
    pval = float(2 * norm.sf(abs(z)))
    gap = observed / predicted - 1
    significant = pval < ALPHA
    status = ("red" if significant and abs(gap) > CAL_RED
              else "amber" if significant and abs(gap) > CAL_AMBER else "green")  # fmt: skip
    return {"observed": float(observed), "predicted": float(predicted), "ratio": float(observed / predicted),
            "z": float(z), "p_value": pval, "status": status}  # fmt: skip


def bins_from_spec(feature: dict) -> FeatureBins:
    """Rebuild a feature's fitted bins from models/scorecard.json."""
    labels = [b["bin"] for b in feature["bins"]]
    return FeatureBins(
        name=feature["name"], kind=feature["kind"], labels=labels,
        woe=[b["woe"] for b in feature["bins"]], bads=[0] * len(labels), counts=[0] * len(labels),
        iv=feature["iv"], edges=list(feature["edges"]), levels=dict(feature["levels"]),
    )  # fmt: skip


def score(df: pd.DataFrame, spec: dict) -> tuple[np.ndarray, pd.DataFrame]:
    """Scorecard points of each loan and the bin label of each feature."""
    total = np.zeros(len(df))
    labels = {}
    for f in spec["features"]:
        lab = pd.Series(bins_from_spec(f).assign(df[f["name"]]), index=df.index)
        points = {b["bin"]: b["points"] for b in f["bins"]}
        neutral = int(np.round(np.mean(list(points.values()))))
        total += lab.map(points).fillna(neutral).to_numpy(dtype=float)
        labels[f["name"]] = lab
    return total, pd.DataFrame(labels)
