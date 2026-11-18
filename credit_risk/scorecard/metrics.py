"""Discrimination metrics and score scaling.

Gini = 2 * AUC - 1 (0 = random, 1 = perfect ranking); the usual credit-risk measure.
KS = largest gap between the cumulative score distributions of bads and goods.
"""

import math

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

# Points scale: a score of BASE_SCORE means good:bad odds of BASE_ODDS, and every PDO points
# doubles the odds. 600 at 50:1 with 20 PDO is a common convention.
BASE_SCORE = 600
BASE_ODDS = 50
PDO = 20


def gini(y, risk) -> float:
    """risk: higher = more likely to default."""
    return 2 * roc_auc_score(y, risk) - 1


def ks(y, risk) -> float:
    fpr, tpr, _ = roc_curve(y, risk)
    return float(np.max(tpr - fpr))


def bootstrap_gini(y, risk, n: int = 200, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval of the Gini over n bootstrap resamples."""
    y, risk = np.asarray(y), np.asarray(risk)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        stats.append(gini(y[idx], risk[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def scale_factors(base_score: float = BASE_SCORE, base_odds: float = BASE_ODDS,
                  pdo: float = PDO) -> tuple[float, float]:  # fmt: skip
    """score = offset + factor * ln(good:bad odds)."""
    factor = pdo / math.log(2)
    return base_score - factor * math.log(base_odds), factor


def score_from_log_odds_bad(log_odds_bad, **kw) -> np.ndarray:
    """Logistic regression gives ln(p_bad / p_good); a score uses ln(p_good / p_bad)."""
    offset, factor = scale_factors(**kw)
    return offset - factor * np.asarray(log_odds_bad)


def pd_from_score(score, **kw) -> np.ndarray:
    offset, factor = scale_factors(**kw)
    log_odds_good = (np.asarray(score) - offset) / factor
    return 1 / (1 + np.exp(log_odds_good))
