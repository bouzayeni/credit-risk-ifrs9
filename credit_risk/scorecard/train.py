"""Train, validate and test the application scorecard; write the report and scores.

Out-of-time design (loans with a complete 12-month window only):
    train       issued 2007-2014   bins, WoE, coefficients
    validation  issued 2015        feature selection
    test        issued 2016-2017   final evaluation only, never used for a decision

Steps: WoE binning of every candidate on train -> keep IV >= MIN_IV -> drop features whose
WoE correlates above MAX_CORR with a stronger one -> forward selection on validation Gini ->
logistic regression on train (drop wrong-sign or insignificant terms) -> integer points.
Benchmarks on test: Lending Club's sub-grade and interest rate, and XGBoost on the same raw
candidate features.

Usage (from the repo root):
    uv run python -m credit_risk.scorecard.train
"""

import json
import logging
import sys
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

from credit_risk.config import REPO_ROOT, get_settings
from credit_risk.data.build_loans import OUTPUT_NAME
from credit_risk.data.columns import APPLICATION_NUMERIC, FIELDS_FROM_2016
from credit_risk.scorecard import binning
from credit_risk.scorecard import metrics as m
from credit_risk.scorecard.report import render

log = logging.getLogger("scorecard")

TRAIN_YEARS, VALID_YEAR, TEST_YEARS = (2007, 2014), 2015, (2016, 2017)
TARGET = "default_12m"
MIN_IV = 0.02
MAX_CORR = 0.6
MAX_FEATURES = 15
MIN_GINI_GAIN = 0.002
SELECTION_SAMPLE = 200_000
MAX_P_VALUE = 0.01
SEED = 42

# Why some application fields are not candidates (see docs/phase3_scorecard.md).
EXCLUDED = {
    "grade, sub_grade, int_rate, installment": "Lending Club's own risk grade and price: benchmark only",
    "addr_state": "geography: fair-lending (redlining) risk, kept out of the score",
    "annual_inc_joint, dti_joint, is_joint": "joint applications start in late 2015: absent from training years",
    ", ".join(FIELDS_FROM_2016[:3]) + ", ... (14 fields)": "collected from 2016 only: absent from training years",
}  # fmt: skip

NUMERIC = [
    c for c in APPLICATION_NUMERIC
    if c not in FIELDS_FROM_2016
    and c not in ("fico_range_low", "fico_range_high", "annual_inc_joint", "dti_joint")
] + ["fico", "emp_length", "credit_history_months", "term", "loan_to_income"]  # fmt: skip
CATEGORICAL = ["home_ownership", "verification_status", "purpose"]
CANDIDATES = NUMERIC + CATEGORICAL

REPORT_MD = REPO_ROOT / "docs" / "phase3_scorecard.md"
REPORT_JSON = REPO_ROOT / "docs" / "phase3_summary.json"


def load(parquet) -> pd.DataFrame:
    base = [c for c in CANDIDATES if c != "loan_to_income"]
    cols = ["id", "issue_d", "eligible_12m", TARGET, "sub_grade", "int_rate", *base]
    df = pd.read_parquet(parquet, columns=cols)
    for c in NUMERIC:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    df["loan_to_income"] = (df["loan_amnt"] / df["annual_inc"]).astype("float32")
    year = df["issue_d"].dt.year
    df["split"] = np.select(
        [year.between(*TRAIN_YEARS), year == VALID_YEAR, year.between(*TEST_YEARS)],
        ["train", "valid", "test"], "none",
    )  # fmt: skip
    df.loc[~df["eligible_12m"], "split"] = "none"
    return df


def woe_matrix(df: pd.DataFrame, bins: dict[str, binning.FeatureBins], features) -> np.ndarray:
    return np.column_stack([bins[f].transform(df[f]) for f in features]).astype("float32")


def fit_lr(x: np.ndarray, y: np.ndarray) -> LogisticRegression:
    return LogisticRegression(C=1e6, max_iter=500).fit(x, y)


def correlation_filter(woe: np.ndarray, features: list[str], ivs: dict[str, float]):
    order = sorted(range(len(features)), key=lambda i: -ivs[features[i]])
    corr = np.corrcoef(woe, rowvar=False)
    kept: list[int] = []
    dropped = {}
    for i in order:
        clash = [j for j in kept if abs(corr[i, j]) > MAX_CORR]
        if clash:
            dropped[features[i]] = features[clash[0]]
        else:
            kept.append(i)
    return [features[i] for i in kept], dropped


def forward_select(xt, yt, xv, yv, features: list[str]) -> list[dict]:
    chosen: list[int] = []
    steps = []
    best = 0.0
    while len(chosen) < MAX_FEATURES:
        trials = {}
        for j in range(len(features)):
            if j in chosen:
                continue
            cols = [*chosen, j]
            model = fit_lr(xt[:, cols], yt)
            trials[j] = m.gini(yv, model.decision_function(xv[:, cols]))
        j, g = max(trials.items(), key=lambda kv: kv[1])
        if g - best < MIN_GINI_GAIN:
            break
        chosen.append(j)
        best = g
        steps.append({"step": len(chosen), "added": features[j], "valid_gini": round(g, 4)})
        log.info("step %d: +%s -> validation Gini %.4f", len(chosen), features[j], g)
    return steps


def final_logit(x: np.ndarray, y: np.ndarray, features: list[str], index: dict[str, int]):
    """statsmodels fit; drop the worst wrong-sign or insignificant term until all are clean.

    With WoE inputs (positive = safer) and a default target, every coefficient must be
    negative; a positive one means the feature acts against its own binning.
    """
    features = list(features)
    removed = []
    while True:
        cols = [index[f] for f in features]
        res = sm.Logit(y, sm.add_constant(x[:, cols].astype("float64"))).fit(disp=0)
        coef, pval = res.params[1:], res.pvalues[1:]
        bad = [(f, c, p) for f, c, p in zip(features, coef, pval, strict=True)
               if c >= 0 or p > MAX_P_VALUE]  # fmt: skip
        if not bad:
            return res, features, removed
        worst = max(bad, key=lambda t: (t[1] >= 0, t[2]))
        removed.append({"feature": worst[0], "coef": round(float(worst[1]), 4),
                        "p_value": float(worst[2])})  # fmt: skip
        features.remove(worst[0])


def points_table(bins, features, coefs, intercept) -> pd.DataFrame:
    offset, factor = m.scale_factors()
    base_share = (offset - factor * intercept) / len(features)
    rows = []
    for f, beta in zip(features, coefs, strict=True):
        t = bins[f].table()
        t["points"] = np.round(base_share - factor * beta * t["woe"]).astype(int)
        rows.append(t[["feature", "bin", "loans", "share", "bad_rate", "woe", "points"]])
    return pd.concat(rows, ignore_index=True)


def score(df: pd.DataFrame, bins, table: pd.DataFrame) -> np.ndarray:
    total = np.zeros(len(df))
    for f, t in table.groupby("feature", sort=False):
        lookup = dict(zip(t["bin"], t["points"], strict=True))
        labels = pd.Series(bins[f].assign(df[f]))
        # A bin unseen in training scores like WoE 0: the feature's neutral points.
        neutral = int(np.round(t["points"].mean()))
        total += labels.map(lookup).fillna(neutral).to_numpy(dtype=float)
    return total


def evaluate(y, risk) -> dict:
    return {"loans": int(len(y)), "default_rate": round(float(np.mean(y)), 4),
            "gini": round(m.gini(y, risk), 4), "ks": round(m.ks(y, risk), 4)}  # fmt: skip


def train_xgboost(df: pd.DataFrame, x: pd.DataFrame) -> xgb.XGBClassifier:
    """Gradient boosting on the raw candidate features (missing values handled natively)."""
    tr, va = df["split"] == "train", df["split"] == "valid"
    model = xgb.XGBClassifier(
        n_estimators=2000, learning_rate=0.05, max_depth=5, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=50, tree_method="hist",
        enable_categorical=True, eval_metric="auc", early_stopping_rounds=50,
        random_state=SEED, n_jobs=4,
    )  # fmt: skip
    model.fit(x[tr], df.loc[tr, TARGET], eval_set=[(x[va], df.loc[va, TARGET])], verbose=False)
    return model


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    started = time.perf_counter()
    df = load(get_settings().processed_path / OUTPUT_NAME)
    split = df["split"]
    tr, va, te = (split == "train"), (split == "valid"), (split == "test")
    y = df[TARGET].astype(int).to_numpy()
    log.info("loaded %s loans: train %s, valid %s, test %s", len(df), tr.sum(), va.sum(), te.sum())

    # 1. Bin every candidate on train.
    bins = {f: binning.fit(f, df.loc[tr, f], df.loc[tr, TARGET].astype(int)) for f in CANDIDATES}
    ivs = {f: b.iv for f, b in bins.items()}
    informative = [f for f in CANDIDATES if ivs[f] >= MIN_IV]

    # 2. Correlation filter on train WoE.
    index = {f: i for i, f in enumerate(informative)}
    xtr = woe_matrix(df[tr], bins, informative)
    xva = woe_matrix(df[va], bins, informative)
    kept, corr_dropped = correlation_filter(xtr, informative, ivs)

    # 3. Forward selection on validation Gini (fits on a train sample for speed).
    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(xtr), size=min(SELECTION_SAMPLE, len(xtr)), replace=False)
    kept_idx = [index[f] for f in kept]
    steps = forward_select(xtr[sample][:, kept_idx], y[tr][sample], xva[:, kept_idx], y[va], kept)
    selected = [s["added"] for s in steps]

    # 4. Final logistic regression on all of train.
    res, final, removed = final_logit(xtr, y[tr], selected, index)
    coefs, intercept = res.params[1:], res.params[0]
    table = points_table(bins, final, coefs, intercept)

    # 5. Score every loan and evaluate (risk = minus score: low score = risky).
    df["score"] = score(df, bins, table)
    df["pd_scorecard"] = m.pd_from_score(df["score"])
    risk = -df["score"].to_numpy()
    perf = {name: evaluate(y[mask], risk[mask]) for name, mask in
            (("train", tr), ("valid", va), ("test", te))}  # fmt: skip
    perf["test"]["gini_ci95"] = [round(v, 4) for v in m.bootstrap_gini(y[te], risk[te])]
    by_year = {
        str(yr): evaluate(y[te & (df["issue_d"].dt.year == yr)], risk[te & (df["issue_d"].dt.year == yr)])
        for yr in range(TEST_YEARS[0], TEST_YEARS[1] + 1)
    }  # fmt: skip

    # 6. Benchmarks on test.
    sub_grade_rank = df["sub_grade"].rank(method="dense").to_numpy()
    x_raw = df[CANDIDATES].copy()
    for c in CATEGORICAL:  # categories from the whole file, so every split shares them
        x_raw[c] = x_raw[c].astype("category")
    model = train_xgboost(df, x_raw)
    xgb_risk = model.predict_proba(x_raw[te])[:, 1]
    del x_raw
    benchmarks = {
        "scorecard": {**perf["test"], "features": len(final)},
        "lc_sub_grade": {**evaluate(y[te], sub_grade_rank[te]),
                         "gini_ci95": [round(v, 4) for v in m.bootstrap_gini(y[te], sub_grade_rank[te])]},
        "lc_int_rate": evaluate(y[te], df.loc[te, "int_rate"].to_numpy()),
        "xgboost": {**evaluate(y[te], xgb_risk),
                    "gini_ci95": [round(v, 4) for v in m.bootstrap_gini(y[te], xgb_risk)],
                    "trees": int(model.best_iteration) + 1, "features": len(CANDIDATES)},
    }  # fmt: skip

    # 7. Score bands (train deciles) and calibration preview on test.
    edges = np.unique(np.quantile(df.loc[tr, "score"], np.linspace(0, 1, 11)[1:-1]))
    band = np.searchsorted(edges, df["score"], side="right")
    bands = []
    for b in range(len(edges) + 1):
        lo = "min" if b == 0 else int(edges[b - 1])
        hi = "max" if b == len(edges) else int(edges[b])
        sel_te = te.to_numpy() & (band == b)
        sel_tr = tr.to_numpy() & (band == b)
        bands.append({"band": b + 1, "score_range": f"[{lo}, {hi})",
                      "test_loans": int(sel_te.sum()),
                      "test_default_rate": round(float(y[sel_te].mean()), 4),
                      "train_default_rate": round(float(y[sel_tr].mean()), 4),
                      "scorecard_pd": round(float(df.loc[sel_te, "pd_scorecard"].mean()), 4)})  # fmt: skip

    # 8. Save model, scores, report.
    out_dir = REPO_ROOT / "models"
    out_dir.mkdir(exist_ok=True)
    model_spec = {
        "target": TARGET, "train_years": TRAIN_YEARS, "valid_year": VALID_YEAR,
        "scale": {"base_score": m.BASE_SCORE, "base_odds": m.BASE_ODDS, "pdo": m.PDO},
        "intercept": float(intercept),
        "features": [{"name": f, "coef": float(c), "iv": ivs[f], "kind": bins[f].kind,
                      "edges": bins[f].edges, "levels": bins[f].levels,
                      "bins": table[table["feature"] == f][["bin", "woe", "points"]].to_dict("records")}
                     for f, c in zip(final, coefs, strict=True)],
    }  # fmt: skip
    (out_dir / "scorecard.json").write_text(json.dumps(model_spec, indent=2, default=float))
    scores_path = get_settings().processed_path / "scores.parquet"
    df[["id", "score", "pd_scorecard"]].to_parquet(scores_path, index=False)

    summary = {
        "splits": {k: perf[k] for k in ("train", "valid", "test")},
        "test_by_year": by_year,
        "iv": sorted(({"feature": f, "iv": round(v, 4), "bins": len(bins[f].labels)}
                      for f, v in ivs.items()), key=lambda r: -r["iv"]),  # fmt: skip
        "correlation_dropped": corr_dropped,
        "candidates_after_filters": kept,
        "selection_steps": steps,
        "removed_in_final_fit": removed,
        "final_features": [{"feature": f, "coef": round(float(c), 4), "p_value": float(p),
                            "iv": round(ivs[f], 4)}
                           for f, c, p in zip(final, coefs, res.pvalues[1:], strict=True)],
        "intercept": round(float(intercept), 4),
        "benchmarks_test": benchmarks,
        "score_bands": bands,
        "scorecard": table.to_dict("records"),
        "runtime_s": round(time.perf_counter() - started),
    }  # fmt: skip
    REPORT_JSON.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    REPORT_MD.write_text(render(summary), encoding="utf-8")
    log.info("done in %ss: test Gini %.4f (scorecard) vs %.4f (sub-grade) vs %.4f (xgboost)",
             summary["runtime_s"], perf["test"]["gini"], benchmarks["lc_sub_grade"]["gini"],
             benchmarks["xgboost"]["gini"])  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
