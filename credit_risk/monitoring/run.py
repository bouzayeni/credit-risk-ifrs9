"""Phase 7: model monitoring by issue quarter, 2015Q1-2018Q4.

Usage (from the repo root, after credit_risk.pd.run):
    uv run python -m credit_risk.monitoring.run
"""

import json
import logging
import sys

import numpy as np
import pandas as pd

from credit_risk.config import REPO_ROOT, get_settings
from credit_risk.data.build_loans import OUTPUT_NAME
from credit_risk.monitoring import metrics as mm
from credit_risk.monitoring.report import render
from credit_risk.scorecard.metrics import pd_from_score

log = logging.getLogger("monitoring")

MODELS = REPO_ROOT / "models"
REPORT_MD = REPO_ROOT / "docs" / "phase7_monitoring.md"
REPORT_JSON = REPO_ROOT / "docs" / "phase7_summary.json"
DEV_YEARS = (2007, 2014)  # scorecard development (training) years
# Stability reference: development years in which every scorecard field is reported
# (four bureau fields start in 2012), so the MISSING bin does not show up as drift.
REF_YEARS = (2013, 2014)
FIRST_Q, LAST_Q = pd.Period("2015Q1"), pd.Period("2018Q4")
EARLY_MOB = 6  # early-warning horizon: default within the first 6 months on book
LAG = 3  # months before a missed payment is recognised as default (Phase 1)
SNAPSHOT = pd.Period("2019-03", "M")


def load(spec: dict) -> pd.DataFrame:
    processed = get_settings().processed_path
    feats = [f["name"] for f in spec["features"] if f["name"] != "loan_to_income"]
    cols = ["id", "issue_d", "term", "eligible_12m", "default_12m", "is_default", "months_to_default",
            "loan_amnt", "annual_inc", *feats]  # fmt: skip
    df = pd.read_parquet(processed / OUTPUT_NAME, columns=list(dict.fromkeys(cols)))
    df["loan_to_income"] = (df["loan_amnt"] / df["annual_inc"]).astype("float32")
    for f in spec["features"]:
        if f["kind"] == "numeric":
            df[f["name"]] = pd.to_numeric(df[f["name"]], errors="coerce").astype("float32")
    df = df.merge(pd.read_parquet(processed / "scores.parquet", columns=["id", "score"]), on="id")
    df = df.merge(
        pd.read_parquet(processed / "ratings.parquet", columns=["id", "rating", "pd_12m_grade"]),
        on="id",
    )
    df["quarter"] = df["issue_d"].dt.to_period("Q")
    return df


def early_default(df: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    """Default within EARLY_MOB months (observed) and its expected rate from the PD curves."""
    issue_m = df["issue_d"].dt.to_period("M")
    observable = (SNAPSHOT.ordinal - issue_m.map(lambda p: p.ordinal)) >= EARLY_MOB + LAG
    observed = df["is_default"] & (df["months_to_default"] <= EARLY_MOB)
    cum = curves[curves["mob"] == EARLY_MOB].set_index(["rating", "term"])["cumulative_pd"]
    expected = (
        pd.Series(list(zip(df["rating"], df["term"].astype(int), strict=True))).map(cum).to_numpy()
    )
    return pd.DataFrame({"observable": observable.to_numpy(), "observed": observed.fillna(False).to_numpy(),
                         "expected": expected}, index=df.index)  # fmt: skip


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    spec = json.loads((MODELS / "scorecard.json").read_text())
    dev_gini = json.loads((REPO_ROOT / "docs" / "phase3_summary.json").read_text(encoding="utf-8"))[
        "splits"
    ]["train"]["gini"]
    curves = pd.read_parquet(get_settings().processed_path / "pd_term_structure.parquet")
    df = load(spec)
    log.info("%s loans", len(df))

    rescored, labels = mm.score(df, spec)
    mismatch = int((np.abs(rescored - df["score"].to_numpy()) > 1e-6).sum())
    log.info("re-scoring from scorecard.json: %s mismatches", mismatch)

    df["pd_dev"] = pd_from_score(df["score"])  # the scorecard's own points-scale PD (Phase 3)
    early = early_default(df, curves)
    dev = df["issue_d"].dt.year.between(*REF_YEARS)
    score_edges = np.unique(np.quantile(df.loc[dev, "score"], np.linspace(0, 1, 11)))
    score_edges[0], score_edges[-1] = -np.inf, np.inf
    score_band = pd.cut(df["score"], score_edges)
    ref_score = score_band[dev].value_counts()
    ref_feat = {f: labels.loc[dev, f].value_counts() for f in labels.columns}

    rows, csi_rows = [], []
    for q in pd.period_range(FIRST_Q, LAST_Q, freq="Q"):
        m = (df["quarter"] == q).to_numpy()
        sub = df[m]
        r = {"quarter": str(q), "loans": int(m.sum()), "mean_score": float(sub["score"].mean()),
             "psi_score": mm.psi(ref_score, score_band[m].value_counts())}  # fmt: skip
        r["psi_status"] = mm.psi_status(r["psi_score"])
        csi = {f: mm.psi(ref_feat[f], labels.loc[m, f].value_counts()) for f in labels.columns}
        for f, v in csi.items():
            csi_rows.append({"quarter": str(q), "feature": f, "csi": v, "status": mm.psi_status(v)})
        worst = max(csi, key=csi.get)
        r.update(
            {
                "max_csi": csi[worst],
                "max_csi_feature": worst,
                "features_amber_or_red": int(sum(v >= mm.PSI_AMBER for v in csi.values())),
            }
        )
        elig = sub["eligible_12m"].to_numpy()
        if elig.any():
            y = sub.loc[elig, "default_12m"]
            g, se = mm.gini_with_se(y, -sub.loc[elig, "score"])
            r.update({"gini": g, "gini_se": se, "gini_status": mm.gini_status(g, dev_gini)})
            cd = mm.calibration(y, sub.loc[elig, "pd_dev"])
            cp = mm.calibration(y, sub.loc[elig, "pd_12m_grade"])
            r.update(
                {f"cal_dev_{k}": v for k, v in cd.items()}
                | {f"cal_prod_{k}": v for k, v in cp.items()}
            )
        e = early[m]
        if e["observable"].all():
            g6, se6 = mm.gini_with_se(e["observed"], -sub["score"])
            ce = mm.calibration(e["observed"], e["expected"])
            r.update({"gini_6m": g6, "gini_6m_se": se6} | {f"early_{k}": v for k, v in ce.items()})
        rows.append(r)

    quarters = pd.DataFrame(rows)
    csi = pd.DataFrame(csi_rows)
    long = pd.concat([
        quarters.melt(id_vars="quarter", value_vars=[c for c in quarters.columns if quarters[c].dtype.kind == "f"],
                      var_name="metric").dropna(),
        csi.rename(columns={"csi": "value"}).assign(metric=lambda x: "csi_" + x["feature"])[["quarter", "metric", "value"]],
    ])  # fmt: skip
    long.to_parquet(get_settings().processed_path / "monitoring.parquet", index=False)

    ref_mix = score_band[dev].value_counts(normalize=True).sort_index()
    q_last = (df["quarter"] == LAST_Q).to_numpy()
    last_mix = score_band[q_last].value_counts(normalize=True).sort_index()
    summary = {
        "dev_years": DEV_YEARS, "ref_years": REF_YEARS, "ref_loans": int(dev.sum()),
        "missing_share_full_dev": float(labels.loc[df["issue_d"].dt.year.between(*DEV_YEARS), "mo_sin_old_rev_tl_op"].eq("MISSING").mean()), "dev_gini": dev_gini, "rescoring_mismatches": mismatch,
        "thresholds": {"psi": [mm.PSI_AMBER, mm.PSI_RED], "gini_share": [mm.GINI_AMBER, mm.GINI_RED],
                       "calibration_gap": [mm.CAL_AMBER, mm.CAL_RED], "alpha": mm.ALPHA},
        "quarters": quarters.to_dict("records"), "csi": csi.to_dict("records"),
        "score_mix": {"band": [str(b) for b in ref_mix.index], "dev": ref_mix.tolist(), "last": last_mix.reindex(ref_mix.index).tolist()},
        "early_mob": EARLY_MOB,
    }  # fmt: skip
    REPORT_JSON.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    REPORT_MD.write_text(render(summary), encoding="utf-8")
    log.info("wrote %s", REPORT_MD.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
