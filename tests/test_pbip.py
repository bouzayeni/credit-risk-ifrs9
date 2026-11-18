import json
import re

import pytest

from credit_risk.bi import pbip

# The bi view columns the model needs (a stand-in for information_schema).
FAKE_CATALOG = {
    "ecl_loan": [(c, "numeric") for c in (
        "gross_carrying_amt", "ecl", "ecl_base", "ecl_upside", "ecl_downside", "pd_12m_fwd_base", "pd_orig")]
    + [("loan_id", "bigint"), ("stage", "smallint"), ("notches_upgraded", "integer"), ("term_months", "smallint"),
       ("rating_orig_rank", "smallint"), ("rating_now_rank", "smallint")]
    + [(c, "text") for c in ("stage_label", "stage_reason", "rating_orig", "rating_now", "region", "loan_status")],
    "rating_scale": [("rating", "text"), ("rating_rank", "smallint")]
    + [(c, "numeric") for c in ("pd_12m", "pd_12m_upside", "pd_12m_downside", "lgd")],
    "pd_term_structure": [("rating", "text"), ("term_months", "smallint"), ("mob", "smallint"),
                          ("cumulative_pd", "numeric")],
    "ecl_scenario": [("scenario", "text"), ("sort_order", "smallint"), ("ecl", "numeric")],
    "monitoring_quarter": [("issue_quarter", "text"), ("quarter_start", "date")]
    + [(c, "numeric") for c in ("psi_score", "gini", "gini_6m", "scorecard_ratio", "production_ratio", "early_ratio")],
    "monitoring_feature": [("issue_quarter", "text"), ("feature", "text"), ("csi", "numeric")],
    "vintage_default_12m": [("vintage_year", "smallint"), ("grade", "text"), ("loans", "bigint"),
                            ("defaults_12m", "bigint")],
}  # fmt: skip


@pytest.fixture(scope="module")
def model():
    return pbip.build_model(FAKE_CATALOG)


def test_measures_reference_existing_columns_and_measures(model):
    fields = pbip.model_fields(model)
    for t in model["model"]["tables"]:
        for m in t.get("measures", []):
            for table, col in re.findall(
                r"'?([A-Za-z][A-Za-z0-9 ]*?)'?\[([A-Za-z0-9_]+)\]", m["expression"]
            ):
                assert (table, col) in fields, (
                    f"{t['name']}[{m['name']}] uses unknown {table}[{col}]"
                )
            for ref in re.findall(r"(?<![\w'\]])\[([^\]]+)\]", m["expression"]):
                assert (t["name"], ref) in fields, f"{m['name']} uses unknown measure [{ref}]"


def test_sort_and_relationship_columns_exist(model):
    fields = pbip.model_fields(model)
    for t in model["model"]["tables"]:
        for c in t["columns"]:
            if "sortByColumn" in c:
                assert (t["name"], c["sortByColumn"]) in fields
    for r in model["model"]["relationships"]:
        assert (r["fromTable"], r["fromColumn"]) in fields and (
            r["toTable"],
            r["toColumn"],
        ) in fields


def test_report_builds_and_every_visual_query_is_consistent(model):
    report = pbip.build_report(pbip.model_fields(model))
    assert [s["displayName"] for s in report["sections"]] == list(pbip.PAGES)
    for s in report["sections"]:
        for vc in s["visualContainers"]:
            cfg = json.loads(vc["config"])
            sv = cfg["singleVisual"]
            names = {sel["Name"] for sel in sv["prototypeQuery"]["Select"]}
            refs = {p["queryRef"] for role in sv["projections"].values() for p in role}
            assert refs == names
            aliases = {f["Name"] for f in sv["prototypeQuery"]["From"]}
            for sel in sv["prototypeQuery"]["Select"]:
                expr = sel.get("Measure") or sel.get("Column")
                assert expr["Expression"]["SourceRef"]["Source"] in aliases


def test_unknown_field_is_rejected(model):
    bad = pbip.Visual("card", 0, 0, 10, 10, {"Values": [("ECL Loan", "No Such Measure")]}, "bad")
    with pytest.raises(KeyError, match="No Such Measure"):
        pbip.visual_config(bad, 0, pbip.model_fields(model))


def test_missing_view_is_reported():
    with pytest.raises(KeyError, match="load_results"):
        pbip.build_model({k: v for k, v in FAKE_CATALOG.items() if k != "ecl_scenario"})
