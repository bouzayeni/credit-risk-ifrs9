"""Generate the Power BI project (powerbi/CreditRiskIFRS9.pbip) from the warehouse.

The semantic model (model.bim, TMSL) imports the bi views over PostgreSQL. Its columns and
types are read from information_schema as the read-only role, so the model always matches
the warehouse. The report (report.json) is built from a compact page spec; every field a
visual uses is checked against the model before anything is written.

Usage (from the repo root, after credit_risk.warehouse.load_results):
    uv run python -m credit_risk.bi.pbip
"""

import json
import logging
import sys
import uuid
from dataclasses import dataclass, field

import psycopg

from credit_risk.config import REPO_ROOT, get_db_settings

log = logging.getLogger("pbip")

NAME = "CreditRiskIFRS9"
OUT = REPO_ROOT / "powerbi"
GINI_DEV = 0.352  # development Gini (Phase 3), for the amber line

# Model table name -> bi view
TABLES = {
    "ECL Loan": "ecl_loan",
    "Rating": "rating_scale",
    "PD Curve": "pd_term_structure",
    "Scenario": "ecl_scenario",
    "Monitoring": "monitoring_quarter",
    "Feature Stability": "monitoring_feature",
    "Vintage": "vintage_default_12m",
}
PG_TYPES = {
    "bigint": "int64", "integer": "int64", "smallint": "int64", "numeric": "double",
    "double precision": "double", "real": "double", "text": "string", "character": "string",
    "character varying": "string", "boolean": "boolean", "date": "dateTime",
}  # fmt: skip
HIDDEN = {
    "loan_id",
    "rating_orig_rank",
    "rating_now_rank",
    "rating_rank",
    "sort_order",
    "quarter_start",
}
SORT_BY = {
    ("ECL Loan", "rating_orig"): "rating_orig_rank",
    ("ECL Loan", "rating_now"): "rating_now_rank",
    ("Rating", "rating"): "rating_rank",
    ("Scenario", "scenario"): "sort_order",
}
PCT_COLUMNS = {"pd_orig", "pd_now", "pd_12m", "pd_12m_upside", "pd_12m_downside", "lgd", "pd_lower", "pd_upper",
               "int_rate", "pd_12m_fwd_base", "pd_lifetime_fwd_base", "cumulative_pd", "marginal_pd",
               "default_rate_12m", "weight", "observed_12m", "scorecard_pd", "production_pd",
               "early_observed", "early_expected"}  # fmt: skip
MONEY = "\\$#,0"
PCT1, PCT2, DEC3 = "0.0%", "0.00%", "0.000"

MEASURES = {
    "ECL Loan": [
        ("Loans", "COUNTROWS('ECL Loan')", "#,0"),
        ("Gross Carrying Amount", "SUM('ECL Loan'[gross_carrying_amt])", MONEY),
        ("ECL", "SUM('ECL Loan'[ecl])", MONEY),
        ("Coverage", "DIVIDE([ECL], [Gross Carrying Amount])", PCT2),
        ("ECL Base", "SUM('ECL Loan'[ecl_base])", MONEY),
        ("ECL Upside", "SUM('ECL Loan'[ecl_upside])", MONEY),
        ("ECL Downside", "SUM('ECL Loan'[ecl_downside])", MONEY),
        ("Scenario Uplift vs Base", "DIVIDE([ECL], [ECL Base]) - 1", PCT1),
        ("Stage 2 Share", "DIVIDE(CALCULATE([Gross Carrying Amount], 'ECL Loan'[stage] = 2), [Gross Carrying Amount])", PCT1),
        ("Stage 3 Share", "DIVIDE(CALCULATE([Gross Carrying Amount], 'ECL Loan'[stage] = 3), [Gross Carrying Amount])", PCT1),
        ("Downgraded Share", "DIVIDE(CALCULATE([Loans], 'ECL Loan'[notches_upgraded] < 0), [Loans])", PCT1),
        ("Forward 12m PD", "AVERAGE('ECL Loan'[pd_12m_fwd_base])", PCT2),
    ],
    "PD Curve": [
        ("Cumulative PD 36m", "CALCULATE(AVERAGE('PD Curve'[cumulative_pd]), 'PD Curve'[term_months] = 36)", PCT1),
        ("Cumulative PD 60m", "CALCULATE(AVERAGE('PD Curve'[cumulative_pd]), 'PD Curve'[term_months] = 60)", PCT1),
    ],
    "Scenario": [("Scenario ECL", "SUM(Scenario[ecl])", MONEY)],
    "Vintage": [("Default Rate 12m", "DIVIDE(SUM(Vintage[defaults_12m]), SUM(Vintage[loans]))", PCT2)],
    "Monitoring": [
        ("Score PSI", "AVERAGE(Monitoring[psi_score])", DEC3),
        ("PSI Amber Line", "0.10", DEC3),
        ("PSI Red Line", "0.25", DEC3),
        ("Gini", "AVERAGE(Monitoring[gini])", DEC3),
        ("Gini 6m", "AVERAGE(Monitoring[gini_6m])", DEC3),
        ("Gini Amber Line", f"{0.9 * GINI_DEV:.4f}", DEC3),
        ("Observed / Scorecard PD", "AVERAGE(Monitoring[scorecard_ratio])", "0.00"),
        ("Observed / Production PD", "AVERAGE(Monitoring[production_ratio])", "0.00"),
        ("Early Defaults Observed / Expected", "AVERAGE(Monitoring[early_ratio])", "0.00"),
    ],
    "Feature Stability": [("CSI", "AVERAGE('Feature Stability'[csi])", DEC3)],
}  # fmt: skip
RELATIONSHIPS = [
    ("ECL Loan", "rating_now", "Rating", "rating"),
    ("PD Curve", "rating", "Rating", "rating"),
    ("Feature Stability", "issue_quarter", "Monitoring", "issue_quarter"),
]


# ---------------------------------------------------------------- semantic model


def catalog(conn) -> dict[str, list[tuple[str, str]]]:
    """Columns (name, Postgres type) of each bi view, in order."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT table_name, column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = 'bi' ORDER BY table_name, ordinal_position"
        )
        out: dict[str, list[tuple[str, str]]] = {}
        for t, c, dt in cur.fetchall():
            out.setdefault(t, []).append((c, dt))
    return out


def m_query(view: str) -> list[str]:
    return [
        "let",
        "    Source = PostgreSQL.Database(Server, Database),",
        f'    Data = Source{{[Schema = "bi", Item = "{view}"]}}[Data]',
        "in",
        "    Data",
    ]


def build_model(cols: dict[str, list[tuple[str, str]]]) -> dict:
    tables = []
    for name, view in TABLES.items():
        if view not in cols:
            raise KeyError(f"bi.{view} not found; run credit_risk.warehouse.load_results first")
        columns = []
        for c, pg in cols[view]:
            dtype = PG_TYPES[pg]
            col = {"name": c, "dataType": dtype, "sourceColumn": c,
                   "summarizeBy": "none", "lineageTag": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{name}.{c}"))}  # fmt: skip
            if c in HIDDEN:
                col["isHidden"] = True
            if (name, c) in SORT_BY:
                col["sortByColumn"] = SORT_BY[(name, c)]
            if c in PCT_COLUMNS:
                col["formatString"] = PCT2
            elif dtype == "double" and any(k in c for k in ("ecl", "amt", "outstanding", "funded")):
                col["formatString"] = MONEY
            columns.append(col)
        table = {
            "name": name, "lineageTag": str(uuid.uuid5(uuid.NAMESPACE_URL, name)), "columns": columns,
            "partitions": [{"name": name, "mode": "import", "source": {"type": "m", "expression": m_query(view)}}],
        }  # fmt: skip
        if name in MEASURES:
            table["measures"] = [{"name": m, "expression": e, "formatString": f,
                                  "lineageTag": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{name}[{m}]"))}
                                 for m, e, f in MEASURES[name]]  # fmt: skip
        tables.append(table)
    rels = [{"name": f"{ft}_{fc}_{tt}", "fromTable": ft, "fromColumn": fc, "toTable": tt, "toColumn": tc}
            for ft, fc, tt, tc in RELATIONSHIPS]  # fmt: skip
    param = 'meta [IsParameterQuery = true, Type = "Text", IsParameterQueryRequired = true]'
    return {
        "compatibilityLevel": 1567,
        "model": {
            "culture": "en-US",
            "defaultPowerBIDataSourceVersion": "powerBI_V3",
            "sourceQueryCulture": "en-US",
            "dataAccessOptions": {"legacyRedirects": True, "returnErrorValuesAsNull": True},
            "tables": tables,
            "relationships": rels,
            "expressions": [
                {"name": "Server", "kind": "m", "expression": f'"localhost:5432" {param}'},
                {"name": "Database", "kind": "m", "expression": f'"credit_dw" {param}'},
            ],
            "annotations": [{"name": "__PBI_TimeIntelligenceEnabled", "value": "0"}],
        },
    }


def model_fields(model: dict) -> dict[tuple[str, str], str]:
    """(table, field) -> 'column' or 'measure'."""
    out = {}
    for t in model["model"]["tables"]:
        out.update({(t["name"], c["name"]): "column" for c in t["columns"]})
        out.update({(t["name"], m["name"]): "measure" for m in t.get("measures", [])})
    return out


# ---------------------------------------------------------------- report


@dataclass
class Visual:
    kind: str  # Power BI visualType
    x: int
    y: int
    w: int
    h: int
    roles: dict[str, list[tuple[str, str]]]  # role -> [(table, field)]
    title: str = ""
    extra: dict = field(default_factory=dict)


def literal(v) -> dict:
    return {"expr": {"Literal": {"Value": v}}}


def visual_config(v: Visual, idx: int, fields: dict) -> dict:
    alias = {}
    froms, selects, projections = [], [], {}
    for role, refs in v.roles.items():
        projections[role] = []
        for table, name in refs:
            kind = fields.get((table, name))
            if kind is None:
                raise KeyError(f"visual '{v.title}': {table}[{name}] is not in the model")
            if table not in alias:
                alias[table] = f"t{len(alias)}"
                froms.append({"Name": alias[table], "Entity": table, "Type": 0})
            ref = f"{table}.{name}"
            expr = {"Expression": {"SourceRef": {"Source": alias[table]}}, "Property": name}
            if not any(s["Name"] == ref for s in selects):
                selects.append({("Measure" if kind == "measure" else "Column"): expr, "Name": ref})
            projections[role].append({"queryRef": ref})
    single = {"visualType": v.kind, "projections": projections,
              "prototypeQuery": {"Version": 2, "From": froms, "Select": selects}, "drillFilterOtherVisuals": True}  # fmt: skip
    single.update(v.extra)
    if v.title:
        single["vcObjects"] = {
            "title": [{"properties": {"show": literal("true"), "text": literal(f"'{v.title}'")}}]
        }
    return {"name": uuid.uuid5(uuid.NAMESPACE_URL, f"{v.title}{idx}").hex[:20],
            "layouts": [{"id": 0, "position": {"x": v.x, "y": v.y, "z": idx, "width": v.w, "height": v.h}}],
            "singleVisual": single}  # fmt: skip


E, R, S, P, M, F, V = (
    "ECL Loan",
    "Rating",
    "Scenario",
    "PD Curve",
    "Monitoring",
    "Feature Stability",
    "Vintage",
)


def cards(*measures, y=10):
    w = (1260 - 10 * (len(measures) - 1)) // len(measures)
    return [
        Visual("card", 10 + i * (w + 10), y, w, 110, {"Values": [(t, m)]}, m)
        for i, (t, m) in enumerate(measures)
    ]


PAGES = {
    "IFRS 9 ECL": [
        *cards((E, "ECL"), (E, "Gross Carrying Amount"), (E, "Coverage"), (E, "Stage 2 Share"), (E, "Stage 3 Share")),
        Visual("tableEx", 10, 130, 620, 200, {"Values": [(E, "stage_label"), (E, "Loans"), (E, "Gross Carrying Amount"), (E, "ECL"), (E, "Coverage")]}, "ECL by stage"),
        Visual("slicer", 640, 130, 300, 200, {"Values": [(E, "region")]}, "Region"),
        Visual("slicer", 950, 130, 320, 200, {"Values": [(E, "term_months")]}, "Term (months)"),
        Visual("columnChart", 10, 340, 820, 370, {"Category": [(R, "rating")], "Series": [(E, "stage_label")], "Y": [(E, "ECL")]}, "ECL by current rating and stage"),
        Visual("clusteredBarChart", 840, 340, 430, 370, {"Category": [(R, "rating")], "Y": [(E, "Coverage")]}, "Coverage by current rating"),
    ],
    "Staging": [
        *cards((E, "Loans"), (E, "Downgraded Share"), (E, "Forward 12m PD")),
        Visual("tableEx", 10, 130, 620, 230, {"Values": [(E, "stage_reason"), (E, "Loans"), (E, "Gross Carrying Amount"), (E, "ECL"), (E, "Coverage")]}, "Why each loan is in its stage"),
        Visual("clusteredColumnChart", 640, 130, 630, 230, {"Category": [(E, "loan_status")], "Y": [(E, "Coverage")]}, "Coverage by loan status"),
        Visual("pivotTable", 10, 370, 1260, 340, {"Rows": [(E, "rating_orig")], "Columns": [(E, "rating_now")], "Values": [(E, "Loans")]}, "Rating migration: origination (rows) to today (columns)"),
    ],
    "Scenarios": [
        *cards((E, "ECL Upside"), (E, "ECL Base"), (E, "ECL Downside"), (E, "ECL"), (E, "Scenario Uplift vs Base")),
        Visual("clusteredColumnChart", 10, 130, 620, 580, {"Category": [(S, "scenario")], "Y": [(S, "Scenario ECL")]}, "ECL by scenario (probability-weighted last)"),
        Visual("tableEx", 640, 130, 630, 580, {"Values": [(R, "rating"), (R, "pd_12m_upside"), (R, "pd_12m"), (R, "pd_12m_downside"), (R, "lgd")]}, "12-month PD by scenario and LGD"),
    ],
    "Lifetime PD": [
        Visual("lineChart", 10, 10, 625, 700, {"Category": [(P, "mob")], "Series": [(R, "rating")], "Y": [(P, "Cumulative PD 36m")]}, "Cumulative PD, 36-month loans"),
        Visual("lineChart", 645, 10, 625, 700, {"Category": [(P, "mob")], "Series": [(R, "rating")], "Y": [(P, "Cumulative PD 60m")]}, "Cumulative PD, 60-month loans"),
    ],
    "Vintages": [
        Visual("lineChart", 10, 10, 1260, 700, {"Category": [(V, "vintage_year")], "Series": [(V, "grade")], "Y": [(V, "Default Rate 12m")]}, "12-month default rate by issue year and Lending Club grade"),
    ],
    "Monitoring": [
        Visual("lineChart", 10, 10, 625, 345, {"Category": [(M, "issue_quarter")], "Y": [(M, "Score PSI"), (M, "PSI Amber Line"), (M, "PSI Red Line")]}, "Score stability (PSI vs 2013-2014)"),
        Visual("lineChart", 645, 10, 625, 345, {"Category": [(M, "issue_quarter")], "Y": [(M, "Gini"), (M, "Gini 6m"), (M, "Gini Amber Line")]}, "Discrimination (Gini)"),
        Visual("lineChart", 10, 365, 625, 345, {"Category": [(M, "issue_quarter")], "Y": [(M, "Observed / Scorecard PD"), (M, "Observed / Production PD"), (M, "Early Defaults Observed / Expected")]}, "Calibration: observed / predicted"),
        Visual("pivotTable", 645, 365, 625, 345, {"Rows": [(F, "feature")], "Columns": [(F, "issue_quarter")], "Values": [(F, "CSI")]}, "Feature stability (CSI)"),
    ],
}  # fmt: skip


def build_report(fields: dict) -> dict:
    sections = []
    for i, (page, visuals) in enumerate(PAGES.items()):
        containers = []
        for j, v in enumerate(visuals):
            cfg = visual_config(v, j, fields)
            containers.append({"x": v.x, "y": v.y, "z": j, "width": v.w, "height": v.h,
                               "config": json.dumps(cfg), "filters": "[]"})  # fmt: skip
        sections.append({"name": uuid.uuid5(uuid.NAMESPACE_URL, page).hex[:20], "displayName": page,
                         "displayOption": 1, "width": 1280, "height": 720, "ordinal": i,
                         "config": "{}", "filters": "[]", "visualContainers": containers})  # fmt: skip
    config = {"version": "5.59", "themeCollection": {}, "activeSectionIndex": 0,
              "defaultDrillFilterOtherVisuals": True, "settings": {"useStylableVisualContainerHeader": True}}  # fmt: skip
    return {"config": json.dumps(config), "layoutOptimization": 0, "sections": sections}


# ---------------------------------------------------------------- files


def platform(kind: str) -> dict:
    return {
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/gitIntegration/platformProperties/2.0.0/schema.json",
        "metadata": {"type": kind, "displayName": NAME},
        "config": {
            "version": "2.0",
            "logicalId": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{NAME}.{kind}")),
        },
    }


def write_project(model: dict, report: dict) -> None:
    sm, rp = OUT / f"{NAME}.SemanticModel", OUT / f"{NAME}.Report"
    files = {
        OUT / f"{NAME}.pbip": {"version": "1.0", "artifacts": [{"report": {"path": f"{NAME}.Report"}}],
                               "settings": {"enableAutoRecovery": True}},
        sm / "definition.pbism": {"version": "1.0", "settings": {}},
        sm / "model.bim": model,
        sm / ".platform": platform("SemanticModel"),
        rp / "definition.pbir": {"version": "1.0", "datasetReference": {"byPath": {"path": f"../{NAME}.SemanticModel"}, "byConnection": None}},
        rp / "report.json": report,
        rp / ".platform": platform("Report"),
    }  # fmt: skip
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    (OUT / ".gitignore").write_text(
        "**/.pbi/localSettings.json\n**/.pbi/cache.abf\n", encoding="utf-8"
    )


def write_doc(model: dict) -> None:
    """docs/phase8_powerbi.md, generated from the same specs as the project."""
    tables = "\n".join(
        f"| {t['name']} | `bi.{TABLES[t['name']]}` | {len(t['columns'])} | {len(t.get('measures', []))} |"
        for t in model["model"]["tables"]
    )
    rels = "\n".join(
        f"- `{ft}[{fc}]` → `{tt}[{tc}]` (many to one)" for ft, fc, tt, tc in RELATIONSHIPS
    )
    measures = "\n".join(
        f"| {t} | {m} | `{e.replace('|', '¦')}` |" for t, ms in MEASURES.items() for m, e, _f in ms
    )
    pages = "\n".join(
        f"| {page} | "
        + "; ".join(v.title for v in visuals if v.kind != "card")
        + (
            f" (cards: {', '.join(v.title for v in visuals if v.kind == 'card')})"
            if any(v.kind == "card" for v in visuals)
            else ""
        )
        + " |"
        for page, visuals in PAGES.items()
    )
    doc = f"""# Phase 8: Power BI reporting

Generated by `python -m credit_risk.bi.pbip` from the same specification as the Power BI
project, so this page and `powerbi/{NAME}.pbip` cannot disagree.

## Data flow

```
Phase 1-7 outputs (Parquet, JSON)
  -> credit_risk.warehouse.load_results  (one transaction, reconciled to the Phase 6 ECL)
  -> PostgreSQL dw.* tables  -> bi.* views
  -> Power BI semantic model (import, read-only role credit_ro)
  -> report pages
```

`load_results` checks the loaded ECL, gross carrying amount and ECL by stage against the
Phase 6 summary to the cent before committing, and that `bi.ecl_loan` keeps every loan
through its joins. The report reads views only, as a role that cannot write.

## Semantic model

| Table | Source | Columns | Measures |
|---|---|---|---|
{tables}

Relationships:

{rels}

Columns and types are read from `information_schema` when the project is generated, and
every field a visual uses is checked against the model (`tests/test_pbip.py` also checks
that each measure's DAX refers to existing columns and measures).

| Table | Measure | DAX |
|---|---|---|
{measures}

## Report pages

| Page | Visuals |
|---|---|
{pages}

## Opening the report

1. Power BI Desktop (December 2024 or later): *File → Options and settings → Options →
   Preview features*, tick **Power BI Project (.pbip) save option**, and restart Desktop.
2. Open `powerbi/{NAME}.pbip`.
3. When asked for credentials for `localhost:5432;credit_dw`, choose **Database** and enter
   the read-only user `credit_ro` and its password (`CREDIT_RO_PASSWORD` in `.env`).
4. **Refresh**. The project contains no data, only the model and the report, so it stays small
   in git; Desktop keeps its local cache in `.pbi/` (git-ignored).

The `Server` and `Database` parameters (*Transform data → Edit parameters*) point the report
at another PostgreSQL server.
"""
    (REPO_ROOT / "docs" / "phase8_powerbi.md").write_text(doc, encoding="utf-8")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    with psycopg.connect(**get_db_settings().conninfo("ro")) as conn:
        cols = catalog(conn)
    model = build_model(cols)
    report = build_report(model_fields(model))
    write_project(model, report)
    write_doc(model)
    n_vis = sum(len(v) for v in PAGES.values())
    n_meas = sum(len(v) for v in MEASURES.values())
    log.info(
        "wrote %s: %s tables, %s measures, %s pages, %s visuals",
        OUT / f"{NAME}.pbip",
        len(TABLES),
        n_meas,
        len(PAGES),
        n_vis,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
