# Credit Risk Scorecard and IFRS 9 Expected Credit Loss

An end-to-end credit risk project on 2.26 million real consumer loans from
[Lending Club](https://www.kaggle.com/datasets/wordsforthewise/lending-club) (2007–2018,
CC0 public domain): an application scorecard, PD / LGD / EAD models, an IFRS 9 expected credit
loss (ECL) calculation with staging and scenarios, model monitoring, a PostgreSQL warehouse
and a Power BI report.

> **Historical data.** Loans issued 2007–2018, with statuses as of March 2019. The IFRS 9
> reporting date is 2019-03-31. Nothing here is live data or financial advice.

## Results

| | |
|---|---|
| **Loss allowance** | **$920.9M on a $9.51bn open book (907,904 loans): 9.68% coverage** |
| Stages | Stage 1: 94.1% of the book at 6.6% coverage. Stage 2: 3.3% at 33.4%. Stage 3: 2.6% at 91.2% |
| Scenarios | Upside $717M / base $880M / downside $1,329M; probability-weighted 4.7% above base |
| Scorecard | 12 features; out-of-time Gini 0.343 (2016–2017), against 0.390 for XGBoost and 0.399 for Lending Club's own sub-grade |
| PD calibration | 2017 back-test: 6.14% predicted vs 6.13% observed |
| LGD | 90.9% (unsecured; 21.6% of defaults recover nothing), from 174,565 completed workouts |
| Back-test | Full PD × LGD × EAD chain on matured vintages: lifetime loss over-predicted 1.15–1.33×, mostly from the first-year calibration level |
| Monitoring | Calibration drift detected from 2015Q3 (defaults up to 1.49× the original scorecard PD); population shift from 2017Q4 |

The phase reports for Phases 1 and 3–9 are written by the code that computes their numbers,
so text and results cannot drift apart (the Phase 0 findings and the star schema notes are
hand-written). Each phase was checked before the next, and the reports keep the errors found
on the way and how they were fixed.

## What is in it

| Question a bank asks | Answer here | Report |
|---|---|---|
| Should we approve this applicant? | Application scorecard: WoE/IV binning, logistic regression, points scale | [Phase 3](docs/phase3_scorecard.md) |
| What share will default in 12 months, and over their life? | Point-in-time PD calibration, R1–R10 master scale, competing-risk lifetime PD curves | [Phase 4](docs/phase4_pd.md) |
| How much do we lose when they do? | LGD from discounted recoveries; EAD from amortisation schedules | [Phase 5](docs/phase5_lgd_ead.md) |
| How much should we provision? | IFRS 9 staging (updated FICO, delinquency, hardship), monthly lifetime ECL engine, Vasicek scenarios, sensitivities | [Phase 6](docs/phase6_ecl.md) |
| Is the model still working? | PSI / CSI, Gini, calibration and early-default monitoring with traffic lights | [Phase 7](docs/phase7_monitoring.md) |
| How do we report it? | PostgreSQL star schema with read-only BI views; generated Power BI project (6 pages) | [Phase 2](docs/star_schema.md), [Phase 8](docs/phase8_powerbi.md) |
| What could be wrong? | Model cards and a model risk register | [Model cards](docs/model_cards.md), [risk register](docs/model_risk_register.md) |

Data preparation: [Phase 0 findings](docs/phase0_findings.md) (download, raw profile) and
[Phase 1 report](docs/phase1_report.md) (default definition, observation windows, leakage
review).

## Design choices worth a look

- **No leakage.** Every column has a role (application, pricing or outcome) in
  [`columns.py`](credit_risk/data/columns.py), and the scorecard can only use application fields.
  Development, validation and test sets are split by time, not at random.
- **Default timing from two sources.** The default month is the earlier of the instalment
  count and the last payment date. Fixing an off-by-one in the instalment count raised the
  agreement between the two from 16% to 62% (Phase 1).
- **Competing risks.** Prepayment is a competing risk in the lifetime PD curves and is not
  applied again to EAD, so it is not counted twice.
- **Honest validation.** The PD calibration design changed after the first 2017 back-test
  failed; the report shows both. Monitoring's stability reference was changed after it
  showed false drift caused by fields missing before 2012.
- **Reconciled loads.** Warehouse loads are single transactions that reconcile row counts and
  ECL to the cent before committing; Power BI reads views only, as a role that cannot write.
- **Generated, checked reporting.** The Power BI model is built from the warehouse's own
  column catalogue, and tests check every DAX and visual reference.

## Architecture

```
Kaggle CSV (hash-checked) -> clean Parquet (Phase 1)
  -> scorecard (3) -> PD calibration + lifetime curves (4) -> LGD / EAD (5) -> ECL (6)
  -> monitoring (7)
  -> PostgreSQL dw star schema + bi views (2, 8) -> Power BI project (8)
  -> model cards + risk register (9)
```

## Phases

| Phase | Scope | Status |
|---|---|---|
| 0 | Setup, download with SHA-256 manifest, raw data profile | Done |
| 1 | Default definition, observation windows, cleaning, leakage review | Done |
| 2 | PostgreSQL star schema, BI views, least-privilege roles, reconciled load | Done |
| 3 | Application scorecard | Done |
| 4 | PD calibration, master scale, lifetime PD | Done |
| 5 | LGD and EAD | Done |
| 6 | IFRS 9 ECL | Done |
| 7 | Monitoring | Done |
| 8 | Results warehouse and Power BI project (generated and tested; not yet opened in Power BI Desktop) | Done |
| 9 | Model cards, model risk register, documentation | Done |

## Setup

Requires [uv](https://docs.astral.sh/uv/), Python 3.11 (uv installs it) and PostgreSQL 16.
About 2 GB of disk for data. Developed on a machine with 8 GB of RAM.

```powershell
# Your user environment sets UV_PROJECT_ENVIRONMENT for another project: override it here.
. .\dev.ps1                      # sets UV_PROJECT_ENVIRONMENT for this repo (venv on K:)
uv sync --all-groups

Copy-Item .env.example .env      # then point CREDIT_RAW_DIR to a drive with ~2 GB free
uv run python -m credit_risk.data.download       # ~650 MB, idempotent, hash-checked
uv run python -m credit_risk.data.profile_raw    # writes docs/phase0_profile.json
uv run python -m credit_risk.data.build_loans    # clean Parquet + docs/phase1_report.md
uv run python -m credit_risk.warehouse.setup_db  # PostgreSQL database, roles (see .env.example)
uv run python -m credit_risk.warehouse.load      # star schema + BI views, reconciled
uv run python -m credit_risk.scorecard.train     # scorecard + docs/phase3_scorecard.md (~6 min)
uv run python -m credit_risk.pd.run              # PD calibration, ratings, lifetime PD curves
uv run python -m credit_risk.loss.run            # LGD and EAD + docs/phase5_lgd_ead.md
uv run python -m credit_risk.ecl.run             # IFRS 9 ECL + docs/phase6_ecl.md (~2 min)
uv run python -m credit_risk.monitoring.run      # monitoring + docs/phase7_monitoring.md
uv run python -m credit_risk.warehouse.load_results  # model results into PostgreSQL (bi views)
uv run python -m credit_risk.bi.pbip             # Power BI project + docs/phase8_powerbi.md
uv run python -m credit_risk.reporting.cards     # model cards + model risk register
uv run pytest                                    # unit tests + warehouse checks (-m db)
```

To open the report, see [docs/phase8_powerbi.md](docs/phase8_powerbi.md#opening-the-report).

## Repository layout

```
credit_risk/
  data/         download, raw profile, cleaning, default definition, column roles
  warehouse/    PostgreSQL setup, star schema load, model results load
  scorecard/    WoE binning, scorecard training, metrics
  pd/           calibration, master scale, lifetime PD curves
  loss/         LGD and EAD
  ecl/          staging, scenarios, ECL engine
  monitoring/   stability, discrimination and calibration monitoring
  bi/           Power BI project generator
  reporting/    model cards and risk register
warehouse/sql/  star schema, results tables and BI views
powerbi/        Power BI project (.pbip): model.bim + report.json
docs/           phase reports (generated), summaries (JSON), model cards, risk register
models/         fitted model specs: scorecard, PD calibration, LGD (git-ignored, written by the pipeline)
tests/          unit tests (+ warehouse checks marked db)
```

Data (`raw/`, `processed/`) lives outside the repository, at the paths set in `.env`.

## License

Code: MIT. Data: Lending Club loan data as republished on Kaggle under CC0.
