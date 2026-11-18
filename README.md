# Credit Risk Scorecard and IFRS 9 Expected Credit Loss

Credit risk modelling on 2.2 million real consumer loans from
[Lending Club](https://www.kaggle.com/datasets/wordsforthewise/lending-club) (2007–2018,
CC0 public domain): an application scorecard, PD / LGD / EAD models, an IFRS 9 expected
credit loss (ECL) calculation with staging and macroeconomic scenarios, model monitoring,
and a Power BI risk dashboard.

> **Historical data.** Loans issued 2007–2018, with statuses as of March 2019. Nothing here
> is live data, and the IFRS 9 reporting date is 2019-03-31.

## Goals

| Question a bank asks | Where this project answers it |
|---|---|
| Should we approve this applicant, and at what price? | Application scorecard (WoE/IV, logistic regression, points scale) |
| What share of loans will default in the next 12 months, and over their life? | PD model, rating master scale, lifetime PD term structure |
| How much do we lose when a loan defaults? | LGD from recoveries, EAD from amortisation schedules |
| How much should we provision today? | IFRS 9 ECL: stage 1/2/3, 12-month vs lifetime, weighted scenarios |
| Is the model still working? | Monitoring: Gini by vintage, PSI, calibration back-tests |
| How is the portfolio doing? | Power BI: portfolio quality, vintages, provisions by stage, concentration |

## Key facts so far

- 2,257,919 loans after removing 33 footer lines and 2,749 loans from a pre-2010 credit policy.
- 12-month default rate 5.80% on the 1,762,677 loans with a complete 12-month window
  (from 1.48% for grade A to 22.68% for grade G).
- Only 41% of defaults happen in the first 12 months, so lifetime PD matters for IFRS 9.
- Open book at the 2019-03-31 reporting date: 912,609 loans, $9.51 bn outstanding principal.
- Column roles (application, pricing, outcome) are fixed in
  [`credit_risk/data/columns.py`](credit_risk/data/columns.py), so post-origination fields cannot
  leak into the scorecard.

## Plan

Each phase is checked before the next, with one commit per phase.

| Phase | Scope | Status |
|---|---|---|
| 0 | Setup, download with SHA-256 manifest, raw data profile ([findings](docs/phase0_findings.md)) | Done |
| 1 | Default definition, observation windows, cleaning, leakage review ([report](docs/phase1_report.md)) | Done |
| 2 | PostgreSQL star schema, BI views, least-privilege roles, reconciled load ([docs](docs/star_schema.md)) | Done |
| 3 | Application scorecard: 12 features, WoE/IV, points scale; out-of-time test Gini 0.343 vs XGBoost 0.390 and LC sub-grade 0.399 ([report](docs/phase3_scorecard.md)) | Done |
| 4 | Point-in-time PD calibration (2017 back-test 6.14% vs 6.13% observed), R1–R10 master scale, lifetime PD curves with prepayment as a competing risk ([report](docs/phase4_pd.md)) | Done |
| 5 | LGD from 174,565 completed recovery workouts (discounted at the loan rate, 90.9%), EAD from contractual amortisation ([report](docs/phase5_lgd_ead.md)) | Done |
| 6 | IFRS 9 ECL: staging with updated FICO and delinquency, lifetime ECL engine, three Vasicek scenarios, matured-vintage back-test; $920.9M on $9.51bn (9.7%) ([report](docs/phase6_ecl.md)) | Done |
| 7 | Monitoring: PSI, Gini by vintage, calibration back-tests | |
| 8 | Power BI risk dashboard | |
| 9 | Documentation, model cards, limitations | |

## Setup

Requires [uv](https://docs.astral.sh/uv/), Python 3.11 (uv installs it) and PostgreSQL 16.

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
uv run pytest                                    # unit tests + warehouse checks (-m db)
```

## Repository layout

```
credit_risk/        Python package: config, data download and profiling, models (later phases)
data/raw/           Downloaded files + _manifest.json (git-ignored; may live on another drive)
data/processed/     Cleaned and derived datasets (git-ignored)
docs/               Findings, data profile, model cards
warehouse/sql/      Star schema DDL and BI views (PostgreSQL)
powerbi/            Power BI project (PBIP)
notebooks/          Exploration
tests/              Unit tests
```

## License

Code: MIT. Data: Lending Club loan data as republished on Kaggle under CC0.
