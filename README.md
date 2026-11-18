# Credit Risk Scorecard and IFRS 9 Expected Credit Loss

Credit risk modelling on 2.2 million real consumer loans from
[Lending Club](https://www.kaggle.com/datasets/wordsforthewise/lending-club) (2007–2018,
CC0 public domain): an application scorecard, PD / LGD / EAD models, an IFRS 9 expected
credit loss (ECL) calculation with staging and macroeconomic scenarios, model monitoring,
and a Power BI risk dashboard.

> **Historical data.** Loan statuses are a snapshot taken at the end of 2018. Nothing here is
> live data, and the "reporting date" for ECL is 2018-12-31.

## Goals

| Question a bank asks | Where this project answers it |
|---|---|
| Should we approve this applicant, and at what price? | Application scorecard (WoE/IV, logistic regression, points scale) |
| What share of loans will default in the next 12 months, and over their life? | PD model, rating master scale, lifetime PD term structure |
| How much do we lose when a loan defaults? | LGD from recoveries, EAD from amortisation schedules |
| How much should we provision today? | IFRS 9 ECL: stage 1/2/3, 12-month vs lifetime, weighted scenarios |
| Is the model still working? | Monitoring: Gini by vintage, PSI, calibration back-tests |
| How is the portfolio doing? | Power BI: portfolio quality, vintages, provisions by stage, concentration |

## Plan

Each phase is checked before the next, with one commit per phase.

| Phase | Scope | Status |
|---|---|---|
| 0 | Setup, download with SHA-256 manifest, raw data profile ([findings](docs/phase0_findings.md)) | Done |
| 1 | Default definition, observation windows, cleaning, leakage review (drop post-origination fields) | |
| 2 | Loan-level dataset in PostgreSQL: star schema (loan fact; grade, purpose, state, date dimensions) | |
| 3 | Application scorecard: binning, WoE/IV, logistic regression, points scaling; out-of-time validation (Gini, KS); XGBoost benchmark | |
| 4 | PD calibration, rating master scale, lifetime PD curves from vintage data | |
| 5 | LGD (recoveries, cure) and EAD (amortisation schedule, prepayment) | |
| 6 | IFRS 9 ECL: staging (SICR rules, 30 days past due backstop), discounting, base/upside/downside scenarios | |
| 7 | Monitoring: PSI, Gini by vintage, calibration back-tests | |
| 8 | Power BI risk dashboard | |
| 9 | Documentation, model cards, limitations | |

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 (uv installs it).

```powershell
# Your user environment sets UV_PROJECT_ENVIRONMENT for another project: override it here.
. .\dev.ps1                      # sets UV_PROJECT_ENVIRONMENT for this repo (venv on K:)
uv sync --all-groups

Copy-Item .env.example .env      # then point CREDIT_RAW_DIR to a drive with ~2 GB free
uv run python -m credit_risk.data.download       # ~650 MB, idempotent, hash-checked
uv run python -m credit_risk.data.profile_raw    # writes docs/phase0_profile.json
uv run pytest
```

## Repository layout

```
credit_risk/        Python package: config, data download and profiling, models (later phases)
data/raw/           Downloaded files + _manifest.json (git-ignored; may live on another drive)
data/processed/     Cleaned and derived datasets (git-ignored)
docs/               Findings, data profile, model cards
warehouse/          SQL for the PostgreSQL star schema
powerbi/            Power BI project (PBIP)
notebooks/          Exploration
tests/              Unit tests
```

## License

Code: MIT. Data: Lending Club loan data as republished on Kaggle under CC0.
