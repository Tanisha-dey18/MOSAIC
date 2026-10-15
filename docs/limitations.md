# Remaining limitations and unverified features

## Verified in this environment

| Item | Evidence |
|---|---|
| Real dataset downloaded and verified | 708/708 files matched `SHA256SUMS.txt` |
| Ingestion of all 100 sessions | API ingestion → dataset `ready`, 20,654 windows |
| Training and inference of all detector types | 13 methods in the experiment suite; 6 models trained via the API and CLI |
| Experiments E1–E9 on real data | `reports/experiments/main/` (runtime 1,060 s) |
| Automated tests | 91 passed (`python -m pytest`) |
| Alembic migration | upgrade, downgrade and upgrade on SQLite; executed by the test fixture |
| Frontend | `tsc` strict and `vite build` pass; Dashboard, Monitoring and Explainability checked in a browser against the live backend with real data |
| Report generation | HTML report with embedded figures generated from run `main` |
| Phone-width layout | All 8 pages measured at 375 px: no horizontal page scroll (wide tables scroll inside their cards) |

## Not verified

| Item | Reason |
|---|---|
| Docker images and `docker compose up` | The Docker daemon was not running on the development machine |
| PostgreSQL | No PostgreSQL server was available; tests used SQLite. The migration is dialect-neutral but was not executed on PostgreSQL |
| Cloud deployment (Netlify and backend host) | Not performed. There is no deployed URL |
| ZIP upload of the full PhysioNet release through the UI | Code path tested with small archives; the 70 MB archive was not uploaded through the browser |
| Literature DOIs other than the dataset references | Listed from standard sources; verify before submission |

## Methodological limitations

See [report.md §22](report.md#22-limitations). In short:

* proxy labels;
* a small, homogeneous laboratory cohort;
* a post-hoc secondary analysis;
* short personal baselines without multi-day history;
* offline processing only;
* no multiple-comparison correction in the tables.

## Requirement checklist

| Requirement | Status |
|---|---|
| Real dataset investigation and selection, with provenance documented | Done |
| Configurable ingestion (Empatica CSV, generic CSV, ZIP), quality checks | Done |
| Statistical baseline (rolling/median/MAD robust z, configurable thresholds) | Done |
| Isolation Forest, One-Class SVM, PyTorch autoencoder | Done |
| Personalised baseline with eligibility, period, fallback and insufficient-data reporting | Done |
| Temporal model (LSTM/GRU) | Deliberately not implemented (justified in report §8) |
| Leakage-free evaluation, metrics, CIs, ablations, error analysis | Done |
| Explainability | Done (exact decompositions and occlusion; SHAP not used, justified) |
| FastAPI backend with all listed endpoints, pagination, logging and error handling | Done |
| SQLAlchemy models and Alembic migrations | Done (verified on SQLite only) |
| React frontend with 8 pages | Done |
| Background training jobs | Done (thread worker plus experiment subprocess) |
| Security measures and tests | Done |
| Docker, Compose and Netlify configuration | Written, not executed |
| Academic documentation and viva | Done |
| Deployment | Not performed |
