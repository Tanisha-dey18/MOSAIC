# MOSAIC

**Multimodal Observation System for Anomaly Identification and Clinical Trends**: a final-year B.Tech CSE (Data Science) major project.

MOSAIC is a research platform for studying whether **personalised baselines** improve the detection of unusual physiological patterns in wearable-sensor data, compared with global thresholds and non-personalised machine learning. It contains a complete pipeline from raw Empatica E4 files to a web application:

* ingestion with data-quality checks,
* signal-specific preprocessing and window features,
* statistical, Isolation Forest, One-Class SVM and PyTorch autoencoder detectors, each with a *global*, *personal* and (where applicable) *rolling* reference,
* leakage-free participant-grouped evaluation,
* explanations,
* a FastAPI + PostgreSQL backend,
* a React dashboard,
* reproducible experiments and generated reports.

> **Research use only.** MOSAIC is not a medical device, is not clinically validated and must not be used to diagnose disease, predict medical emergencies or replace professional assessment. Scores are relative deviation measures, not probabilities.

## Dataset

The project uses the open-access PhysioNet **Wearable Device Dataset from Induced Stress and Structured Exercise Sessions**, v1.0.1 (Hongn, Bosch, Prado & Bonomini, 2025):

* DOI: [10.13026/he0v-tf17](https://doi.org/10.13026/he0v-tf17)
* License: ODC-By 1.0
* Device: Empatica E4, recording EDA (4 Hz), BVP (64 Hz), HR (1 Hz), skin temperature (4 Hz), 3-axis acceleration (32 Hz) and IBI
* Sessions: 36 participants in the STRESS protocol, plus aerobic and anaerobic exercise sessions
* Labels: protocol phases recovered from the button tags (stressor task vs rest)

The data is **not** committed. Download it and verify it with `python scripts/download_dataset.py`. See [datasets/README.md](datasets/README.md).

## Key results (measured, experiment run `exp_20261015T064235Z`)

Setup: 33 eligible participants, 3,903 evaluation windows (789 stressor / 3,114 rest), 5-fold participant-grouped CV, 95% CIs from a participant-level bootstrap. Full tables are in [reports/experiments/main/summary.md](reports/experiments/main/summary.md) (generated, not hand-written) and in [docs/report.md](docs/report.md).

| Method | ROC-AUC [95% CI] | Median per-participant AUC |
|---|---|---|
| Fixed rule HR > 100 bpm | 0.680 [0.613, 0.765] | 0.930 |
| Robust z: global | 0.587 [0.535, 0.639] | 0.654 |
| Robust z: personal (fixed pre-stress baseline) | 0.420 [0.329, 0.513] | 0.367 |
| Isolation Forest: global | 0.598 [0.548, 0.654] | 0.672 |
| Arousal-directed z: rolling personal *(post hoc)* | 0.796 [0.745, 0.846] | 0.875 |

* **The primary hypothesis was not supported.** A fixed 5-minute personal baseline made two-sided detectors *worse* than global references, because slow drift (especially skin temperature) and relaxation below baseline also count as "deviations".
* An exploratory, post-hoc variant (an adaptive rolling personal baseline with arousal-directed scoring) performed best. It needs confirmation on independent data.

## Quick start (local, no Docker)

Requires Python 3.12+ (tested with 3.13) and Node 20+ (tested with 24).

```bash
pip install -r backend/requirements-dev.txt
```

```bash
cd frontend && npm install && cd ..
```

```bash
python scripts/download_dataset.py --out datasets/raw/wearable-device-dataset-1.0.1
```

Create the database schema (SQLite by default; set `DATABASE_URL` for PostgreSQL):

```bash
cd backend && alembic upgrade head && cd ..
```

```bash
python scripts/ingest.py --path datasets/raw/wearable-device-dataset-1.0.1
```

The ingest script prints the processed-features directory (`storage/processed/dataset_<id>`), which the experiment command below needs.

```bash
python scripts/train.py
```

```bash
python -m ml.experiments.run_experiments --features-dir storage/processed/dataset_<id> --out reports/experiments/main
```

Start the API (http://localhost:8000/docs) and the UI (http://localhost:5173) in two terminals:

```bash
uvicorn app.main:app --app-dir backend --port 8000
```

```bash
npm --prefix frontend run dev
```

Run the tests:

```bash
python -m pytest
```

## Docker (PostgreSQL + API + UI)

```bash
cp .env.example .env
```

Set `POSTGRES_PASSWORD` in `.env`, then:

```bash
docker compose up --build
```

The UI is at http://localhost:8080 and the API at http://localhost:8000/docs. Ingest from the Data Explorer page; `./datasets` is mounted read-only into the backend.

## Repository layout

```
MOSAIC/
├── ml/                     # research code (importable package)
│   ├── data_ingestion/     # Empatica reader, PhysioNet adapter, generic CSV, session model
│   ├── preprocessing/      # quality checks, signal-specific filters
│   ├── features/           # windowing + 14 interpretable features
│   ├── models/             # normalisation (global/personal/rolling), detectors, autoencoder
│   ├── evaluation/         # grouped CV, metrics, bootstrap, plots
│   ├── experiments/        # run_experiments.py (E1–E9), train.py
│   └── tests/              # unit tests + SYNTHETIC test-data generator
├── backend/                # FastAPI app, SQLAlchemy models, Alembic migrations, API tests
├── frontend/               # React + Vite + TypeScript + Tailwind + Recharts
├── scripts/                # download_dataset.py, ingest.py, train.py
├── docs/                   # academic report, architecture, API, deployment, testing, viva
├── datasets/               # (git-ignored) raw data; README explains access
├── reports/experiments/    # generated experiment results (results.json, summary.md, figures)
├── docker-compose.yml, .env.example, pytest.ini
```

## Documentation

* [docs/report.md](docs/report.md): full academic project report (all chapters, results, limitations, references)
* [docs/architecture.md](docs/architecture.md): system architecture, data flow, UML and ER diagrams (Mermaid)
* [docs/api.md](docs/api.md): API reference
* [docs/deployment.md](docs/deployment.md): Netlify + backend/database deployment
* [docs/testing.md](docs/testing.md): test report and reproduction commands
* [docs/viva.md](docs/viva.md): viva questions and answers
* [docs/limitations.md](docs/limitations.md): remaining limitations and unverified features

## Citation of the data

Hongn, A., Bosch, F., Prado, L., & Bonomini, P. (2025). *Wearable Device Dataset from Induced Stress and Structured Exercise Sessions* (version 1.0.1). PhysioNet. https://doi.org/10.13026/he0v-tf17
