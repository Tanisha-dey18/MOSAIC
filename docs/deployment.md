# Deployment

> **Status:** MOSAIC has **not** been deployed to any cloud service as part of this project. The configuration below was written for deployment but has not been verified in production. Docker was installed on the development machine but its daemon was not running, so the Docker images were **not built or run**. Follow the verification checklist at the end after deploying.

## Architecture

Netlify serves only static files, so it cannot run the persistent Python/ML workload. The deployment is therefore split:

| Part | Where | Notes |
|---|---|---|
| Frontend (React build) | **Netlify** | `netlify.toml` at the repository root |
| Backend (FastAPI + ML) | Any container host with a persistent disk (e.g. Render, Railway, Fly.io, a VM) | `backend/Dockerfile` |
| Database | Managed PostgreSQL (e.g. Neon, Supabase, Render, RDS) | Migrations run automatically on container start |
| Storage | Persistent volume mounted at `/data/storage` | Features, artifacts, reports. Must survive restarts |
| Dataset | Volume at `/data/datasets` (read-only), or upload via the UI | Raw files are not in the image |

## 1. Database

Create a PostgreSQL 14+ database and user, then note the connection string:

```
postgresql+psycopg2://USER:PASSWORD@HOST:5432/DBNAME?sslmode=require
```

Migrations run automatically at start-up via `alembic upgrade head` in the container command. To run them manually:

```bash
cd backend && DATABASE_URL=... alembic upgrade head
```

## 2. Backend

Build from the repository root:

```bash
docker build -f backend/Dockerfile -t mosaic-backend .
```

Required environment variables:

| Variable | Example | Notes |
|---|---|---|
| `DATABASE_URL` | `postgresql+psycopg2://…` | Secret, set in the host's secret store |
| `CORS_ORIGINS` | `https://mosaic-yourname.netlify.app` | Exact frontend origin(s), comma-separated; never `*` |
| `STORAGE_DIR` | `/data/storage` | Mount a persistent disk here |
| `DATASETS_DIR` | `/data/datasets` | Optional volume with the PhysioNet folder |
| `REPORTS_DIR` | `/data/reports` | Experiment runs. Mount persistently |
| `MAX_UPLOAD_MB` | `200` | Upload limit |
| `MOSAIC_ENV` | `production` | **Required.** In any other mode, CORS also accepts any `localhost`/`127.0.0.1` port for local development |
| `PORT` | set by the host | The container listens on `$PORT` (default 8000) |

The image includes a `HEALTHCHECK` on `/api/health`. Configure the platform's health check to the same path.

**Persistent model storage.** Model artifacts live in `$STORAGE_DIR/artifacts/<method>/v<version>_run<id>/` (`metadata.json`, `model.bin`). If the disk is lost, the database still references these artifacts, and detection returns `409 model_not_ready` rather than a fake prediction. Retrain to recover.

**Dataset on the server.** Either copy `datasets/raw/wearable-device-dataset-1.0.1` to the dataset volume and call `POST /api/dataset/ingest`, or run `python scripts/download_dataset.py --out /data/datasets/raw/wearable-device-dataset-1.0.1` once inside the container. That requires write access to the volume.

**Scaling.** Use a single instance with one worker: the job runner is in-process. For more instances, move jobs to a queue (Celery/RQ + Redis) and storage to object storage.

## 3. Frontend on Netlify

1. Create a new site from the repository.
2. Netlify reads `netlify.toml` from the repository root, which sets the base directory `frontend`, the build command `npm ci && npm run build` and the publish directory `dist`.
3. Under *Site settings → Environment variables*, set `VITE_API_BASE_URL=https://<your-backend-host>` (no trailing slash). It is embedded at build time, so **redeploy after changing it**. It must not be `localhost`.
4. The SPA redirect (`/* → /index.html`) and security headers are already configured.
5. Add the Netlify URL to the backend's `CORS_ORIGINS` and restart the backend.

## 4. Local full stack with Docker Compose

```bash
cp .env.example .env
```

Edit `POSTGRES_PASSWORD`, then:

```bash
docker compose up --build
```

The frontend is at http://localhost:8080, the API at http://localhost:8000/docs, and PostgreSQL runs inside the compose network.

## 5. Verification checklist (to perform after deployment)

```bash
curl -s https://<backend>/api/health
```

Expect `"status":"ok"` and `"database":{"ok":true}`.

* `GET /api/dataset/metadata` returns `configured:false` before ingestion and dataset details after.
* Open the Netlify site. The header should show "API online", and the browser console should have no CORS errors.
* Train a model in the Model Laboratory and confirm the run reaches `completed`.
* Run an analysis in Anomaly Monitoring and open an explanation.
* Generate and download a report.
* Restart the backend and confirm models and reports persist.

## Secrets

Never commit `.env`, database URLs, or raw data. `.gitignore` already excludes `.env*` (except the examples), `datasets/*`, `storage/` and `artifacts/*`.
