# Deployment

> **Status:** The frontend has been deployed by the author to https://mosaicappn.netlify.app. The backend start-up bootstrap (download → ingest → train) was tested locally, from an empty database, in production mode. The Hugging Face Docker image itself has **not** been built here, because the Docker daemon was not running on the development machine. Follow the verification checklist at the end after deploying.

## Recommended free setup: Hugging Face Spaces (backend) + Netlify (frontend)

The live frontend is at https://mosaicappn.netlify.app. Netlify serves only static files, so the backend runs as a free **Hugging Face Docker Space** (2 vCPU, 16 GB RAM). Render's free tier (512 MB RAM, 0.1 CPU) is too small for this ML stack.

### Step 1: create the backend Space (about 10 minutes, mostly automatic)

1. Sign in at https://huggingface.co and choose **New → Space**.
2. Enter a name such as `mosaic-api`, select the **Docker** SDK (blank template) and **CPU basic (free)** hardware, and make the Space **public**. A private Space cannot be called from the Netlify site.
3. In the Space, open **Files → Add file → Create a new file** twice. Create `Dockerfile` and `README.md` with the contents of [`deploy/huggingface/Dockerfile`](../deploy/huggingface/Dockerfile) and [`deploy/huggingface/README.md`](../deploy/huggingface/README.md).
4. The Space builds from the public GitHub repository. On first start it downloads the PhysioNet dataset (checksums verified), processes it and trains the default models. The API answers immediately; data and models appear over several minutes.
5. Note the Space's URL, `https://<hf-username>-mosaic-api.hf.space`, and check it:

   ```bash
   curl https://<hf-username>-mosaic-api.hf.space/api/health
   ```

   Expect `"status":"ok"`; `"dataset"` becomes `configured: true` once ingestion finishes.

Free Spaces have no persistent disk and sleep after a period without visits. After every restart, the bootstrap rebuilds the data and models automatically. Trained models, detections and reports created in the app are lost on restart. To keep them, add persistent storage or move to a host with a disk (section 2 below).

If the Netlify site URL changes, set **Settings → Variables → `CORS_ORIGINS`** in the Space to the new `https://…netlify.app` origin, then restart the Space. After pushing new code to GitHub, use **Settings → Factory rebuild**.

### Connecting the site to a backend without rebuilding

The site chooses its backend **in the browser**, so Netlify never needs rebuilding when the backend address changes:

* open `https://mosaicappn.netlify.app/?api=https://<backend-address>`. The address is remembered in that browser and removed from the address bar; or
* open the site and use the **Connect** form (or the connection menu in the header). "Copy shareable link" produces the `?api=` link for others.

Only `https://` addresses are accepted (`http://` only for localhost). `VITE_API_BASE_URL` (step 2) is optional: it sets the default backend for visitors who arrive without a link.

**Temporary demo without any account (backend on your laptop):**

```bash
python -m uvicorn app.main:app --app-dir backend --port 8000
```

```bash
cloudflared tunnel --url http://localhost:8000
```

Share `https://mosaicappn.netlify.app/?api=<the https://….trycloudflare.com address it prints>`. The backend's `.env` must list the Netlify origin in `CORS_ORIGINS`. It works only while both commands run, and the tunnel address changes on every restart (send the new link).

### Step 2: point the Netlify site at the backend (optional default)

1. In Netlify, open the site and go to **Site configuration → Environment variables → Add a variable**.
2. Set `VITE_API_BASE_URL` = `https://<hf-username>-mosaic-api.hf.space` (no trailing slash).
3. Choose **Deploys → Trigger deploy → Deploy site**. The URL is compiled into the build, so a redeploy is required.

The build now **fails on purpose** if `VITE_API_BASE_URL` is missing, points to localhost, or is not `https`. This prevents deploying a site that cannot reach any backend, which is what happened with the first deployment: it was built with the default `http://localhost:8000`.

---

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
