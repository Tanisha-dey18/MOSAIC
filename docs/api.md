# MOSAIC API reference

Base URL: `VITE_API_BASE_URL` (e.g. `http://localhost:8000`). The OpenAPI schema is at `/openapi.json`, with Swagger UI at `/docs` and ReDoc at `/redoc`.

**Errors** always have this shape:

```json
{"error": {"code": "model_not_ready", "message": "…", "details": null}}
```

| HTTP | code (examples) | When |
|---|---|---|
| 400 | `invalid_path`, `invalid_name`, `empty_file`, `corrupted_file`, `unsafe_archive` | Unsafe or invalid input |
| 404 | `not_found` | Unknown dataset, session, model, anomaly, report, run or route |
| 409 | `dataset_not_configured`, `dataset_not_ready`, `model_not_ready`, `experiment_running` | Precondition not met |
| 410 | `report_missing`, `signals_missing`, `result_missing` | Stored file gone |
| 413 | `payload_too_large`, `archive_too_large` | Upload limits |
| 415 | `unsupported_file_type` | Not .csv or .zip |
| 422 | `validation_error`, `unknown_method`, `no_features`, `invalid_dataset` | Request validation |
| 429 | `too_many_jobs` | More than 5 queued or running training jobs |
| 500 | `internal_error` | Unexpected error (details only in the server log) |

Paginated responses have the shape `{"items": [...], "total": n, "page": p, "page_size": s}`.

## Health

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Status, version, DB connectivity, storage, active dataset, number of models |

## Dataset

| Method | Path | Body / query | Description |
|---|---|---|---|
| GET | `/api/dataset/metadata` | | Active dataset provenance, modalities, sampling rates, protocols and features. Returns `{"configured": false, "how_to_import": …}` if none |
| GET | `/api/datasets` / `/api/datasets/{id}` | | Ingestion records with status and progress |
| GET | `/api/dataset/sources` | | PhysioNet folders found under `DATASETS_DIR` |
| POST | `/api/dataset/validate` | multipart `file` (.csv/.zip) | Validation report without ingesting |
| POST | `/api/dataset/validate-path` | `{"path"}` | Validate a server folder |
| POST | `/api/dataset/ingest` | `{"path", "protocols"?, "feature_config"?}` | **202**; background ingestion of a server folder |
| POST | `/api/dataset/upload` | multipart `file`, `name` | **202**; validate and ingest an uploaded CSV or ZIP |
| GET | `/api/dataset/quality` | `dataset_id?` | Per-session and per-modality quality, missing-feature fractions, exclusions |
| GET | `/api/sessions` | `participant?`, `protocol?` | Sessions with phases, quality and sampling rates |
| GET | `/api/sessions/{key}/signals` | `start_s`, `end_s`, `signals=EDA,HR`, `max_points` | 1 Hz cleaned signal summaries (block-averaged to `max_points`) |
| GET | `/api/sessions/{key}/windows` | | Window features, quality and labels |

## Participants

| Method | Path | Description |
|---|---|---|
| GET | `/api/participants` | Participants, sessions, complete calibration windows and baseline status (`available`, `insufficient`, `no_calibration_period`) |
| GET | `/api/participants/{id}/baseline?model_id=` | Construction rule and period, baseline statistics per feature, comparison with later observations, deviation trend, insufficient-history warnings |

## Models

| Method | Path | Body | Description |
|---|---|---|---|
| GET | `/api/models/methods` | | Trainable methods with description, limitations and analysis type |
| POST | `/api/models/train` | `{"method", "modalities"?, "params"?, "threshold_quantile"=0.95, "n_folds"=5, "seed"=42}` | **202**; queues a training run |
| GET | `/api/models/training-runs` / `/{id}` | | Training history and status |
| GET | `/api/models` | `method?` | Model versions with summary metrics and artifact status |
| GET | `/api/models/{id}` | | One model version |
| GET | `/api/models/{id}/metrics` | | Stored evaluation: metrics, ROC/PR curves, per-participant AUC, timing, folds |

`params` accepts only these whitelisted numeric or string hyper-parameters: `n_estimators`, `max_samples`, `contamination`, `nu`, `gamma`, `max_train`, `hidden`, `bottleneck`, `dropout`, `lr`, `weight_decay`, `epochs`, `batch_size`, `patience`.

## Anomalies and explanations

| Method | Path | Body / query | Description |
|---|---|---|---|
| POST | `/api/anomalies/detect` | `{"model_id", "session_key", "threshold"?}` | Scores every window of a session offline, stores the run and its flagged windows, and returns the timeline. Returns **409 `model_not_ready`** if the artifact is missing or invalid |
| GET | `/api/anomalies` | `model_id`, `participant`, `session_key`, `detection_id`, `page`, `page_size` | Flagged windows |
| GET | `/api/anomalies/detections` / `/{id}` | | Detection history and a stored timeline |
| GET | `/api/anomalies/{id}` | | One event with a full explanation |
| POST | `/api/explain` | `{"model_id", "session_key", "window_index"}` | Explanation for any window |

## Reports and experiments

| Method | Path | Description |
|---|---|---|
| POST | `/api/reports` | `{"title", "experiment_run_id"?, "include_models"}` generates a self-contained HTML report (**201**) |
| GET | `/api/reports` | Paginated list |
| GET | `/api/reports/{id}/download` | Downloads the HTML report |
| GET | `/api/experiments` | Experiment runs found in `REPORTS_DIR/experiments` |
| GET | `/api/experiments/{run_id}` | Full `results.json` |
| GET | `/api/experiments/{run_id}/figures/{name}.png` | A figure |
| POST | `/api/experiments/run` | `{"quick": false}` launches the E1–E9 suite in a separate process (**202**) |

## Example session

```bash
curl -X POST localhost:8000/api/dataset/ingest -H "Content-Type: application/json" -d '{"path":"raw/wearable-device-dataset-1.0.1"}'
```

```bash
curl -X POST localhost:8000/api/models/train -H "Content-Type: application/json" -d '{"method":"arousal_z_rolling"}'
```

```bash
curl -X POST localhost:8000/api/anomalies/detect -H "Content-Type: application/json" -d '{"model_id":1,"session_key":"STRESS_S05"}'
```
