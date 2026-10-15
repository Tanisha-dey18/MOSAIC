# Testing report

## How to run

```bash
pip install -r backend/requirements-dev.txt
```

```bash
python -m pytest
```

```bash
npm --prefix frontend run typecheck
```

```bash
npm --prefix frontend run build
```

## Result (development machine, 15 Oct 2026)

Windows 11, Python 3.13.15, pytest 9.1.1:

```
================== 93 passed, 1 warning in 79.86s ==================
```

The single warning is a Starlette deprecation notice about the `httpx` test client; it does not affect the results. The frontend passed `tsc --noEmit` (strict mode) and `vite build` without errors.

Tests use **synthetic** data in the PhysioNet file format (`ml/tests/synthetic.py`), so they run without downloading the dataset and never mix test data with research results.

## Coverage by requirement

| Area | Tests (file::test) |
|---|---|
| Dataset validation | `test_ingestion.py`: UTC and Unix headers, invalid headers (parametrised), corrupted rows → NaN, empty file, tags and IBI, wrong tag count, missing modality file, unrelated folder, generic CSV errors, duplicates and gaps |
| Preprocessing | `test_preprocessing_features.py`: plausibility mask, flat-line detection, constant signal → unusable, empty signal, filters preserve length and NaN positions, ACC magnitude |
| Feature generation | canonical feature order, modality subsets, missing and invalid modalities → NaN, low-quality window rejected, constant signal |
| Missing values | the above, plus `test_sensor_dropout_masks_only_post_calibration` |
| Leakage prevention | windows never cross phase or calibration boundaries; evaluation windows start after the baseline; rest guard; excluded sessions; disjoint folds; **no participant in both train and test**; identical windows across methods |
| Statistical detection | robust z contributions sum to s²; arousal direction is one-sided; global vs personal normalisation |
| Isolation Forest / OCSVM / autoencoder | fit and score for all 11 method variants; determinism with seed; constant features; autoencoder early stopping |
| Personal baselines | baseline unchanged when only future data changes; insufficient history reported and unscored |
| Serialisation | save/load round-trip with identical scores for 5 methods; missing artifact; corrupted `model.bin` (checksum); missing weights; tampered feature schema; corrupted JSON; untrained model refuses to score or save |
| Evaluation | known confusion-matrix values; event sensitivity; false alarms/h; bootstrap CI ordering; paired comparison; train-and-evaluate artifact; unlabelled data gives no metrics |
| API endpoints | `backend/tests/test_api.py`: 25 tests covering health, empty state, ingestion, sessions, signals, quality, participants, baselines, training, models, metrics, detection, events, pagination, explanations, reports, experiments, unknown routes |
| Database | the schema is created by the **Alembic migration** in the test fixture; ORM writes and reads through every endpoint |
| Error responses | 404 / 405 / 409 `model_not_ready` / 413 / 415 / 422 validation / `unknown_method` |
| File upload validation | wrong extension, empty file, oversize file, invalid CSV, corrupted ZIP, **zip-slip** |
| Security | **path traversal** (5 variants), figure-name and run-id validation, session-key pattern, parameter whitelist, CORS allow/deny |
| Report generation | create, list, download, honest "no results" text, invalid run id |

## Edge cases explicitly covered

* empty datasets (empty CSV, file without samples)
* insufficient history
* missing modalities
* constant-valued signals
* invalid timestamps
* corrupted files
* missing model artifacts
* invalid API requests

## Bugs found by the tests and fixed

1. An invalid timestamp raised pandas' `DateParseError` instead of the documented `IngestionError`.
2. `GET /api/sessions` failed because the schema field `participant` collided with the ORM relationship.
3. Baseline and explanation responses could contain NaN, which is invalid JSON. A shared `jsonsafe.clean` now converts it to `null`.
4. A first ingestion through the API failed because a new participant's session counter was `None` before flush.

## Not covered by automated tests

* Frontend unit tests: none are written. The frontend was checked with strict type-checking, a production build, and a manual browser check of the Dashboard, Monitoring and Explainability pages against the real backend.
* PostgreSQL-specific behaviour. Tests run on SQLite; the same Alembic migration targets PostgreSQL but was not executed against a PostgreSQL server here.
* Docker images (not built; see deployment.md).
