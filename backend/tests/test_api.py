"""Integration tests for the FastAPI backend (SYNTHETIC data in the PhysioNet format)."""
from __future__ import annotations

import io
import zipfile

import pytest

from ml.tests.synthetic import generic_csv_frame


# ------------------------------------------------------------------ empty state
def test_health_and_empty_state(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["database"]["ok"] and body["status"] == "ok"
    # No dataset yet: honest empty state, not fabricated data (runs before the ingest fixture).
    meta = client.get("/api/dataset/metadata").json()
    if not meta["configured"]:
        assert "how_to_import" in meta
        r = client.get("/api/participants")
        assert r.status_code == 409 and r.json()["error"]["code"] == "dataset_not_configured"


def test_detect_without_model_is_refused(client):
    r = client.post("/api/anomalies/detect", json={"model_id": 12345, "session_key": "STRESS_S01"})
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"


# ------------------------------------------------------------------ ingestion
def test_sources_listing(client):
    paths = [s["path"] for s in client.get("/api/dataset/sources").json()["sources"]]
    assert "synthetic_release" in paths


@pytest.mark.parametrize("path", ["../", "../../etc", "/etc/passwd", "C:\\Windows", "synthetic_release/../../.."])
def test_ingest_path_traversal_blocked(client, path):
    r = client.post("/api/dataset/ingest", json={"path": path})
    assert r.status_code in (400, 404)
    assert r.json()["error"]["code"] in ("invalid_path", "not_found")


def test_ingest_validation_error(client):
    r = client.post("/api/dataset/ingest", json={"path": "synthetic_release", "feature_config": {"window_s": 10, "step_s": 20}})
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"


def test_ingest_and_metadata(client, ingested):
    assert ingested["n_sessions"] == 7 and ingested["n_participants"] == 7
    meta = client.get("/api/dataset/metadata").json()
    assert meta["configured"] and set(meta["modalities"]) == {"EDA", "HR", "TEMP", "BVP", "ACC"}
    assert meta["sampling_rates"]["BVP"] == [64.0]


def test_sessions_signals_quality(client, ingested):
    sessions = client.get("/api/sessions").json()
    assert len(sessions) == 7
    s02 = next(s for s in sessions if s["session_key"] == "STRESS_S02")
    assert s02["excluded"] and s02["exclusion_reason"]
    sig = client.get("/api/sessions/STRESS_S01/signals", params={"signals": "EDA,HR", "max_points": 100}).json()
    assert sig["signals"] == ["EDA", "HR"] and len(sig["points"]) <= 100 and sig["phases"]
    r = client.get("/api/sessions/STRESS_S01/signals", params={"signals": "XYZ"})
    assert r.status_code == 422
    r = client.get("/api/sessions/STRESS_S01/signals", params={"start_s": 500, "end_s": 100})
    assert r.status_code == 422
    assert client.get("/api/sessions/NOPE/signals").status_code == 404
    q = client.get("/api/dataset/quality").json()
    assert q["n_sessions"] == 7 and "EDA" in q["per_modality"]


def test_participants_and_descriptive_baseline(client, ingested):
    parts = client.get("/api/participants").json()["participants"]
    status = {p["participant"]: p["baseline_status"] for p in parts}
    assert status["S01"] == "available"
    b = client.get("/api/participants/S01/baseline").json()
    assert b["status"] == "ok" and b["period"]["end_s"] <= 300
    assert client.get("/api/participants/ZZZ/baseline").status_code == 404


# ------------------------------------------------------------------ training
def test_train_rejects_bad_requests(client, ingested):
    assert client.post("/api/models/train", json={"method": "bogus"}).json()["error"]["code"] == "unknown_method"
    r = client.post("/api/models/train", json={"method": "iforest_global", "params": {"__class__": 1}})
    assert r.status_code == 422
    r = client.post("/api/models/train", json={"method": "iforest_global", "modalities": ["XYZ"]})
    assert r.status_code == 422
    r = client.post("/api/models/train", json={"method": "hr_rule", "modalities": ["EDA"]})
    assert r.status_code == 422


def test_training_creates_versioned_model(client, trained):
    assert trained["method"] == "robust_z_personal" and trained["version"] >= 1 and trained["artifact_ok"]
    assert trained["metrics"]["n_participants"] == 6  # S02 excluded
    metrics = client.get(f"/api/models/{trained['id']}/metrics").json()
    assert metrics["evaluations"][0]["curves"]["roc"]
    runs = client.get("/api/models/training-runs").json()
    assert runs["total"] >= 1 and runs["items"][0]["status"] in ("completed", "failed")
    methods = client.get("/api/models/methods").json()["methods"]
    assert any(m["method"] == "autoencoder_personal" for m in methods)


def test_baseline_with_model(client, trained):
    b = client.get(f"/api/participants/f01/baseline", params={"model_id": trained["id"]}).json()
    assert b["status"] == "ok" and b["stored_profile"] and len(b["features"]) == 14


# ------------------------------------------------------------------ detection & explanation
def test_detect_and_explain(client, trained):
    r = client.post("/api/anomalies/detect", json={"model_id": trained["id"], "session_key": "STRESS_f01"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["mode"] == "offline_retrospective" and d["n_windows"] == len(d["timeline"])
    assert all(p["flagged"] == (p["score"] is not None and p["score"] > d["threshold"]) for p in d["timeline"])
    # Explicit low threshold -> some windows flagged and stored as events.
    d2 = client.post("/api/anomalies/detect", json={"model_id": trained["id"], "session_key": "STRESS_f01", "threshold": 0.5}).json()
    assert d2["n_flagged"] > 0 and len(d2["event_ids"]) == d2["n_flagged"]
    page = client.get("/api/anomalies", params={"detection_id": d2["detection_id"], "page_size": 5}).json()
    assert page["total"] == d2["n_flagged"] and len(page["items"]) <= 5
    ev = client.get(f"/api/anomalies/{d2['event_ids'][0]}").json()
    exp = ev["explanation"]
    assert exp["model"]["method"] == "robust_z_personal" and exp["contributions_supported"]
    assert exp["features"][0]["contribution"] is not None and exp["caveats"]
    assert client.get(f"/api/anomalies/detections/{d2['detection_id']}").json()["n_flagged"] == d2["n_flagged"]
    r = client.post("/api/explain", json={"model_id": trained["id"], "session_key": "STRESS_f01", "window_index": 99999})
    assert r.status_code == 404


def test_detect_input_validation(client, trained):
    r = client.post("/api/anomalies/detect", json={"model_id": trained["id"], "session_key": "../../etc"})
    assert r.status_code == 422
    r = client.post("/api/anomalies/detect", json={"model_id": trained["id"], "session_key": "STRESS_NOPE"})
    assert r.status_code == 404


def test_missing_artifact_gives_model_not_ready(client, trained, env):
    from app.core.config import get_settings
    from app.services.anomalies import clear_cache

    r = client.post("/api/models/train", json={"method": "iforest_global", "n_folds": 3})
    run = client.get(f"/api/models/training-runs/{r.json()['id']}").json()
    mv = client.get(f"/api/models/{run['model_version_id']}").json()
    art = get_settings().storage_dir / "artifacts" / "iforest_global"
    for f in art.rglob("model.bin"):
        f.unlink()
    clear_cache()
    r = client.post("/api/anomalies/detect", json={"model_id": mv["id"], "session_key": "STRESS_S01"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "model_not_ready"


# ------------------------------------------------------------------ uploads
def test_upload_validation(client):
    r = client.post("/api/dataset/validate", files={"file": ("x.exe", b"MZ...", "application/octet-stream")})
    assert r.status_code == 415
    r = client.post("/api/dataset/validate", files={"file": ("x.csv", b"", "text/csv")})
    assert r.status_code == 400
    r = client.post("/api/dataset/validate", files={"file": ("big.csv", b"a" * (3 * 1024 * 1024), "text/csv")})
    assert r.status_code == 413
    r = client.post("/api/dataset/validate", files={"file": ("bad.csv", b"foo,bar\n1,2\n", "text/csv")})
    assert r.status_code == 200 and r.json()["valid"] is False
    r = client.post("/api/dataset/validate", files={"file": ("bad.zip", b"not a zip", "application/zip")})
    assert r.status_code == 400 and r.json()["error"]["code"] == "corrupted_file"


def test_zip_slip_rejected(client):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../../evil.csv", "a,b\n1,2\n")
    r = client.post("/api/dataset/validate", files={"file": ("evil.zip", buf.getvalue(), "application/zip")})
    assert r.status_code == 400 and r.json()["error"]["code"] == "unsafe_archive"


def test_generic_csv_upload_ingests(client, ingested):
    data = generic_csv_frame(2, minutes=10).to_csv(index=False).encode()
    r = client.post("/api/dataset/validate", files={"file": ("u.csv", data, "text/csv")})
    assert r.json()["valid"] and r.json()["participants"] == 2
    r = client.post("/api/dataset/upload", files={"file": ("u.csv", data, "text/csv")}, data={"name": "test upload"})
    assert r.status_code == 202
    ds = client.get(f"/api/datasets/{r.json()['id']}").json()
    assert ds["status"] == "ready" and ds["source_type"] == "generic_csv" and ds["n_participants"] == 2


# ------------------------------------------------------------------ reports & experiments
def test_reports(client, trained):
    r = client.post("/api/reports", json={"title": "Test report"})
    assert r.status_code == 201
    rep = r.json()
    dl = client.get(f"/api/reports/{rep['id']}/download")
    assert dl.status_code == 200 and "text/html" in dl.headers["content-type"]
    assert b"No experiment run is available" in dl.content  # honest when no results exist
    assert client.get("/api/reports").json()["total"] >= 1
    assert client.get("/api/reports/9999/download").status_code == 404
    r = client.post("/api/reports", json={"title": "x", "experiment_run_id": "../etc"})
    assert r.status_code == 422


def test_experiments_endpoints(client):
    assert client.get("/api/experiments").json()["runs"] == []
    assert client.get("/api/experiments/nope").status_code == 404
    assert client.get("/api/experiments/nope/figures/..%2Fsecret.png").status_code in (400, 404)
    assert client.get("/api/experiments/run1/figures/evil.svg").status_code == 400


def test_unknown_route_and_method(client):
    assert client.get("/api/does-not-exist").json()["error"]["code"] == "not_found"
    assert client.delete("/api/health").status_code == 405


def test_cors_allows_configured_origin_only(client):
    ok = client.options("/api/health", headers={"Origin": "http://testserver", "Access-Control-Request-Method": "GET"})
    assert ok.headers.get("access-control-allow-origin") == "http://testserver"
    bad = client.options("/api/health", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" not in bad.headers
    # Non-production: any local dev-server port (Vite falls back to 5174, 5175, ... when 5173 is busy).
    local = client.options("/api/health", headers={"Origin": "http://localhost:5176", "Access-Control-Request-Method": "GET"})
    assert local.headers.get("access-control-allow-origin") == "http://localhost:5176"
    spoof = client.options("/api/health", headers={"Origin": "http://localhost.evil.example", "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" not in spoof.headers


def test_production_cors_is_allow_list_only(env, monkeypatch):
    from fastapi.testclient import TestClient

    from app.core.config import get_settings
    from app.main import create_app

    monkeypatch.setenv("MOSAIC_ENV", "production")
    get_settings.cache_clear()
    try:
        prod = TestClient(create_app())
        r = prod.options("/api/health", headers={"Origin": "http://localhost:5176", "Access-Control-Request-Method": "GET"})
        assert "access-control-allow-origin" not in r.headers
    finally:
        monkeypatch.setenv("MOSAIC_ENV", "test")
        get_settings.cache_clear()


def test_bootstrap_is_idempotent_when_data_and_models_exist(client, trained):
    from app.services.bootstrap import run_bootstrap

    before = len(client.get("/api/datasets").json())
    runs_before = client.get("/api/models/training-runs").json()["total"]
    run_bootstrap()  # must not download, re-ingest or retrain
    assert len(client.get("/api/datasets").json()) == before
    assert client.get("/api/models/training-runs").json()["total"] == runs_before
