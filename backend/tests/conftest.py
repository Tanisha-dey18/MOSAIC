"""API test fixtures.

Each test session gets an isolated temporary storage directory, a temporary
SQLite database (schema created by the Alembic migration) and a datasets
directory containing a SYNTHETIC dataset in the PhysioNet file format.
Background jobs run synchronously (JOBS_SYNC=1) for deterministic tests.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))


@pytest.fixture(scope="session")
def env(tmp_path_factory):
    base = tmp_path_factory.mktemp("mosaic_api")
    from ml.tests.synthetic import write_physionet_tree

    datasets = base / "datasets"
    write_physionet_tree(datasets / "synthetic_release")
    os.environ.update({
        "DATABASE_URL": f"sqlite:///{(base / 'test.db').as_posix()}",
        "STORAGE_DIR": str(base / "storage"),
        "DATASETS_DIR": str(datasets),
        "REPORTS_DIR": str(base / "reports"),
        "JOBS_SYNC": "1",
        "MAX_UPLOAD_MB": "2",
        "CORS_ORIGINS": "http://testserver",
        "MOSAIC_ENV": "test",
    })
    from app.core.config import get_settings
    from app.database.session import reset_engine

    get_settings.cache_clear()
    reset_engine()
    # Create the schema with the real migration (tests the migration as well).
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(ROOT / "backend" / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "backend" / "alembic"))
    command.upgrade(cfg, "head")
    return base


@pytest.fixture(scope="session")
def client(env):
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app()) as c:
        yield c


@pytest.fixture(scope="session")
def ingested(client):
    r = client.post("/api/dataset/ingest", json={"path": "synthetic_release"})
    assert r.status_code == 202, r.text
    ds = client.get(f"/api/datasets/{r.json()['id']}").json()
    assert ds["status"] == "ready", ds
    return ds


@pytest.fixture(scope="session")
def trained(client, ingested):
    r = client.post("/api/models/train", json={"method": "robust_z_personal", "n_folds": 3})
    assert r.status_code == 202, r.text
    run = client.get(f"/api/models/training-runs/{r.json()['id']}").json()
    assert run["status"] == "completed", run
    return client.get(f"/api/models/{run['model_version_id']}").json()
