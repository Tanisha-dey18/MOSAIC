"""MOSAIC API.

Run (from the repository root):
    uvicorn app.main:app --app-dir backend --reload
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Make the top-level `ml` package importable when started from backend/.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from app.api import anomalies, dataset, health, models, participants, reports  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.errors import install_handlers  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402
from ml import __version__  # noqa: E402


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.ensure_dirs()
    if settings.bootstrap:
        from app.services import jobs
        from app.services.bootstrap import run_bootstrap

        jobs.submit(run_bootstrap)  # background thread: health answers while data is prepared
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="MOSAIC API",
        version=__version__,
        description=("Multimodal Observation System for Anomaly Identification and Clinical Trends. "
                     "Research software for offline analysis of wearable physiological data. "
                     "Not a medical device; not for diagnosis."),
        lifespan=lifespan,
    )
    # Outside production, any local dev-server port is allowed (Vite picks the next free port
    # when 5173 is busy). Production uses only the explicit CORS_ORIGINS allow-list.
    local_regex = None if settings.mosaic_env == "production" else r"^http://(localhost|127\.0\.0\.1)(:\d+)?$"
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list, allow_origin_regex=local_regex,
                       allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
    install_handlers(app)
    for r in (health.router, dataset.router, participants.router, models.router, anomalies.router, reports.router):
        app.include_router(r)
    return app


app = create_app()
