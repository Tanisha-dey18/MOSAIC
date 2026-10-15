from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.database.models import Dataset, ModelVersion
from app.database.session import check_database, get_db
from app.schemas.schemas import HealthOut
from ml import __version__

router = APIRouter(tags=["health"])


@router.get("/api/health", response_model=HealthOut, summary="Service, database, storage and model readiness")
def health(db: Session = Depends(get_db)):
    settings = get_settings()
    db_ok, db_err = check_database()
    storage_ok = settings.storage_dir.exists() and settings.storage_dir.is_dir()
    dataset = {"configured": False}
    models = {"count": 0}
    if db_ok:
        ds = db.scalars(select(Dataset).where(Dataset.status == "ready").order_by(Dataset.id.desc())).first()
        if ds:
            dataset = {"configured": True, "id": ds.id, "name": ds.name, "version": ds.version}
        models = {"count": int(db.scalar(select(func.count(ModelVersion.id))) or 0)}
    return {
        "status": "ok" if db_ok and storage_ok else "degraded",
        "version": __version__, "environment": settings.mosaic_env,
        "database": {"ok": db_ok, "error": db_err}, "storage": {"ok": storage_ok},
        "dataset": dataset, "models": models,
    }
