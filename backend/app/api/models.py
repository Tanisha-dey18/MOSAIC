from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import APIError, not_found
from app.database.models import EvaluationResult, ModelVersion, TrainingRun
from app.database.session import get_db
from app.schemas.schemas import ModelVersionOut, Page, TrainingRunOut, TrainRequest
from app.services import datasets as dsvc
from app.services import jobs
from app.services import training as tsvc
from ml.models.detectors import DESCRIPTIONS, LIMITATIONS, parse_method

router = APIRouter(prefix="/api/models", tags=["models"])


def _run_out(run: TrainingRun) -> TrainingRunOut:
    out = TrainingRunOut.model_validate(run)
    out.model_version_id = run.model_version.id if run.model_version else None
    return out


def _mv_out(db: Session, mv: ModelVersion) -> ModelVersionOut:
    out = ModelVersionOut.model_validate(mv)
    ev = db.scalars(select(EvaluationResult).where(EvaluationResult.model_version_id == mv.id)).first()
    out.metrics = ev.metrics if ev else None
    path = get_settings().storage_dir / mv.artifact_path / "metadata.json"
    out.artifact_ok = path.exists()
    return out


@router.get("/methods", summary="Catalogue of trainable methods with descriptions and limitations")
def methods():
    out = []
    for m in tsvc.TRAINABLE:
        scorer, norm = parse_method(m)
        out.append({"method": m, "scorer": scorer, "normalization": norm, "description": DESCRIPTIONS[scorer],
                    "limitations": LIMITATIONS[scorer],
                    "analysis": "secondary (post hoc)" if scorer == "arousal_z" else "primary"})
    return {"methods": out}


@router.post("/train", status_code=202, response_model=TrainingRunOut, summary="Start a training job")
def train(req: TrainRequest, db: Session = Depends(get_db)):
    if req.method not in tsvc.TRAINABLE:
        raise APIError(422, "unknown_method", f"method must be one of {tsvc.TRAINABLE}")
    ds = dsvc.active_dataset(db, req.dataset_id)
    active = db.scalar(select(func.count(TrainingRun.id)).where(TrainingRun.status.in_(["queued", "running"])))
    if active and active >= 5:
        raise APIError(429, "too_many_jobs", "too many queued training jobs; wait for them to finish")
    run = tsvc.create_run(db, ds, req.method, req.modalities, req.params, req.threshold_quantile, req.n_folds, req.seed)
    jobs.submit(tsvc.execute_run, run.id)
    db.refresh(run)
    return _run_out(run)


@router.get("/training-runs", response_model=Page[TrainingRunOut], summary="Training history")
def training_runs(page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100), db: Session = Depends(get_db)):
    total = db.scalar(select(func.count(TrainingRun.id))) or 0
    runs = db.scalars(select(TrainingRun).order_by(TrainingRun.id.desc()).offset((page - 1) * page_size).limit(page_size)).all()
    return {"items": [_run_out(r) for r in runs], "total": total, "page": page, "page_size": page_size}


@router.get("/training-runs/{run_id}", response_model=TrainingRunOut)
def training_run(run_id: int, db: Session = Depends(get_db)):
    run = db.get(TrainingRun, run_id)
    if run is None:
        raise not_found("training run", run_id)
    return _run_out(run)


@router.get("", response_model=list[ModelVersionOut], summary="Trained model versions")
def list_models(method: str | None = Query(None, max_length=50), db: Session = Depends(get_db)):
    q = select(ModelVersion).order_by(ModelVersion.id.desc())
    if method:
        q = q.where(ModelVersion.method == method)
    return [_mv_out(db, mv) for mv in db.scalars(q).all()]


@router.get("/{model_id}", response_model=ModelVersionOut)
def get_model(model_id: int, db: Session = Depends(get_db)):
    mv = db.get(ModelVersion, model_id)
    if mv is None:
        raise not_found("model", model_id)
    return _mv_out(db, mv)


@router.get("/{model_id}/metrics", summary="Stored evaluation results of a model version")
def model_metrics(model_id: int, db: Session = Depends(get_db)):
    mv = db.get(ModelVersion, model_id)
    if mv is None:
        raise not_found("model", model_id)
    evs = db.scalars(select(EvaluationResult).where(EvaluationResult.model_version_id == mv.id)).all()
    scorer, _ = parse_method(mv.method)
    return {"model_id": mv.id, "method": mv.method, "version": mv.version, "description": DESCRIPTIONS[scorer],
            "limitations": LIMITATIONS[scorer],
            "evaluations": [{"id": e.id, "kind": e.kind, "metrics": e.metrics, "curves": e.curves,
                             "per_participant": e.per_participant, "details": e.details,
                             "created_at": e.created_at.isoformat()} for e in evs]}
