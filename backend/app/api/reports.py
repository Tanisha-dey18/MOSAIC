from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import APIError, not_found
from app.database.models import ResearchReport
from app.database.session import get_db
from app.schemas.schemas import ExperimentLaunch, Page, ReportOut, ReportRequest
from app.services import datasets as dsvc
from app.services import experiments as esvc
from app.services import reports as rsvc

router = APIRouter(prefix="/api", tags=["reports & experiments"])


@router.post("/reports", status_code=201, response_model=ReportOut, summary="Generate an HTML research report")
def create_report(req: ReportRequest, db: Session = Depends(get_db)):
    return rsvc.generate(db, req.title, req.experiment_run_id, req.include_models)


@router.get("/reports", response_model=Page[ReportOut], summary="Generated reports")
def list_reports(page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100), db: Session = Depends(get_db)):
    total = db.scalar(select(func.count(ResearchReport.id))) or 0
    items = db.scalars(select(ResearchReport).order_by(ResearchReport.id.desc())
                       .offset((page - 1) * page_size).limit(page_size)).all()
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.get("/reports/{report_id}/download", summary="Download a generated report")
def download(report_id: int, db: Session = Depends(get_db)):
    rep = db.get(ResearchReport, report_id)
    if rep is None:
        raise not_found("report", report_id)
    storage = get_settings().storage_dir.resolve()
    path = (storage / rep.file_path).resolve()
    if storage not in path.parents or not path.exists():
        raise APIError(410, "report_missing", "report file is no longer available")
    return FileResponse(path, media_type="text/html", filename=f"mosaic_report_{rep.id}.html")


@router.get("/experiments", summary="Experiment runs (results produced by the research pipeline)")
def list_experiments():
    return {"runs": esvc.list_runs()}


@router.get("/experiments/{run_id}", summary="Full results of an experiment run")
def get_experiment(run_id: str):
    return esvc.get_run(run_id)


@router.get("/experiments/{run_id}/figures/{name}", summary="A figure (PNG) of an experiment run")
def get_figure(run_id: str, name: str):
    return FileResponse(esvc.figure_path(run_id, name), media_type="image/png")


@router.post("/experiments/run", status_code=202, summary="Launch the full experiment suite in a separate process")
def run_experiments(req: ExperimentLaunch, db: Session = Depends(get_db)):
    ds = dsvc.active_dataset(db)
    if ds.source_type != "physionet_wearable":
        raise APIError(422, "unsupported_dataset", "the experiment suite requires the labelled PhysioNet STRESS protocol")
    return esvc.launch_run(dsvc.feature_dir(ds), req.quick)
