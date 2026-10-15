from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.core.errors import APIError, not_found
from app.database.models import Dataset, Participant, RecordingSession
from app.database.session import get_db
from app.schemas.schemas import DatasetOut, IngestRequest, SessionOut, ValidatePathRequest
from app.services import datasets as svc
from app.services import jobs, store
from ml.config import FeatureConfig
from ml.data_ingestion import physionet_wearable
from ml.features.extract import FEATURE_SPECS
from ml.pipeline import load_signals

router = APIRouter(prefix="/api", tags=["dataset"])

IMPORT_HELP = {
    "steps": [
        "Download the open-access PhysioNet dataset: python scripts/download_dataset.py",
        "Ingest it: POST /api/dataset/ingest {\"path\": \"raw/wearable-device-dataset-1.0.1\"} "
        "or python scripts/ingest.py --path datasets/raw/wearable-device-dataset-1.0.1",
        "Alternatively upload a long-format CSV (timestamp, participant_id, eda/hr/temp/bvp/acc_*) in the Data Explorer.",
    ],
    "dataset_url": physionet_wearable.DATASET_URL,
}


@router.get("/dataset/metadata", summary="Active dataset metadata (or an empty state with import instructions)")
def metadata(db: Session = Depends(get_db)):
    ds = svc.active_dataset(db, required=False)
    latest = db.scalars(select(Dataset).order_by(Dataset.id.desc())).first()
    if ds is None:
        return {"configured": False, "latest_attempt": DatasetOut.model_validate(latest).model_dump() if latest else None,
                "how_to_import": IMPORT_HELP}
    man = ds.manifest or {}
    sessions = db.scalars(select(RecordingSession).where(RecordingSession.dataset_id == ds.id)).all()
    modalities = sorted({m for s in sessions for m in (s.modalities or [])})
    rates = {}
    for s in sessions:
        for k, v in (s.sampling_rates or {}).items():
            rates.setdefault(k, set()).add(v)
    return {
        "configured": True, "dataset": DatasetOut.model_validate(ds).model_dump(),
        "provenance": man.get("dataset", {}), "processed_at": man.get("created_at"), "ml_version": man.get("ml_version"),
        "modalities": modalities, "sampling_rates": {k: sorted(v) for k, v in rates.items()},
        "protocols": sorted({s.protocol for s in sessions}),
        "features": [{"name": f[0], "modality": f[1], "unit": f[2], "description": f[3]} for f in FEATURE_SPECS],
        "ingestion_errors": man.get("ingestion_errors", []),
        "how_to_import": IMPORT_HELP,
    }


@router.get("/datasets", response_model=list[DatasetOut], summary="All dataset ingestion records")
def list_datasets(db: Session = Depends(get_db)):
    return db.scalars(select(Dataset).order_by(Dataset.id.desc())).all()


@router.get("/datasets/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    ds = db.get(Dataset, dataset_id)
    if ds is None:
        raise not_found("dataset", dataset_id)
    return ds


@router.get("/dataset/sources", summary="Server-side dataset folders available for ingestion")
def sources():
    return {"sources": svc.list_sources()}


@router.post("/dataset/validate", summary="Validate an uploaded CSV or ZIP without ingesting it")
async def validate_upload(file: UploadFile = File(...)):
    path = await svc.save_upload(file)
    try:
        return svc.validate_upload_file(path)
    finally:
        path.unlink(missing_ok=True)


@router.post("/dataset/validate-path", summary="Validate a server-side dataset folder")
def validate_path(req: ValidatePathRequest):
    return svc.validate_physionet_dir(svc.resolve_server_source(req.path))


@router.post("/dataset/ingest", status_code=202, response_model=DatasetOut,
             summary="Ingest a server-side PhysioNet dataset folder (background job)")
def ingest(req: IngestRequest, db: Session = Depends(get_db)):
    source = svc.resolve_server_source(req.path)
    report = svc.validate_physionet_dir(source)
    if not report["valid"]:
        raise APIError(422, "invalid_dataset", "dataset folder failed validation", report["errors"])
    cfg = FeatureConfig.from_dict(req.feature_config.model_dump())
    ds = svc.create_dataset(db, "physionet_wearable", req.path, physionet_wearable.DATASET_NAME, cfg)
    jobs.submit(svc.run_ingestion, ds.id, str(source), req.protocols)
    db.refresh(ds)
    return ds


@router.post("/dataset/upload", status_code=202, response_model=DatasetOut,
             summary="Upload and ingest a CSV (long format) or a ZIP of the PhysioNet release")
async def upload(file: UploadFile = File(...), name: str = Form("Uploaded dataset", max_length=120),
                 db: Session = Depends(get_db)):
    path = await svc.save_upload(file)
    report = svc.validate_upload_file(path)
    if not report.get("valid"):
        path.unlink(missing_ok=True)
        raise APIError(422, "invalid_dataset", "uploaded file failed validation", report.get("errors"))
    if path.suffix.lower() == ".zip":
        root = svc._find_physionet_root(path.with_suffix(""))
        ds = svc.create_dataset(db, "physionet_wearable", f"upload:{path.name}", name, FeatureConfig())
        jobs.submit(svc.run_ingestion, ds.id, str(root), None)
    else:
        ds = svc.create_dataset(db, "generic_csv", f"upload:{path.name}", name, FeatureConfig())
        jobs.submit(svc.run_ingestion, ds.id, str(path), None)
    db.refresh(ds)
    return ds


@router.get("/dataset/quality", summary="Data-quality summary per session and modality")
def quality(dataset_id: int | None = None, db: Session = Depends(get_db)):
    ds = svc.active_dataset(db, dataset_id)
    sessions = db.scalars(select(RecordingSession).options(selectinload(RecordingSession.participant))
                          .where(RecordingSession.dataset_id == ds.id).order_by(RecordingSession.session_key)).all()
    df = store.features(svc.feature_dir(ds))
    per_modality = {}
    for col in [c for c in df.columns if c.startswith("q_")]:
        m = col[2:]
        per_modality[m] = {"mean_valid_fraction": round(float(df[col].mean()), 4),
                           "windows_below_threshold": int((df[col] < (ds.feature_config or {}).get("min_valid_fraction", 0.8)).sum())}
    feature_missing = {f[0]: round(float(df[f[0]].isna().mean()), 4) for f in FEATURE_SPECS if f[0] in df}
    status_counts = pd.Series([s.quality_status for s in sessions]).value_counts().to_dict() if sessions else {}
    return {
        "dataset_id": ds.id, "n_sessions": len(sessions), "status_counts": status_counts,
        "excluded_sessions": [{"session_key": s.session_key, "reason": s.exclusion_reason} for s in sessions if s.excluded],
        "per_modality": per_modality, "feature_missing_fraction": feature_missing,
        "sessions": [{"session_key": s.session_key, "participant": s.participant.pseudonym, "protocol": s.protocol,
                      "quality_status": s.quality_status, "excluded": s.excluded,
                      "modalities": {k: {"status": v["status"], "missing_fraction": v["missing_fraction"],
                                         "out_of_range_fraction": v["out_of_range_fraction"],
                                         "flatline_fraction": v["flatline_fraction"], "issues": v["issues"]}
                                     for k, v in (s.quality or {}).items()}} for s in sessions],
    }


@router.get("/sessions", response_model=list[SessionOut], summary="Recording sessions of the active dataset")
def list_sessions(participant: str | None = Query(None, max_length=64), protocol: str | None = Query(None, max_length=30),
                  dataset_id: int | None = None, db: Session = Depends(get_db)):
    ds = svc.active_dataset(db, dataset_id)
    q = select(RecordingSession).options(selectinload(RecordingSession.participant)).where(RecordingSession.dataset_id == ds.id)
    if protocol:
        q = q.where(RecordingSession.protocol == protocol)
    if participant:
        q = q.join(Participant).where(Participant.pseudonym == participant)
    out = []
    for s in db.scalars(q.order_by(RecordingSession.session_key)).all():
        data = {c.name: getattr(s, c.name) for c in RecordingSession.__table__.columns}
        data["participant"] = s.participant.pseudonym  # pseudonym, not the ORM relationship
        out.append(SessionOut(**data))
    return out


@router.get("/sessions/{session_key}/signals", summary="Per-second signal summaries for charts")
def session_signals(session_key: str, start_s: float = Query(0, ge=0), end_s: float | None = Query(None, gt=0),
                    signals: str | None = Query(None, description="comma-separated, e.g. EDA,HR"),
                    max_points: int = Query(1500, ge=50, le=10000), dataset_id: int | None = None,
                    db: Session = Depends(get_db)):
    ds = svc.active_dataset(db, dataset_id)
    rs = db.scalars(select(RecordingSession).where(RecordingSession.dataset_id == ds.id,
                                                   RecordingSession.session_key == session_key)).first()
    if rs is None:
        raise not_found("session", session_key)
    try:
        sig = load_signals(svc.feature_dir(ds), session_key)
    except FileNotFoundError:
        raise APIError(410, "signals_missing", "processed signal file is missing; re-ingest the dataset")
    end = end_s if end_s is not None else float(sig["t_s"].max() + 1)
    if end <= start_s:
        raise APIError(422, "invalid_range", "end_s must be greater than start_s")
    sig = sig[(sig["t_s"] >= start_s) & (sig["t_s"] < end)]
    wanted = [c for c in sig.columns if c != "t_s"]
    if signals:
        req = [s.strip().upper() for s in signals.split(",") if s.strip()]
        unknown = set(req) - set(wanted)
        if unknown:
            raise APIError(422, "unknown_signal", f"session has no signal(s): {sorted(unknown)}")
        wanted = req
    step = max(int(np.ceil(len(sig) / max_points)), 1)
    if step > 1:  # block-average to keep the payload small
        grp = (np.arange(len(sig)) // step)
        sig = sig.groupby(grp).mean(numeric_only=True)
    missing = {c: round(float(sig[c].isna().mean()), 4) for c in wanted}
    sig = sig.astype(float).round(4)
    points = sig[["t_s"] + wanted].replace({np.nan: None}).to_dict(orient="records")
    return {"session_key": session_key, "protocol": rs.protocol, "origin": rs.origin, "resolution_s": step,
            "signals": wanted, "units": {"EDA": "uS", "HR": "bpm", "TEMP": "degC", "BVP": "a.u. (1 s std)", "ACC": "g"},
            "sampling_rates": rs.sampling_rates, "missing_fraction": missing, "phases": rs.phases or [],
            "calibration_end_s": rs.calibration_end_s, "points": points}


@router.get("/sessions/{session_key}/windows", summary="Window-level features and quality for a session")
def session_windows(session_key: str, dataset_id: int | None = None, db: Session = Depends(get_db)):
    ds = svc.active_dataset(db, dataset_id)
    df = store.features(svc.feature_dir(ds))
    g = df[df["session_id"] == session_key]
    if g.empty:
        raise not_found("session", session_key)
    cols = ["window_index", "start_s", "end_s", "phase", "label", "is_calibration", "is_eval"] + \
        [c for c in g.columns if c.startswith("q_")] + [f[0] for f in FEATURE_SPECS]
    return {"session_key": session_key, "windows": g[cols].replace({np.nan: None}).to_dict(orient="records")}
