from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.errors import not_found
from app.core.jsonsafe import clean
from app.database.models import AnomalyEvent, DetectionRun, ModelVersion, Participant, RecordingSession
from app.database.session import get_db
from app.schemas.schemas import AnomalyEventOut, DetectRequest, ExplainRequest, Page
from app.services import anomalies as svc

router = APIRouter(prefix="/api", tags=["anomalies"])


@router.post("/anomalies/detect", summary="Score every window of a recorded session (offline analysis)")
def detect(req: DetectRequest, db: Session = Depends(get_db)):
    return clean(svc.detect(db, req.model_id, req.session_key, req.threshold))


@router.get("/anomalies", response_model=Page[AnomalyEventOut], summary="Detected anomalous windows (paginated)")
def list_events(model_id: int | None = Query(None, ge=1), participant: str | None = Query(None, max_length=64),
                session_key: str | None = Query(None, max_length=100), detection_id: int | None = Query(None, ge=1),
                page: int = Query(1, ge=1), page_size: int = Query(25, ge=1, le=200), db: Session = Depends(get_db)):
    q = (select(AnomalyEvent, RecordingSession.session_key, Participant.pseudonym)
         .join(RecordingSession, AnomalyEvent.session_id == RecordingSession.id)
         .join(Participant, AnomalyEvent.participant_id == Participant.id))
    if model_id:
        q = q.where(AnomalyEvent.model_version_id == model_id)
    if participant:
        q = q.where(Participant.pseudonym == participant)
    if session_key:
        q = q.where(RecordingSession.session_key == session_key)
    if detection_id:
        q = q.where(AnomalyEvent.detection_run_id == detection_id)
    total = db.scalar(select(func.count()).select_from(q.subquery())) or 0
    rows = db.execute(q.order_by(AnomalyEvent.id.desc()).offset((page - 1) * page_size).limit(page_size)).all()
    items = []
    for ev, key, pseudo in rows:
        out = AnomalyEventOut.model_validate(ev)
        out.session_key, out.participant = key, pseudo
        items.append(out)
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.get("/anomalies/detections", summary="Detection history")
def detections(page: int = Query(1, ge=1), page_size: int = Query(25, ge=1, le=100), db: Session = Depends(get_db)):
    total = db.scalar(select(func.count(DetectionRun.id))) or 0
    rows = db.execute(select(DetectionRun, RecordingSession.session_key, ModelVersion.method, ModelVersion.version)
                      .join(RecordingSession, DetectionRun.session_id == RecordingSession.id)
                      .join(ModelVersion, DetectionRun.model_version_id == ModelVersion.id)
                      .order_by(DetectionRun.id.desc()).offset((page - 1) * page_size).limit(page_size)).all()
    return {"items": [{"id": r.id, "session_key": k, "model_id": r.model_version_id, "method": m, "version": v,
                       "threshold": r.threshold, "n_windows": r.n_windows, "n_scored": r.n_scored,
                       "n_flagged": r.n_flagged, "mode": r.mode, "created_at": r.created_at.isoformat()}
                      for r, k, m, v in rows], "total": total, "page": page, "page_size": page_size}


@router.get("/anomalies/detections/{detection_id}", summary="A stored detection timeline")
def detection(detection_id: int, db: Session = Depends(get_db)):
    return clean(svc.get_detection(db, detection_id))


@router.get("/anomalies/{anomaly_id}", summary="One anomalous window with its explanation")
def get_event(anomaly_id: int, db: Session = Depends(get_db)):
    ev = db.get(AnomalyEvent, anomaly_id)
    if ev is None:
        raise not_found("anomaly", anomaly_id)
    rs = db.get(RecordingSession, ev.session_id)
    explanation = svc.explain_window(db, ev.model_version_id, rs.session_key, ev.window_index)
    return clean({"id": ev.id, "detection_id": ev.detection_run_id, "session_key": rs.session_key,
            "participant": db.get(Participant, ev.participant_id).pseudonym, "score": ev.score,
            "threshold": ev.threshold, "start_s": ev.start_s, "end_s": ev.end_s, "start_time": ev.start_time,
            "phase": ev.phase, "quality": ev.quality, "explanation": explanation})


@router.post("/explain", summary="Explain the score of any window (flagged or not)")
def explain(req: ExplainRequest, db: Session = Depends(get_db)):
    return clean(svc.explain_window(db, req.model_id, req.session_key, req.window_index))
