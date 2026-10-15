"""Offline (retrospective) anomaly detection and explanations.

MOSAIC analyses recorded sessions after the fact. It does not perform
real-time monitoring; responses label results as "offline_retrospective".
"""
from __future__ import annotations

import logging
import math
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import APIError, model_not_ready, not_found
from app.database.models import AnomalyEvent, Dataset, DetectionRun, ModelVersion, Participant, RecordingSession
from app.services import store
from app.services.datasets import feature_dir
from ml.explainability.contributions import CAVEATS, METHOD_NOTES, feature_table
from ml.models.detectors import ArtifactError, Detector
from ml.models.normalization import shrunk_scale, build_baselines, robust_center_scale

log = logging.getLogger("mosaic.anomalies")
_cache: dict[str, Detector] = {}
_lock = threading.Lock()
def load_detector(mv: ModelVersion) -> Detector:
    path = get_settings().storage_dir / mv.artifact_path
    key = str(path)
    with _lock:
        if key in _cache:
            return _cache[key]
    try:
        det = Detector.load(path)
    except ArtifactError as exc:
        raise model_not_ready(f"model {mv.id} ({mv.method} v{mv.version}) cannot be loaded: {exc}") from exc
    if det.metadata()["feature_hash"] != mv.feature_hash:
        raise model_not_ready("artifact feature schema does not match the registered model version")
    with _lock:
        _cache[key] = det
    return det


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def _get_model(db: Session, model_id: int) -> ModelVersion:
    mv = db.get(ModelVersion, model_id)
    if mv is None:
        raise not_found("model", model_id)
    return mv


def _session(db: Session, ds_id: int, session_key: str) -> RecordingSession:
    rs = db.scalars(select(RecordingSession).where(RecordingSession.dataset_id == ds_id,
                                                   RecordingSession.session_key == session_key)).first()
    if rs is None:
        raise not_found("session", session_key)
    return rs


def _participant_rows(db: Session, mv: ModelVersion, rs: RecordingSession) -> tuple[pd.DataFrame, pd.Series]:
    ds = db.get(Dataset, mv.dataset_id)
    df = store.features(feature_dir(ds))
    pseudonym = db.get(Participant, rs.participant_id).pseudonym
    rows = df[df["participant_id"] == pseudonym]
    return rows, rows["session_id"] == rs.session_key


def _num(x):
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) or math.isinf(x) else x


def _baseline_info(det: Detector, rows: pd.DataFrame, rs: RecordingSession) -> dict:
    if det.normalization == "personal":
        b = build_baselines(rows, det.ref, det.feature_cfg).get(str(rows["participant_id"].iloc[0]))
        if b is None or b.status != "ok":
            return {"type": "personal", "status": "insufficient", "reason": b.reason if b else "no calibration data"}
        info = {"type": "personal", "status": "ok", "session": b.session_id, "n_windows": b.n_windows,
                "period_start_s": b.period_start_s, "period_end_s": b.period_end_s,
                "period_start_time": b.period_start_time, "period_end_time": b.period_end_time}
        if b.session_id != rs.session_key:
            info["warning"] = (f"baseline comes from a different recording ({b.session_id}); recordings of different "
                               "protocols may have been made on different days")
        return info
    if det.normalization == "rolling":
        return {"type": "rolling", "status": "ok",
                "description": "median of the participant's windows ending within the previous 300 s (strictly past)"}
    if det.normalization == "global":
        return {"type": "global", "status": "ok",
                "description": f"population median/scale of {det.train_info.get('n_train_participants', '?')} training participants"}
    return {"type": "none", "status": "ok", "description": "fixed rule, no learned reference"}


def detect(db: Session, model_id: int, session_key: str, threshold: float | None = None) -> dict:
    mv = _get_model(db, model_id)
    det = load_detector(mv)
    rs = _session(db, mv.dataset_id, session_key)
    rows, in_session = _participant_rows(db, mv, rs)
    if not in_session.any():
        raise APIError(409, "no_windows", f"session {session_key} has no feature windows")
    thr = float(threshold) if threshold is not None else det.threshold
    if thr is None:
        raise model_not_ready("model has no calibrated threshold; provide one explicitly")
    scores = det.score(rows)[in_session.to_numpy()]
    contrib = det.contributions(rows)
    contrib = contrib[in_session.to_numpy()] if contrib is not None else None
    sess = rows[in_session]
    flagged = np.isfinite(scores) & (scores > thr)

    timeline = pd.DataFrame({
        "window_index": sess["window_index"].to_numpy(), "start_s": sess["start_s"].to_numpy(),
        "end_s": sess["end_s"].to_numpy(), "start_time": sess["start_time"].to_numpy(),
        "phase": sess["phase"].to_numpy(), "label": sess["label"].to_numpy(),
        "is_calibration": sess["is_calibration"].to_numpy(), "score": scores, "flagged": flagged,
    })
    for m in det.feature_cfg.modalities:
        if f"q_{m}" in sess:
            timeline[f"q_{m}"] = sess[f"q_{m}"].to_numpy()
    out_dir = get_settings().detections_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run = DetectionRun(model_version_id=mv.id, session_id=rs.id, threshold=thr, n_windows=int(len(sess)),
                       n_scored=int(np.isfinite(scores).sum()), n_flagged=int(flagged.sum()), result_path="")
    db.add(run)
    db.flush()
    result_path = out_dir / f"detection_{run.id}.parquet"
    timeline.to_parquet(result_path, index=False)
    run.result_path = str(Path("detections") / result_path.name)

    for k in np.flatnonzero(flagged):
        r = sess.iloc[k]
        c = {f: _num(contrib[k, j]) for j, f in enumerate(det.features)} if contrib is not None else None
        db.add(AnomalyEvent(
            detection_run_id=run.id, model_version_id=mv.id, session_id=rs.id, participant_id=rs.participant_id,
            window_index=int(r["window_index"]), start_s=float(r["start_s"]), end_s=float(r["end_s"]),
            start_time=r["start_time"], score=float(scores[k]), threshold=thr, phase=r["phase"],
            quality={m: _num(r.get(f"q_{m}")) for m in det.feature_cfg.modalities},
            features={f: _num(r[f]) for f in det.features}, contributions=c))
    db.commit()
    log.info("detection run", extra={"detection_id": run.id, "model_id": mv.id, "n_flagged": run.n_flagged})
    return detection_payload(db, run, timeline, det, rows, rs)


def detection_payload(db: Session, run: DetectionRun, timeline: pd.DataFrame, det: Detector, rows: pd.DataFrame,
                      rs: RecordingSession) -> dict:
    mv = db.get(ModelVersion, run.model_version_id)
    finite = timeline["score"].replace([np.inf, -np.inf], np.nan).dropna()
    records = timeline.replace({np.nan: None}).to_dict(orient="records")
    for rec in records:
        rec["flagged"] = bool(rec["flagged"])
        rec["is_calibration"] = bool(rec["is_calibration"])
    events = db.scalars(select(AnomalyEvent.id, ).where(AnomalyEvent.detection_run_id == run.id)).all()
    return {
        "detection_id": run.id, "mode": run.mode,
        "mode_note": "Offline retrospective analysis of a recorded session; not real-time monitoring.",
        "model": {"id": mv.id, "method": mv.method, "version": mv.version, "threshold": mv.threshold},
        "session": {"session_key": rs.session_key, "protocol": rs.protocol, "quality_status": rs.quality_status,
                    "excluded": rs.excluded, "exclusion_reason": rs.exclusion_reason, "phases": rs.phases},
        "threshold": run.threshold, "n_windows": run.n_windows, "n_scored": run.n_scored, "n_flagged": run.n_flagged,
        "score_summary": {"median": _num(finite.median()) if len(finite) else None,
                          "p95": _num(finite.quantile(0.95)) if len(finite) else None,
                          "max": _num(finite.max()) if len(finite) else None},
        "reference": _baseline_info(det, rows, rs),
        "timeline": records, "event_ids": list(events), "created_at": run.created_at.isoformat(),
    }


def get_detection(db: Session, detection_id: int) -> dict:
    run = db.get(DetectionRun, detection_id)
    if run is None:
        raise not_found("detection", detection_id)
    mv = db.get(ModelVersion, run.model_version_id)
    det = load_detector(mv)
    rs = db.get(RecordingSession, run.session_id)
    path = get_settings().storage_dir / run.result_path
    if not path.exists():
        raise APIError(410, "result_missing", "stored detection result file is missing")
    rows, _ = _participant_rows(db, mv, rs)
    return detection_payload(db, run, pd.read_parquet(path), det, rows, rs)


def explain_window(db: Session, model_id: int, session_key: str, window_index: int) -> dict:
    mv = _get_model(db, model_id)
    det = load_detector(mv)
    rs = _session(db, mv.dataset_id, session_key)
    rows, in_session = _participant_rows(db, mv, rs)
    sel = in_session & (rows["window_index"] == window_index)
    if not sel.any():
        raise not_found("window", window_index)
    pos = int(np.flatnonzero(sel.to_numpy())[0])
    score = float(det.score(rows)[pos]) if det.scorer != "hr_rule" else float(rows.iloc[pos]["hr_mean"])
    contrib = det.contributions(rows)
    row = rows.iloc[pos]

    # Reference values for the comparison "observation vs baseline".
    center = scale = None
    if det.normalization == "personal":
        b = build_baselines(rows, det.ref, det.feature_cfg).get(str(row["participant_id"]))
        if b is not None and b.status == "ok":
            center, scale = np.asarray(b.center), np.asarray(b.scale)
    elif det.normalization == "global":
        center, scale = np.asarray(det.ref.global_center), np.asarray(det.ref.global_scale)
    elif det.normalization == "rolling":
        sess = rows[in_session]
        past = sess[(sess["end_s"] <= row["start_s"]) & (sess["end_s"] > row["start_s"] - 300)][det.features].dropna()
        if len(past) >= det.feature_cfg.min_calibration_windows // 2:
            center, mad = robust_center_scale(past.to_numpy(float))
            scale = shrunk_scale(mad, len(past), det.ref, det.feature_cfg, 5.0)

    feats = feature_table(det.features, row[det.features].to_numpy(float),
                          contrib[pos] if contrib is not None else None, center, scale)
    method_note = METHOD_NOTES[det.scorer]
    return {
        "model": {"id": mv.id, "method": mv.method, "version": mv.version, "scorer": det.scorer,
                  "normalization": det.normalization, "limitations": det.metadata()["limitations"]},
        "session_key": session_key, "window_index": int(window_index), "start_s": float(row["start_s"]),
        "end_s": float(row["end_s"]), "start_time": row["start_time"], "phase": row["phase"],
        "label": _num(row["label"]), "score": _num(score), "threshold": det.threshold,
        "flagged": bool(np.isfinite(score) and det.threshold is not None and score > det.threshold),
        "quality": {m: _num(row.get(f"q_{m}")) for m in det.feature_cfg.modalities},
        "reference": _baseline_info(det, rows, rs), "explanation_method": method_note,
        "contributions_supported": contrib is not None, "features": feats, "caveats": CAVEATS,
    }
