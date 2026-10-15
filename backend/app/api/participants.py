from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.core.errors import APIError, not_found
from app.core.jsonsafe import clean
from app.database.models import BaselineProfile, ModelVersion, Participant
from app.database.session import get_db
from app.services import datasets as dsvc
from app.services import store
from app.services.anomalies import load_detector
from ml.config import FeatureConfig
from ml.experiments.train import select_training_frame
from ml.features.extract import FEATURE_NAMES, FEATURE_SPECS
from ml.models.normalization import build_baselines, fit_reference, normalize

router = APIRouter(prefix="/api/participants", tags=["participants"])
UNITS = {f[0]: f[2] for f in FEATURE_SPECS}


def _num(x):
    return None if x is None or not np.isfinite(x) else float(x)


@router.get("", summary="Participants with sessions and baseline availability")
def list_participants(dataset_id: int | None = None, db: Session = Depends(get_db)):
    ds = dsvc.active_dataset(db, dataset_id)
    cfg = FeatureConfig.from_dict(ds.feature_config or {})
    df = store.features(dsvc.feature_dir(ds))
    cal = df[df["is_calibration"]].dropna(subset=FEATURE_NAMES).groupby("participant_id").size()
    parts = db.scalars(select(Participant).options(selectinload(Participant.sessions))
                       .where(Participant.dataset_id == ds.id).order_by(Participant.pseudonym)).all()
    out = []
    for p in parts:
        n_cal = int(cal.get(p.pseudonym, 0))
        any_cal = any(s.calibration_end_s is not None for s in p.sessions if not s.excluded)
        status = "available" if n_cal >= cfg.min_calibration_windows else ("insufficient" if any_cal else "no_calibration_period")
        out.append({
            "participant": p.pseudonym, "protocol_version": p.protocol_version,
            "sessions": [{"session_key": s.session_key, "protocol": s.protocol, "duration_s": s.duration_s,
                          "quality_status": s.quality_status, "excluded": s.excluded} for s in p.sessions],
            "total_minutes": round(sum(s.duration_s for s in p.sessions) / 60, 1),
            "complete_calibration_windows": n_cal, "baseline_status": status,
            "min_calibration_windows": cfg.min_calibration_windows,
        })
    return {"dataset_id": ds.id, "participants": out}


@router.get("/{participant}/baseline", summary="Personal baseline, coverage and comparison with later observations")
def baseline(participant: str, model_id: int | None = Query(None, ge=1), dataset_id: int | None = None,
             db: Session = Depends(get_db)):
    ds = dsvc.active_dataset(db, dataset_id)
    p = db.scalars(select(Participant).options(selectinload(Participant.sessions))
                   .where(Participant.dataset_id == ds.id, Participant.pseudonym == participant)).first()
    if p is None:
        raise not_found("participant", participant)
    cfg = FeatureConfig.from_dict(ds.feature_config or {})
    df = select_training_frame(store.features(dsvc.feature_dir(ds)))
    rows = df[df["participant_id"] == participant]

    mv = None
    if model_id is not None:
        mv = db.get(ModelVersion, model_id)
        if mv is None or mv.dataset_id != ds.id:
            raise not_found("model", model_id)
        if mv.normalization not in ("personal", "rolling"):
            raise APIError(422, "not_personalized", "the selected model does not use a personal baseline")
    else:
        mv = db.scalars(select(ModelVersion).where(ModelVersion.dataset_id == ds.id, ModelVersion.normalization == "personal")
                        .order_by(ModelVersion.id.desc())).first()

    if mv is not None:
        det = load_detector(mv)
        ref, features, source = det.ref, det.features, f"model {mv.method} v{mv.version} (id {mv.id})"
        stored = db.scalars(select(BaselineProfile).where(BaselineProfile.model_version_id == mv.id,
                                                          BaselineProfile.participant_id == p.id)).first()
    else:
        # Descriptive reference statistics from all participants (no model has been trained yet).
        try:
            ref = fit_reference(df, FEATURE_NAMES, cfg)
        except ValueError as exc:
            raise APIError(409, "no_calibration_data", str(exc))
        features, source, stored = FEATURE_NAMES, "descriptive (no personalised model trained yet)", None

    response = {
        "participant": participant, "protocol_version": p.protocol_version, "reference_source": source,
        "model_id": mv.id if mv else None, "stored_profile": stored is not None,
        "coverage": [{"session_key": s.session_key, "protocol": s.protocol, "duration_min": round(s.duration_s / 60, 1),
                      "calibration_end_s": s.calibration_end_s, "n_calibration_windows": s.n_calibration_windows,
                      "n_eval_windows": s.n_eval_windows, "excluded": s.excluded} for s in p.sessions],
        "construction_rule": (f"Median and shrunk MAD of the participant's complete windows in the last "
                              f"{cfg.calibration_s:.0f} s before the first stressor (window {cfg.window_s:.0f} s, hop "
                              f"{cfg.step_s:.0f} s); requires >= {cfg.min_calibration_windows} windows. Only data recorded "
                              "before the evaluation period is used."),
    }
    if rows.empty:
        response.update({"status": "insufficient", "reason": "participant has no labelled-protocol sessions"})
        return clean(response)
    b = build_baselines(rows, ref, cfg).get(participant)
    if b is None or b.status != "ok":
        response.update({"status": "insufficient", "reason": b.reason if b else "no calibration windows",
                         "warning": "Insufficient history: personalised models fall back to 'not scored' for this participant."})
        return clean(response)
    after = rows[rows["start_s"] >= rows["calibration_end_s"].fillna(np.inf)]
    recent = after[features].median(numeric_only=True)
    feats = []
    for j, f in enumerate(features):
        c, s = b.center[j], b.scale[j]
        r = _num(recent.get(f)) if f in recent else None
        feats.append({"feature": f, "unit": UNITS[f], "baseline_median": _num(c), "baseline_scale": _num(s),
                      "population_median": _num(ref.global_center[j]), "after_calibration_median": r,
                      "after_calibration_z": _num((r - c) / s) if r is not None and s else None})
    Z = normalize(rows, "personal", ref, cfg, {participant: b})
    with np.errstate(all="ignore"):
        rms = np.sqrt(np.nanmean(np.clip(Z, -20, 20) ** 2, axis=1))
    trend = [{"session_key": sid, "start_s": float(st), "phase": ph, "is_calibration": bool(ic), "deviation": _num(v)}
             for sid, st, ph, ic, v in zip(rows["session_id"], rows["start_s"], rows["phase"], rows["is_calibration"], rms)]
    response.update({
        "status": "ok", "n_windows": b.n_windows, "baseline_session": b.session_id,
        "period": {"start_s": b.period_start_s, "end_s": b.period_end_s, "start_time": b.period_start_time,
                   "end_time": b.period_end_time},
        "features": feats, "deviation_trend": trend,
        "deviation_definition": "RMS of personal robust z-scores across features (two-sided); descriptive only.",
    })
    return clean(response)
