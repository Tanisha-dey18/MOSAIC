"""Model training jobs."""
from __future__ import annotations

import logging

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import APIError
from app.core.jsonsafe import clean as _clean
from app.database.models import (
    BaselineProfile, Dataset, EvaluationResult, ModelVersion, Participant, TrainingRun, utcnow,
)
from app.database.session import session_factory
from app.services import store
from app.services.datasets import feature_dir
from ml.config import ExperimentConfig, FeatureConfig
from ml.experiments.train import select_training_frame, train_and_evaluate
from ml.features.extract import FEATURE_NAMES, features_for_modalities
from ml.models.detectors import AROUSAL_DIRECTION, parse_method
from ml.models.normalization import build_baselines

log = logging.getLogger("mosaic.training")
TRAINABLE = ["hr_rule", "robust_z_global", "robust_z_personal", "robust_z_rolling", "arousal_z_global",
             "arousal_z_personal", "arousal_z_rolling", "iforest_global", "iforest_personal", "ocsvm_global",
             "ocsvm_personal", "autoencoder_global", "autoencoder_personal"]


def create_run(db: Session, ds: Dataset, method: str, modalities: list[str] | None, params: dict | None,
               threshold_quantile: float, n_folds: int, seed: int) -> TrainingRun:
    if method not in TRAINABLE:
        raise APIError(422, "unknown_method", f"method must be one of {TRAINABLE}")
    feats = features_for_modalities(modalities) if modalities else FEATURE_NAMES
    if not feats:
        raise APIError(422, "no_features", "the selected modalities yield no features")
    if method.startswith("arousal_z") and not any(f in AROUSAL_DIRECTION for f in feats):
        raise APIError(422, "no_features", "arousal_z needs EDA, HR, BVP or TEMP features")
    if method == "hr_rule" and "hr_mean" not in feats:
        raise APIError(422, "no_features", "hr_rule needs the HR modality")
    cfg = ExperimentConfig(seed=seed, n_folds=n_folds, threshold_quantile=threshold_quantile,
                           features=FeatureConfig.from_dict(ds.feature_config or {}))
    run = TrainingRun(dataset_id=ds.id, method=method, status="queued", config=cfg.to_dict(), features=feats,
                      params=params or {})
    db.add(run)
    db.commit()
    db.refresh(run)
    return run




def execute_run(run_id: int) -> None:
    settings = get_settings()
    db = session_factory()()
    try:
        run = db.get(TrainingRun, run_id)
        run.status, run.started_at, run.progress = "running", utcnow(), "loading features"
        db.commit()
        ds = db.get(Dataset, run.dataset_id)
        df = select_training_frame(store.features(feature_dir(ds)))
        if df.empty:
            raise ValueError("the dataset has no non-excluded STRESS-protocol sessions to train on")
        cfg = ExperimentConfig.from_dict(run.config)
        version = (db.scalar(select(func.max(ModelVersion.version)).where(
            ModelVersion.method == run.method, ModelVersion.dataset_id == ds.id)) or 0) + 1
        artifact_rel = f"artifacts/{run.method}/v{version}_run{run.id}"

        def progress(msg: str) -> None:
            run.progress = msg
            db.commit()

        result = train_and_evaluate(
            df, run.method, cfg, run.features, run.params or None, artifact_dir=settings.storage_dir / artifact_rel,
            extra_metadata={"dataset": {"id": ds.id, "name": ds.name, "version": ds.version, "doi": ds.doi},
                            "training_run_id": run.id, "model_version": version},
            progress=progress)
        det = result.pop("detector")
        scorer, norm = parse_method(run.method)
        mv = ModelVersion(training_run_id=run.id, dataset_id=ds.id, method=run.method, scorer=scorer,
                          normalization=norm, version=version, artifact_path=artifact_rel,
                          feature_hash=det.metadata()["feature_hash"], features=det.features,
                          threshold=det.threshold, threshold_quantile=det.threshold_quantile, seed=det.seed)
        db.add(mv)
        db.flush()
        ev = result["evaluation"]
        db.add(EvaluationResult(model_version_id=mv.id, kind=ev["kind"], metrics=_clean(ev.get("metrics")),
                                curves=_clean(ev.get("curves")), per_participant=_clean(ev.get("per_participant_auc")),
                                details=_clean({"timing": ev.get("timing"), "folds": ev.get("folds"),
                                                "reason": ev.get("reason"),
                                                "eligible": result["eligible_participants"],
                                                "ineligible": result["ineligible_participants"],
                                                "score_distribution": result["score_distribution"],
                                                "train_info": result["train_info"]})))
        if norm in ("personal", "rolling") and det.ref is not None:
            store_baselines(db, ds, mv, det)
        run.status, run.progress, run.finished_at = "completed", "complete", utcnow()
        db.commit()
        log.info("training completed", extra={"run_id": run.id, "method": run.method, "model_version": mv.id})
    except Exception as exc:
        db.rollback()
        run = db.get(TrainingRun, run_id)
        run.status, run.error, run.finished_at = "failed", f"{type(exc).__name__}: {exc}"[:2000], utcnow()
        db.commit()
        log.warning("training failed", extra={"run_id": run_id, "error": type(exc).__name__})
    finally:
        db.close()


def store_baselines(db: Session, ds: Dataset, mv: ModelVersion, det) -> None:
    """Persist each participant's personal baseline (calibration period, statistics) for this model."""
    df = select_training_frame(store.features(feature_dir(ds)))
    baselines = build_baselines(df, det.ref, det.feature_cfg)
    pmap = {p.pseudonym: p.id for p in db.scalars(select(Participant).where(Participant.dataset_id == ds.id))}
    for pid, b in baselines.items():
        if pid not in pmap:
            continue
        db.add(BaselineProfile(model_version_id=mv.id, participant_id=pmap[pid], status=b.status, n_windows=b.n_windows,
                               session_key=b.session_id, period_start_s=b.period_start_s, period_end_s=b.period_end_s,
                               period_start_time=b.period_start_time, period_end_time=b.period_end_time,
                               center=_clean(b.center), scale=_clean(b.scale), reason=b.reason))
