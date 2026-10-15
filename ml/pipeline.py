"""Build and read the processed feature store.

Layout of a processed dataset directory::

    <out>/manifest.json            provenance: dataset, version, config, file checksums, counts
    <out>/sessions.json            per-session metadata, quality report, phases, notes
    <out>/features.parquet         one row per window (metadata + quality + features)
    <out>/signals/<session>.parquet 1 Hz summaries of each signal (for charts only)
    <out>/stress_levels.csv        self-reported stress (PhysioNet dataset only)

High-frequency raw signals stay in their original files; PostgreSQL stores only
metadata (see backend/app/database/models.py).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml import __version__ as ML_VERSION
from ml.config import FeatureConfig
from ml.data_ingestion import generic_csv, physionet_wearable
from ml.data_ingestion.session import Session
from ml.features.extract import FEATURE_NAMES, downsample_signals, extract_session_features
from ml.preprocessing.quality import assess_signal

log = logging.getLogger(__name__)


def session_report(session: Session) -> dict:
    quality = {}
    for name, sig in session.signals.items():
        q = assess_signal(name, sig.values, sig.fs, sig.issues).to_dict()
        if name in session.invalid_modalities:
            q["status"] = "unusable"
            q["issues"] = q["issues"] + [session.invalid_modalities[name]]
        quality[name] = q
    statuses = [q["status"] for q in quality.values()]
    overall = "unusable" if all(s == "unusable" for s in statuses) else (
        "degraded" if any(s != "good" for s in statuses) else "good")
    return {
        "session_id": session.session_id, "participant_id": session.participant_id,
        "protocol": session.protocol, "protocol_version": session.protocol_version,
        "origin": session.origin.isoformat() if session.origin is not None else None,
        "duration_s": round(session.duration_s, 2), "modalities": session.modalities,
        "sampling_rates": {n: s.fs for n, s in session.signals.items()},
        "tags_s": [round(t, 3) for t in session.tags_s], "phases": [p.to_dict() for p in session.phases],
        "calibration_end_s": session.calibration_end_s, "excluded": session.excluded,
        "exclusion_reason": session.exclusion_reason, "invalid_modalities": session.invalid_modalities,
        "notes": session.notes, "quality": quality, "quality_status": overall,
        "source_files": session.source_files,
    }


def build_feature_store(sessions: list[Session], out_dir: str | Path, cfg: FeatureConfig, dataset_meta: dict,
                        errors: list[str] | None = None, progress=None) -> dict:
    out = Path(out_dir)
    (out / "signals").mkdir(parents=True, exist_ok=True)
    frames, reports = [], []
    for i, s in enumerate(sessions):
        feats = extract_session_features(s, cfg)
        frames.append(feats)
        downsample_signals(s).to_parquet(out / "signals" / f"{_safe(s.session_id)}.parquet", index=False)
        rep = session_report(s)
        rep["n_windows"] = int(len(feats))
        rep["n_calibration_windows"] = int(feats["is_calibration"].sum())
        rep["n_eval_windows"] = int(feats["is_eval"].sum())
        reports.append(rep)
        if progress:
            progress(i + 1, len(sessions))
    features = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    features.to_parquet(out / "features.parquet", index=False)
    (out / "sessions.json").write_text(json.dumps(reports, indent=1, default=_default), encoding="utf-8")
    manifest = {
        "dataset": dataset_meta, "ml_version": ML_VERSION, "feature_config": cfg.to_dict(),
        "feature_names": FEATURE_NAMES, "created_at": datetime.now(timezone.utc).isoformat(),
        "n_sessions": len(sessions), "n_participants": len({s.participant_id for s in sessions}),
        "n_windows": int(len(features)), "ingestion_errors": errors or [],
        "protocols": sorted({s.protocol for s in sessions}),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=_default), encoding="utf-8")
    return manifest


def ingest_physionet(root: str | Path, out_dir: str | Path, cfg: FeatureConfig | None = None,
                     protocols: tuple[str, ...] = physionet_wearable.PROTOCOLS, progress=None) -> dict:
    cfg = cfg or FeatureConfig()
    sessions, errors = physionet_wearable.load_dataset(root, protocols)
    if not sessions:
        raise ValueError("no sessions could be loaded from " + str(root))
    meta = physionet_wearable.metadata()
    meta["source_path"] = Path(root).name
    manifest = build_feature_store(sessions, out_dir, cfg, meta, errors, progress)
    physionet_wearable.load_stress_levels(root).to_csv(Path(out_dir) / "stress_levels.csv", index=False)
    return manifest


def ingest_generic_csv(path: str | Path, out_dir: str | Path, cfg: FeatureConfig | None = None, name: str = "upload") -> dict:
    cfg = cfg or FeatureConfig()
    sessions = generic_csv.load_csv(path, protocol="UPLOAD")
    meta = {"name": name, "slug": "generic-csv", "version": "user-upload", "license": "unknown (user supplied)",
            "source_path": Path(path).name}
    return build_feature_store(sessions, out_dir, cfg, meta)


def load_features(out_dir: str | Path) -> pd.DataFrame:
    path = Path(out_dir) / "features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"no processed features at {path}")
    return pd.read_parquet(path)


def load_sessions(out_dir: str | Path) -> list[dict]:
    return json.loads((Path(out_dir) / "sessions.json").read_text(encoding="utf-8"))


def load_signals(out_dir: str | Path, session_id: str) -> pd.DataFrame:
    path = Path(out_dir) / "signals" / f"{_safe(session_id)}.parquet"
    if not path.exists():
        raise FileNotFoundError(session_id)
    return pd.read_parquet(path)


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and np.isnan(o):
        return None
    return str(o)
