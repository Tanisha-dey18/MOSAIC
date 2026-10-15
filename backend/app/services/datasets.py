"""Dataset validation, ingestion and lookup."""
from __future__ import annotations

import logging
import shutil
import uuid
import zipfile
from pathlib import Path

import pandas as pd
from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import APIError, no_dataset, not_found
from app.core.paths import safe_join
from app.database.models import Dataset, DatasetFile, Participant, RecordingSession, utcnow
from app.database.session import session_factory
from ml.config import FeatureConfig
from ml.data_ingestion import generic_csv, physionet_wearable
from ml.data_ingestion.empatica import IngestionError
from ml.pipeline import ingest_generic_csv, ingest_physionet, load_sessions

log = logging.getLogger("mosaic.datasets")
ALLOWED_UPLOADS = {".csv", ".zip"}


# ---------------------------------------------------------------- lookup
def active_dataset(db: Session, dataset_id: int | None = None, required: bool = True) -> Dataset | None:
    if dataset_id is not None:
        ds = db.get(Dataset, dataset_id)
        if ds is None:
            raise not_found("dataset", dataset_id)
        if ds.status != "ready":
            raise APIError(409, "dataset_not_ready", f"dataset {dataset_id} is '{ds.status}'")
        return ds
    ds = db.scalars(select(Dataset).where(Dataset.status == "ready").order_by(Dataset.created_at.desc(), Dataset.id.desc())).first()
    if ds is None and required:
        raise no_dataset()
    return ds


def feature_dir(ds: Dataset) -> Path:
    if not ds.storage_path:
        raise APIError(409, "dataset_not_ready", "dataset has no processed features")
    return get_settings().storage_dir / ds.storage_path


def list_sources() -> list[dict]:
    """Server-side dataset folders that look like the PhysioNet release (for the UI picker)."""
    root = get_settings().datasets_dir
    out = []
    if not root.exists():
        return out
    for candidate in sorted(root.rglob("Wearable_Dataset")):
        if candidate.is_dir():
            rel = candidate.parent.relative_to(root).as_posix()
            out.append({"path": rel, "type": "physionet_wearable",
                        "protocols": sorted(p.name for p in candidate.iterdir() if p.is_dir())})
    return out


# ------------------------------------------------------------ validation
def validate_physionet_dir(path: Path) -> dict:
    report = {"type": "physionet_wearable", "valid": False, "errors": [], "warnings": [], "sessions": {}}
    try:
        root = physionet_wearable.resolve_root(path)
    except IngestionError as exc:
        report["errors"].append(str(exc))
        return report
    found = physionet_wearable.discover(root)
    if not found:
        report["errors"].append("no participant folders found under Wearable_Dataset/{STRESS,AEROBIC,ANAEROBIC}")
        return report
    for protocol, folder in found:
        missing = [f"{n}.csv" for n in ("EDA", "BVP", "HR", "TEMP", "ACC") if not (folder / f"{n}.csv").exists()]
        report["sessions"].setdefault(protocol, 0)
        report["sessions"][protocol] += 1
        if missing:
            report["warnings"].append(f"{protocol}/{folder.name}: missing {', '.join(missing)}")
    if not (root / "SHA256SUMS.txt").exists():
        report["warnings"].append("SHA256SUMS.txt not found; file integrity cannot be verified")
    report["valid"] = True
    report["participants"] = len({physionet_wearable.participant_from_folder(f.name) for _, f in found})
    return report


def validate_csv_file(path: Path) -> dict:
    try:
        df = pd.read_csv(path, low_memory=False, nrows=5_000_000)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as exc:
        return {"type": "generic_csv", "valid": False, "errors": [f"cannot parse CSV: {exc}"], "warnings": []}
    rep = generic_csv.validate_frame(df)
    rep["type"] = "generic_csv"
    return rep


async def save_upload(upload: UploadFile) -> Path:
    """Stream an upload to disk with size limit and a server-generated file name."""
    settings = get_settings()
    original = Path(upload.filename or "")
    ext = original.suffix.lower()
    if ext not in ALLOWED_UPLOADS:
        raise APIError(415, "unsupported_file_type", f"only {', '.join(sorted(ALLOWED_UPLOADS))} files are accepted")
    limit = settings.max_upload_mb * 1024 * 1024
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    dest = settings.uploads_dir / f"{uuid.uuid4().hex}{ext}"
    size = 0
    with dest.open("wb") as fh:
        while chunk := await upload.read(1 << 20):
            size += len(chunk)
            if size > limit:
                fh.close()
                dest.unlink(missing_ok=True)
                raise APIError(413, "payload_too_large", f"file exceeds the {settings.max_upload_mb} MB limit")
            fh.write(chunk)
    if size == 0:
        dest.unlink(missing_ok=True)
        raise APIError(400, "empty_file", "uploaded file is empty")
    return dest


def safe_extract_zip(zip_path: Path, dest: Path) -> Path:
    """Extract with zip-slip and zip-bomb protection."""
    limit = get_settings().max_unzipped_mb * 1024 * 1024
    try:
        with zipfile.ZipFile(zip_path) as zf:
            total = sum(i.file_size for i in zf.infolist())
            if total > limit:
                raise APIError(413, "archive_too_large", "uncompressed archive exceeds the configured limit")
            dest.mkdir(parents=True, exist_ok=True)
            root = dest.resolve()
            for info in zf.infolist():
                target = (dest / info.filename).resolve()
                if root not in target.parents and target != root:
                    raise APIError(400, "unsafe_archive", "archive contains paths outside the extraction folder")
                if info.is_dir():
                    continue
                if not info.filename.lower().endswith((".csv", ".txt")):
                    continue  # only data/text files are extracted
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    except zipfile.BadZipFile as exc:
        raise APIError(400, "corrupted_file", "file is not a valid ZIP archive") from exc
    return dest


def validate_upload_file(path: Path) -> dict:
    if path.suffix.lower() == ".csv":
        return validate_csv_file(path)
    extract_dir = path.with_suffix("")
    safe_extract_zip(path, extract_dir)
    try:
        return validate_physionet_dir(_find_physionet_root(extract_dir))
    except APIError:
        return {"type": "zip", "valid": False, "errors": ["ZIP does not contain a Wearable_Dataset folder"], "warnings": []}


def _find_physionet_root(base: Path) -> Path:
    for candidate in base.rglob("Wearable_Dataset"):
        if candidate.is_dir():
            return candidate.parent
    raise APIError(400, "invalid_dataset", "no Wearable_Dataset folder found")


# ------------------------------------------------------------- ingestion
def create_dataset(db: Session, source_type: str, source_path: str, name: str, feature_cfg: FeatureConfig) -> Dataset:
    meta = physionet_wearable.metadata() if source_type == "physionet_wearable" else {}
    ds = Dataset(name=meta.get("name", name), slug=meta.get("slug", "generic-csv"),
                 version=meta.get("version", "user-upload"), doi=meta.get("doi"), license=meta.get("license"),
                 source_type=source_type, source_path=source_path, status="pending",
                 feature_config=feature_cfg.to_dict())
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return ds


def run_ingestion(dataset_id: int, absolute_source: str, protocols: list[str] | None = None) -> None:
    """Background job: build the feature store and register metadata in the DB."""
    settings = get_settings()
    db = session_factory()()
    try:
        ds = db.get(Dataset, dataset_id)
        ds.status, ds.progress = "processing", "reading files"
        db.commit()
        out_rel = f"processed/dataset_{ds.id}"
        out = settings.storage_dir / out_rel
        cfg = FeatureConfig.from_dict(ds.feature_config or {})

        def progress(i, n):
            ds.progress = f"processed {i}/{n} sessions"
            db.commit()

        if ds.source_type == "physionet_wearable":
            manifest = ingest_physionet(absolute_source, out, cfg,
                                        tuple(protocols) if protocols else physionet_wearable.PROTOCOLS, progress)
        else:
            manifest = ingest_generic_csv(absolute_source, out, cfg, name=ds.name)
        register_sessions(db, ds, out)
        ds.storage_path, ds.manifest = out_rel, manifest
        ds.n_sessions, ds.n_participants, ds.n_windows = manifest["n_sessions"], manifest["n_participants"], manifest["n_windows"]
        ds.status, ds.progress, ds.updated_at = "ready", "complete", utcnow()
        db.commit()
        log.info("dataset ingested", extra={"dataset_id": ds.id, "sessions": ds.n_sessions})
    except Exception as exc:
        db.rollback()
        ds = db.get(Dataset, dataset_id)
        if ds is not None:
            ds.status, ds.error = "failed", f"{type(exc).__name__}: {exc}"[:2000]
            db.commit()
        log.warning("dataset ingestion failed", extra={"dataset_id": dataset_id, "error": type(exc).__name__})
    finally:
        db.close()


def register_sessions(db: Session, ds: Dataset, out: Path) -> None:
    reports = load_sessions(out)
    participants: dict[str, Participant] = {}
    for rep in reports:
        pid = rep["participant_id"]
        if pid not in participants:
            participants[pid] = Participant(dataset_id=ds.id, pseudonym=pid, protocol_version=rep.get("protocol_version"),
                                            n_sessions=0)
            db.add(participants[pid])
        p = participants[pid]
        p.n_sessions += 1
        db.flush()
        db.add(RecordingSession(
            dataset_id=ds.id, participant_id=p.id, session_key=rep["session_id"], protocol=rep["protocol"],
            duration_s=rep["duration_s"], origin=rep.get("origin"), quality_status=rep["quality_status"],
            excluded=rep["excluded"], exclusion_reason=rep.get("exclusion_reason"), n_windows=rep["n_windows"],
            n_calibration_windows=rep["n_calibration_windows"], n_eval_windows=rep["n_eval_windows"],
            calibration_end_s=rep.get("calibration_end_s"), modalities=rep["modalities"],
            sampling_rates=rep["sampling_rates"], quality=rep["quality"], phases=rep["phases"], notes=rep["notes"]))
        for f in rep.get("source_files", []):
            db.add(DatasetFile(dataset_id=ds.id, relative_path=f["path"], bytes=f["bytes"], sha256=f.get("sha256"),
                               session_key=rep["session_id"]))
    db.flush()


def resolve_server_source(relative: str) -> Path:
    path = safe_join(get_settings().datasets_dir, relative)
    if not path.exists():
        raise APIError(404, "not_found", f"dataset path '{relative}' does not exist under DATASETS_DIR")
    return path
