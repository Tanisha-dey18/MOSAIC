"""Relational schema.

PostgreSQL stores metadata, provenance, model versions, evaluation results,
baselines, anomaly events and reports. High-frequency signals and window
features stay in files (Parquet) under STORAGE_DIR, referenced by path.
Participants are stored only by the dataset's pseudonymous identifiers;
no demographic or personal data is stored.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)


class Dataset(TimestampMixin, Base):
    __tablename__ = "datasets"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    slug: Mapped[str] = mapped_column(String(100))
    version: Mapped[str] = mapped_column(String(50))
    doi: Mapped[str | None] = mapped_column(String(100))
    license: Mapped[str | None] = mapped_column(String(255))
    source_type: Mapped[str] = mapped_column(String(30))  # physionet_wearable | generic_csv
    source_path: Mapped[str] = mapped_column(String(500))
    storage_path: Mapped[str | None] = mapped_column(String(500))
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)  # pending|processing|ready|failed
    progress: Mapped[str | None] = mapped_column(String(255))
    error: Mapped[str | None] = mapped_column(Text)
    feature_config: Mapped[dict | None] = mapped_column(JSON)
    manifest: Mapped[dict | None] = mapped_column(JSON)
    n_sessions: Mapped[int] = mapped_column(Integer, default=0)
    n_participants: Mapped[int] = mapped_column(Integer, default=0)
    n_windows: Mapped[int] = mapped_column(Integer, default=0)

    files: Mapped[list["DatasetFile"]] = relationship(back_populates="dataset", cascade="all, delete-orphan")
    participants: Mapped[list["Participant"]] = relationship(back_populates="dataset", cascade="all, delete-orphan")


class DatasetFile(Base):
    __tablename__ = "dataset_files"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    relative_path: Mapped[str] = mapped_column(String(500))
    bytes: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str | None] = mapped_column(String(64))
    session_key: Mapped[str | None] = mapped_column(String(100))
    dataset: Mapped[Dataset] = relationship(back_populates="files")
    __table_args__ = (UniqueConstraint("dataset_id", "relative_path", name="uq_dataset_file_path"),)


class Participant(Base):
    __tablename__ = "participants"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    pseudonym: Mapped[str] = mapped_column(String(64))
    protocol_version: Mapped[str | None] = mapped_column(String(10))
    n_sessions: Mapped[int] = mapped_column(Integer, default=0)
    dataset: Mapped[Dataset] = relationship(back_populates="participants")
    sessions: Mapped[list["RecordingSession"]] = relationship(back_populates="participant", cascade="all, delete-orphan")
    __table_args__ = (UniqueConstraint("dataset_id", "pseudonym", name="uq_participant_pseudonym"),)


class RecordingSession(Base):
    __tablename__ = "recording_sessions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    participant_id: Mapped[int] = mapped_column(ForeignKey("participants.id", ondelete="CASCADE"), index=True)
    session_key: Mapped[str] = mapped_column(String(100))
    protocol: Mapped[str] = mapped_column(String(30), index=True)
    duration_s: Mapped[float] = mapped_column(Float)
    origin: Mapped[str | None] = mapped_column(String(40))
    quality_status: Mapped[str] = mapped_column(String(20))
    excluded: Mapped[bool] = mapped_column(Boolean, default=False)
    exclusion_reason: Mapped[str | None] = mapped_column(Text)
    n_windows: Mapped[int] = mapped_column(Integer, default=0)
    n_calibration_windows: Mapped[int] = mapped_column(Integer, default=0)
    n_eval_windows: Mapped[int] = mapped_column(Integer, default=0)
    calibration_end_s: Mapped[float | None] = mapped_column(Float)
    modalities: Mapped[list | None] = mapped_column(JSON)
    sampling_rates: Mapped[dict | None] = mapped_column(JSON)
    quality: Mapped[dict | None] = mapped_column(JSON)
    phases: Mapped[list | None] = mapped_column(JSON)
    notes: Mapped[list | None] = mapped_column(JSON)
    participant: Mapped[Participant] = relationship(back_populates="sessions")
    __table_args__ = (UniqueConstraint("dataset_id", "session_key", name="uq_session_key"),)


class TrainingRun(TimestampMixin, Base):
    __tablename__ = "training_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    method: Mapped[str] = mapped_column(String(50), index=True)
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)  # queued|running|completed|failed
    progress: Mapped[str | None] = mapped_column(String(255))
    config: Mapped[dict] = mapped_column(JSON)
    features: Mapped[list] = mapped_column(JSON)
    params: Mapped[dict | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    model_version: Mapped["ModelVersion | None"] = relationship(back_populates="training_run", uselist=False)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    training_run_id: Mapped[int] = mapped_column(ForeignKey("training_runs.id", ondelete="CASCADE"), unique=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    method: Mapped[str] = mapped_column(String(50), index=True)
    scorer: Mapped[str] = mapped_column(String(30))
    normalization: Mapped[str] = mapped_column(String(20))
    version: Mapped[int] = mapped_column(Integer)
    artifact_path: Mapped[str] = mapped_column(String(500))
    feature_hash: Mapped[str] = mapped_column(String(32))
    features: Mapped[list] = mapped_column(JSON)
    threshold: Mapped[float | None] = mapped_column(Float)
    threshold_quantile: Mapped[float | None] = mapped_column(Float)
    seed: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    training_run: Mapped[TrainingRun] = relationship(back_populates="model_version")
    evaluations: Mapped[list["EvaluationResult"]] = relationship(back_populates="model_version", cascade="all, delete-orphan")
    __table_args__ = (UniqueConstraint("method", "version", "dataset_id", name="uq_model_method_version"),)


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(100))
    metrics: Mapped[dict | None] = mapped_column(JSON)
    curves: Mapped[dict | None] = mapped_column(JSON)
    per_participant: Mapped[dict | None] = mapped_column(JSON)
    details: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    model_version: Mapped[ModelVersion] = relationship(back_populates="evaluations")


class BaselineProfile(Base):
    __tablename__ = "baseline_profiles"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id", ondelete="CASCADE"), index=True)
    participant_id: Mapped[int] = mapped_column(ForeignKey("participants.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(20))  # ok | insufficient
    n_windows: Mapped[int] = mapped_column(Integer)
    session_key: Mapped[str | None] = mapped_column(String(100))
    period_start_s: Mapped[float | None] = mapped_column(Float)
    period_end_s: Mapped[float | None] = mapped_column(Float)
    period_start_time: Mapped[str | None] = mapped_column(String(40))
    period_end_time: Mapped[str | None] = mapped_column(String(40))
    center: Mapped[list | None] = mapped_column(JSON)
    scale: Mapped[list | None] = mapped_column(JSON)
    reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    __table_args__ = (UniqueConstraint("model_version_id", "participant_id", name="uq_baseline_model_participant"),)


class DetectionRun(Base):
    __tablename__ = "detection_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id", ondelete="CASCADE"), index=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("recording_sessions.id", ondelete="CASCADE"), index=True)
    threshold: Mapped[float] = mapped_column(Float)
    n_windows: Mapped[int] = mapped_column(Integer)
    n_scored: Mapped[int] = mapped_column(Integer)
    n_flagged: Mapped[int] = mapped_column(Integer)
    result_path: Mapped[str] = mapped_column(String(500))
    mode: Mapped[str] = mapped_column(String(30), default="offline_retrospective")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AnomalyEvent(Base):
    __tablename__ = "anomaly_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    detection_run_id: Mapped[int] = mapped_column(ForeignKey("detection_runs.id", ondelete="CASCADE"), index=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id", ondelete="CASCADE"), index=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("recording_sessions.id", ondelete="CASCADE"), index=True)
    participant_id: Mapped[int] = mapped_column(ForeignKey("participants.id", ondelete="CASCADE"), index=True)
    window_index: Mapped[int] = mapped_column(Integer)
    start_s: Mapped[float] = mapped_column(Float)
    end_s: Mapped[float] = mapped_column(Float)
    start_time: Mapped[str | None] = mapped_column(String(40))
    score: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    phase: Mapped[str | None] = mapped_column(String(50))
    quality: Mapped[dict | None] = mapped_column(JSON)
    features: Mapped[dict | None] = mapped_column(JSON)
    contributions: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    __table_args__ = (Index("ix_anomaly_score", "model_version_id", "score"),)


class ResearchReport(Base):
    __tablename__ = "research_reports"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(255))
    experiment_run: Mapped[str | None] = mapped_column(String(100))
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.id", ondelete="SET NULL"))
    file_path: Mapped[str] = mapped_column(String(500))
    sha256: Mapped[str] = mapped_column(String(64))
    bytes: Mapped[int] = mapped_column(Integer)
    contents: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
