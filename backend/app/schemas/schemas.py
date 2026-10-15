"""Request/response schemas (validated by Pydantic, documented in /docs)."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

T = TypeVar("T")
MODALITY = Literal["EDA", "HR", "TEMP", "BVP", "ACC"]


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int
    page_size: int


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    environment: str
    database: dict
    storage: dict
    dataset: dict
    models: dict


class FeatureConfigIn(BaseModel):
    window_s: float = Field(30.0, ge=5, le=600)
    step_s: float = Field(10.0, ge=1, le=600)
    min_valid_fraction: float = Field(0.8, ge=0.1, le=1.0)
    rest_guard_s: float = Field(60.0, ge=0, le=600)
    calibration_s: float = Field(300.0, ge=30, le=3600)
    min_calibration_windows: int = Field(10, ge=2, le=1000)

    @field_validator("step_s")
    @classmethod
    def _step(cls, v, info):
        if "window_s" in info.data and v > info.data["window_s"]:
            raise ValueError("step_s must not exceed window_s (windows would leave gaps)")
        return v


class IngestRequest(BaseModel):
    source_type: Literal["physionet_wearable"] = "physionet_wearable"
    path: str = Field(..., min_length=1, max_length=400, description="path relative to DATASETS_DIR")
    protocols: list[Literal["STRESS", "AEROBIC", "ANAEROBIC"]] | None = None
    feature_config: FeatureConfigIn = Field(default_factory=FeatureConfigIn)


class ValidatePathRequest(BaseModel):
    path: str = Field(..., min_length=1, max_length=400)


class DatasetOut(ORM):
    id: int
    name: str
    slug: str
    version: str
    doi: str | None
    license: str | None
    source_type: str
    status: str
    progress: str | None
    error: str | None
    n_sessions: int
    n_participants: int
    n_windows: int
    feature_config: dict | None
    created_at: datetime
    updated_at: datetime


class SessionOut(ORM):
    id: int
    session_key: str
    protocol: str
    duration_s: float
    origin: str | None
    quality_status: str
    excluded: bool
    exclusion_reason: str | None
    n_windows: int
    n_calibration_windows: int
    n_eval_windows: int
    calibration_end_s: float | None
    modalities: list | None
    sampling_rates: dict | None
    quality: dict | None
    phases: list | None
    notes: list | None
    participant: str | None = None


class TrainRequest(BaseModel):
    method: str = Field(..., examples=["robust_z_personal"])
    modalities: list[MODALITY] | None = Field(None, description="subset of modalities; default all")
    params: dict[str, Any] | None = Field(None, description="scorer hyper-parameters")
    threshold_quantile: float = Field(0.95, gt=0.5, lt=1.0)
    n_folds: int = Field(5, ge=2, le=10)
    seed: int = Field(42, ge=0, le=2**31 - 1)
    dataset_id: int | None = None

    @field_validator("params")
    @classmethod
    def _params(cls, v):
        allowed = {"n_estimators", "max_samples", "contamination", "nu", "gamma", "max_train", "hidden", "bottleneck",
                   "dropout", "lr", "weight_decay", "epochs", "batch_size", "patience"}
        if v:
            bad = set(v) - allowed
            if bad:
                raise ValueError(f"unknown parameters: {sorted(bad)}")
            for k, val in v.items():
                if not isinstance(val, (int, float, str)) or isinstance(val, bool):
                    raise ValueError(f"parameter {k} must be a number or string")
            if "epochs" in v and not (1 <= float(v["epochs"]) <= 2000):
                raise ValueError("epochs must be in [1, 2000]")
            if "n_estimators" in v and not (10 <= float(v["n_estimators"]) <= 2000):
                raise ValueError("n_estimators must be in [10, 2000]")
            if "nu" in v and not (0 < float(v["nu"]) <= 1):
                raise ValueError("nu must be in (0, 1]")
        return v


class TrainingRunOut(ORM):
    id: int
    dataset_id: int
    method: str
    status: str
    progress: str | None
    error: str | None
    features: list
    params: dict | None
    config: dict
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    model_version_id: int | None = None


class ModelVersionOut(ORM):
    id: int
    training_run_id: int
    dataset_id: int
    method: str
    scorer: str
    normalization: str
    version: int
    feature_hash: str
    features: list
    threshold: float | None
    threshold_quantile: float | None
    seed: int
    created_at: datetime
    metrics: dict | None = None
    artifact_ok: bool | None = None


class DetectRequest(BaseModel):
    model_id: int = Field(..., ge=1)
    session_key: str = Field(..., min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_\-]+$")
    threshold: float | None = Field(None, description="override the model's calibrated threshold")


class ExplainRequest(BaseModel):
    model_id: int = Field(..., ge=1)
    session_key: str = Field(..., min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_\-]+$")
    window_index: int = Field(..., ge=0)


class AnomalyEventOut(ORM):
    id: int
    detection_run_id: int
    model_version_id: int
    session_key: str | None = None
    participant: str | None = None
    window_index: int
    start_s: float
    end_s: float
    start_time: str | None
    score: float
    threshold: float
    phase: str | None
    quality: dict | None
    created_at: datetime


class ReportRequest(BaseModel):
    title: str = Field("MOSAIC research report", min_length=3, max_length=200)
    experiment_run_id: str | None = Field(None, max_length=100, pattern=r"^[A-Za-z0-9_\-]+$")
    include_models: bool = True


class ReportOut(ORM):
    id: int
    title: str
    experiment_run: str | None
    bytes: int
    sha256: str
    created_at: datetime


class ExperimentLaunch(BaseModel):
    quick: bool = Field(False, description="fewer bootstrap iterations and epochs (smoke test)")
