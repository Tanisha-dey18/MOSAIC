"""Environment-based configuration (12-factor). See .env.example for all variables."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    mosaic_env: str = Field("development", alias="MOSAIC_ENV")
    database_url: str = Field(f"sqlite:///{(PROJECT_ROOT / 'storage' / 'mosaic.db').as_posix()}", alias="DATABASE_URL")
    storage_dir: Path = Field(PROJECT_ROOT / "storage", alias="STORAGE_DIR")
    datasets_dir: Path = Field(PROJECT_ROOT / "datasets", alias="DATASETS_DIR")
    reports_dir: Path = Field(PROJECT_ROOT / "reports", alias="REPORTS_DIR")
    max_upload_mb: int = Field(200, alias="MAX_UPLOAD_MB")
    max_unzipped_mb: int = Field(1024, alias="MAX_UNZIPPED_MB")
    cors_origins: str = Field("http://localhost:5173,http://127.0.0.1:5173", alias="CORS_ORIGINS")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    # Run background jobs inline (used by the test-suite for determinism).
    jobs_sync: bool = Field(False, alias="JOBS_SYNC")
    # Download, ingest and train automatically at start-up (for hosts without persistent disks).
    bootstrap: bool = Field(False, alias="MOSAIC_BOOTSTRAP")
    bootstrap_methods: str = Field(
        "hr_rule,robust_z_global,robust_z_personal,robust_z_rolling,iforest_global,iforest_personal,"
        "ocsvm_global,ocsvm_personal,autoencoder_global,autoencoder_personal,arousal_z_rolling",
        alias="MOSAIC_BOOTSTRAP_METHODS")

    @field_validator("storage_dir", "datasets_dir", "reports_dir", mode="before")
    @classmethod
    def _resolve(cls, v):
        p = Path(v)
        return (PROJECT_ROOT / p).resolve() if not p.is_absolute() else p.resolve()

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def bootstrap_method_list(self) -> list[str]:
        return [m.strip() for m in self.bootstrap_methods.split(",") if m.strip()]

    @property
    def processed_dir(self) -> Path:
        return self.storage_dir / "processed"

    @property
    def artifacts_dir(self) -> Path:
        return self.storage_dir / "artifacts"

    @property
    def uploads_dir(self) -> Path:
        return self.storage_dir / "uploads"

    @property
    def detections_dir(self) -> Path:
        return self.storage_dir / "detections"

    @property
    def generated_reports_dir(self) -> Path:
        return self.storage_dir / "reports"

    @property
    def experiments_dir(self) -> Path:
        return self.reports_dir / "experiments"

    def ensure_dirs(self) -> None:
        for d in (self.storage_dir, self.processed_dir, self.artifacts_dir, self.uploads_dir, self.detections_dir,
                  self.generated_reports_dir, self.experiments_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
