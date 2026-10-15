"""Start-up bootstrap for hosts without persistent storage (e.g. Hugging Face Spaces).

Enabled with MOSAIC_BOOTSTRAP=1. Runs in the background job thread after the API
has started, so /api/health answers immediately while data is being prepared:

1. if no ready dataset exists: download the open PhysioNet release (SHA-256
   verified) into DATASETS_DIR and ingest it (progress is visible in the UI);
2. if no model exists: train MOSAIC_BOOTSTRAP_METHODS with the normal training code.

It is idempotent: with data and models already present it does nothing. Everything
it produces is real pipeline output; nothing is pre-computed or simulated.
"""
from __future__ import annotations

import logging

from sqlalchemy import func, select

from app.core.config import get_settings
from app.database.models import Dataset, ModelVersion
from app.database.session import session_factory
from app.services import datasets as dsvc
from app.services import training as tsvc
from ml.config import FeatureConfig
from ml.data_ingestion import physionet_wearable
from ml.data_ingestion.download import download

log = logging.getLogger("mosaic.bootstrap")
RELEASE_DIR = "raw/wearable-device-dataset-1.0.1"


def run_bootstrap() -> None:
    settings = get_settings()
    db = session_factory()()
    try:
        ds = dsvc.active_dataset(db, required=False)
        if ds is None:
            busy = db.scalars(select(Dataset).where(Dataset.status.in_(["pending", "processing"]))).first()
            if busy is not None:
                log.info("bootstrap skipped: an ingestion is already in progress", extra={"dataset_id": busy.id})
                return
            target = settings.datasets_dir / RELEASE_DIR
            ds = dsvc.create_dataset(db, "physionet_wearable", RELEASE_DIR, physionet_wearable.DATASET_NAME, FeatureConfig())
            ds.progress = "downloading the dataset from PhysioNet (about 250 MB)"
            db.commit()
            ds_id = ds.id
            try:
                download(target, skip_notebook=True, workers=8, log=lambda m: log.info(m))
            except Exception as exc:
                ds.status, ds.error = "failed", f"download failed: {exc}"[:2000]
                db.commit()
                log.warning("bootstrap download failed", extra={"error": str(exc)[:200]})
                return
            db.close()
            dsvc.run_ingestion(ds_id, str(target), None)
            db = session_factory()()
            ds = db.get(Dataset, ds_id)
            if ds.status != "ready":
                log.warning("bootstrap ingestion failed", extra={"error": (ds.error or "")[:200]})
                return
        if (db.scalar(select(func.count(ModelVersion.id))) or 0) > 0:
            log.info("bootstrap: data and models present, nothing to do")
            return
        run_ids = []
        for method in settings.bootstrap_method_list:
            run = tsvc.create_run(db, ds, method, None, None, 0.95, 5, 42)
            run_ids.append(run.id)
        db.close()
        for run_id in run_ids:  # sequential, like the job queue
            tsvc.execute_run(run_id)
        log.info("bootstrap complete", extra={"models": len(run_ids)})
    except Exception:
        log.exception("bootstrap crashed")
    finally:
        db.close()
