"""Minimal background job runner.

Long-running work (dataset ingestion, model training) runs in a single worker
thread so the API process stays responsive; job status is persisted in the
database (datasets.status / training_runs.status), so the UI polls the DB.

This is adequate for a single-instance research deployment. For multi-instance
production use, replace it with a task queue (e.g. Celery/RQ + Redis); the job
functions are already self-contained and take only IDs as arguments.
"""
from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable

from app.core.config import get_settings

log = logging.getLogger("mosaic.jobs")
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mosaic-job")


def submit(fn: Callable, *args) -> Future | None:
    if get_settings().jobs_sync:
        fn(*args)
        return None

    def _wrapped():
        try:
            fn(*args)
        except Exception:  # the job itself records failure in the DB
            log.exception("background job crashed", extra={"job": fn.__name__})

    return _executor.submit(_wrapped)
