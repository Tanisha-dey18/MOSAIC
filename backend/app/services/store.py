"""Cached access to processed feature files (re-read when the file changes)."""
from __future__ import annotations

import threading
from pathlib import Path

import pandas as pd

from ml.pipeline import load_features

_lock = threading.Lock()
_cache: dict[str, tuple[float, pd.DataFrame]] = {}


def features(directory: Path) -> pd.DataFrame:
    path = directory / "features.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    mtime = path.stat().st_mtime
    key = str(path)
    with _lock:
        hit = _cache.get(key)
        if hit and hit[0] == mtime:
            return hit[1]
    df = load_features(directory)
    with _lock:
        _cache[key] = (mtime, df)
    return df
