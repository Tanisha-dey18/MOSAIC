"""Path-safety helpers (path traversal protection)."""
from __future__ import annotations

import re
from pathlib import Path

from app.core.errors import APIError

SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,127}$")


def safe_join(root: Path, relative: str) -> Path:
    """Resolve `relative` inside `root`; reject absolute paths and anything escaping root."""
    if not relative or "\x00" in relative:
        raise APIError(400, "invalid_path", "path must be a non-empty relative path")
    rel = Path(relative)
    if rel.is_absolute() or rel.drive or relative.startswith(("/", "\\")):
        raise APIError(400, "invalid_path", "absolute paths are not allowed")
    root = root.resolve()
    target = (root / rel).resolve()
    if target != root and root not in target.parents:
        raise APIError(400, "invalid_path", "path escapes the permitted directory")
    return target


def safe_name(name: str, what: str = "name") -> str:
    if not SAFE_NAME.match(name or "") or ".." in name:
        raise APIError(400, "invalid_name", f"invalid {what}")
    return name
