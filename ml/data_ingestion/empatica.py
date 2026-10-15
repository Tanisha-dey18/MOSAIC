"""Readers for raw Empatica E4 CSV exports.

Format (as documented in the PhysioNet dataset README):
  * Signal files (ACC, BVP, EDA, HR, TEMP): row 1 = session start time (UTC),
    row 2 = sample rate in Hz, then one sample per row (ACC has 3 columns).
  * IBI.csv: row 1 = start time, then rows "<seconds since start>, <ibi seconds>".
  * tags.csv: one UTC timestamp per row (button presses), possibly empty.

The original Empatica export writes Unix timestamps; this PhysioNet release
converted them to UTC date strings. Both are supported here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SIGNAL_FILES = ("EDA", "BVP", "HR", "TEMP", "ACC")


class IngestionError(ValueError):
    """Raised when a file cannot be parsed as an Empatica export."""


@dataclass
class Signal:
    """A uniformly sampled signal from one sensor."""

    name: str
    start: pd.Timestamp  # UTC
    fs: float  # Hz
    values: np.ndarray  # shape (n_samples, n_channels)
    channels: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return int(self.values.shape[0])

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.fs if self.fs > 0 else 0.0

    def times_s(self) -> np.ndarray:
        """Seconds since this signal's own start."""
        return np.arange(self.n_samples) / self.fs


def _parse_start(token: str) -> pd.Timestamp:
    token = token.strip().strip('"')
    try:
        as_float = float(token)
        return pd.Timestamp(as_float, unit="s", tz="UTC")
    except ValueError:
        pass
    try:
        ts = pd.Timestamp(token)
    except (ValueError, TypeError) as exc:  # pandas' DateParseError subclasses ValueError
        raise IngestionError(f"invalid start timestamp: {token!r}") from exc
    if pd.isna(ts):
        raise IngestionError(f"invalid start timestamp: {token!r}")
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _read_lines(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise IngestionError(f"cannot read {path.name}: {exc}") from exc
    return [ln for ln in text.splitlines() if ln.strip() != ""]


def read_signal(path: str | Path, name: str | None = None) -> Signal:
    """Read an Empatica uniformly-sampled signal file (EDA, BVP, HR, TEMP, ACC)."""
    path = Path(path)
    name = name or path.stem.upper()
    lines = _read_lines(path)
    if len(lines) < 2:
        raise IngestionError(f"{path.name}: expected at least 2 header rows, found {len(lines)}")

    start_tokens = [t for t in lines[0].split(",") if t.strip()]
    rate_tokens = [t for t in lines[1].split(",") if t.strip()]
    if not start_tokens or not rate_tokens:
        raise IngestionError(f"{path.name}: malformed header")
    start = _parse_start(start_tokens[0])
    try:
        fs = float(rate_tokens[0])
    except ValueError as exc:
        raise IngestionError(f"{path.name}: invalid sample rate {rate_tokens[0]!r}") from exc
    if not np.isfinite(fs) or fs <= 0 or fs > 10_000:
        raise IngestionError(f"{path.name}: implausible sample rate {fs}")

    n_cols = len(start_tokens)
    issues: list[str] = []
    body = lines[2:]
    if body:
        frame = pd.read_csv(
            pd.io.common.StringIO("\n".join(body)), header=None, usecols=range(n_cols), on_bad_lines="skip"
        )
        values = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        n_bad = int(np.isnan(values).any(axis=1).sum())
        if n_bad:
            issues.append(f"{n_bad} non-numeric/corrupted rows set to NaN")
        skipped = len(body) - len(frame)
        if skipped:
            issues.append(f"{skipped} malformed rows skipped")
    else:
        values = np.empty((0, n_cols))
        issues.append("file contains no samples")

    if name == "ACC" and n_cols == 3:
        channels = ["x", "y", "z"]
    elif n_cols == 1:
        channels = [name.lower()]
    else:
        channels = [f"c{i}" for i in range(n_cols)]
    return Signal(name=name, start=start, fs=fs, values=values, channels=channels, issues=issues)


def read_ibi(path: str | Path) -> tuple[pd.Timestamp | None, pd.DataFrame]:
    """Read IBI.csv -> (start, DataFrame[t_s, ibi_s]). Empty files return an empty frame."""
    lines = _read_lines(Path(path))
    if not lines:
        return None, pd.DataFrame(columns=["t_s", "ibi_s"])
    start = _parse_start(lines[0].split(",")[0])
    rows = []
    for ln in lines[1:]:
        parts = ln.split(",")
        if len(parts) < 2:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    frame = pd.DataFrame(rows, columns=["t_s", "ibi_s"])
    # Physiologically implausible intervals (<0.25 s = 240 bpm, >2 s = 30 bpm) are dropped.
    frame = frame[(frame["ibi_s"] > 0.25) & (frame["ibi_s"] < 2.0)].reset_index(drop=True)
    return start, frame


def read_tags(path: str | Path) -> list[pd.Timestamp]:
    """Read tags.csv -> sorted list of UTC timestamps (empty list if no tags)."""
    path = Path(path)
    if not path.exists():
        return []
    tags = []
    for ln in _read_lines(path):
        try:
            tags.append(_parse_start(ln.split(",")[0]))
        except (IngestionError, ValueError):
            continue
    return sorted(tags)
