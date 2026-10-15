"""Generic long-format CSV ingestion (for user uploads).

Expected columns (case-insensitive):
  * ``timestamp``      ISO-8601 date-time or Unix seconds (required)
  * ``participant_id`` pseudonymous identifier (required)
  * ``session_id``     optional; defaults to the participant id
  * one or more signal columns: ``eda`` (uS), ``hr`` (bpm), ``temp`` (deg C),
    ``bvp`` (a.u.), ``acc_x``/``acc_y``/``acc_z`` (1/64 g) or ``acc`` (g magnitude x 64)
  * ``phase`` (optional text) and ``label`` (optional 0/1) for annotated data
  * ``calibration`` (optional 0/1) marking rows usable as a personal baseline

The sampling rate is inferred per session from the median timestamp spacing.
Rows with unparseable timestamps are dropped and reported; duplicate
timestamps are collapsed (first value kept) and reported.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.data_ingestion.empatica import IngestionError, Signal
from ml.data_ingestion.session import Phase, Session

SIGNAL_COLUMNS = {"eda": "EDA", "hr": "HR", "temp": "TEMP", "bvp": "BVP"}
ACC_COLUMNS = ("acc_x", "acc_y", "acc_z")
MAX_ROWS = 20_000_000


def _parse_timestamps(col: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(col, errors="coerce")
    if numeric.notna().mean() > 0.9:
        return pd.to_datetime(numeric, unit="s", utc=True, errors="coerce")
    return pd.to_datetime(col, utc=True, errors="coerce", format="mixed")


def validate_frame(df: pd.DataFrame) -> dict:
    """Validate a generic CSV without building sessions. Returns a report dict."""
    errors, warnings = [], []
    cols = {c.lower().strip(): c for c in df.columns}
    for required in ("timestamp", "participant_id"):
        if required not in cols:
            errors.append(f"missing required column '{required}'")
    signals = [k for k in SIGNAL_COLUMNS if k in cols]
    if all(c in cols for c in ACC_COLUMNS) or "acc" in cols:
        signals.append("acc")
    if not signals:
        errors.append("no signal columns found (expected any of eda, hr, temp, bvp, acc_x/acc_y/acc_z)")
    report = {"n_rows": int(len(df)), "columns": list(df.columns), "signals": signals,
              "errors": errors, "warnings": warnings, "participants": 0}
    if len(df) == 0:
        errors.append("file contains no data rows")
    if errors:
        report["valid"] = False
        return report

    ts = _parse_timestamps(df[cols["timestamp"]])
    bad_ts = int(ts.isna().sum())
    if bad_ts:
        warnings.append(f"{bad_ts} rows with invalid timestamps will be dropped")
    pid = df[cols["participant_id"]].astype(str).str.strip()
    if (pid == "").any() or pid.isna().any():
        warnings.append("rows with empty participant_id will be dropped")
    dup = int(pd.DataFrame({"p": pid, "t": ts}).duplicated().sum())
    if dup:
        warnings.append(f"{dup} duplicate (participant, timestamp) rows will be collapsed")
    for key in signals:
        col_names = [cols[c] for c in ACC_COLUMNS] if key == "acc" and "acc" not in cols else [cols[key]]
        for cn in col_names:
            vals = pd.to_numeric(df[cn], errors="coerce")
            frac = float(vals.isna().mean())
            if frac > 0:
                warnings.append(f"column '{cn}': {frac:.1%} missing or non-numeric values")
    report["participants"] = int(pid[pid != ""].nunique())
    report["valid"] = bad_ts < len(df)
    if not report["valid"]:
        errors.append("no valid timestamps")
    return report


def load_frame(df: pd.DataFrame, protocol: str = "UPLOAD") -> list[Session]:
    report = validate_frame(df)
    if not report["valid"]:
        raise IngestionError("; ".join(report["errors"]))
    cols = {c.lower().strip(): c for c in df.columns}
    work = pd.DataFrame({
        "ts": _parse_timestamps(df[cols["timestamp"]]),
        "pid": df[cols["participant_id"]].astype(str).str.strip(),
    })
    work["sid"] = df[cols["session_id"]].astype(str).str.strip() if "session_id" in cols else work["pid"]
    for key, name in SIGNAL_COLUMNS.items():
        if key in cols:
            work[name] = pd.to_numeric(df[cols[key]], errors="coerce")
    if all(c in cols for c in ACC_COLUMNS):
        for c in ACC_COLUMNS:
            work[c] = pd.to_numeric(df[cols[c]], errors="coerce")
    elif "acc" in cols:
        work["acc_x"] = pd.to_numeric(df[cols["acc"]], errors="coerce")
    for opt in ("phase", "label", "calibration"):
        if opt in cols:
            work[opt] = df[cols[opt]]
    work = work.dropna(subset=["ts"])
    work = work[work["pid"] != ""]
    work = work.sort_values(["sid", "ts"]).drop_duplicates(subset=["sid", "ts"], keep="first")

    sessions = []
    for sid, g in work.groupby("sid", sort=True):
        if len(g) < 2:
            continue
        dt = g["ts"].diff().dt.total_seconds().dropna()
        dt = dt[dt > 0]
        if dt.empty:
            continue
        fs = 1.0 / float(dt.median())
        start = g["ts"].iloc[0]
        # Re-grid onto a uniform time base so gaps become NaN rather than being silently compressed.
        t_rel = (g["ts"] - start).dt.total_seconds().to_numpy()
        idx = np.rint(t_rel * fs).astype(int)
        n = int(idx.max()) + 1
        if n > MAX_ROWS:
            raise IngestionError(f"session {sid}: too many samples after re-gridding ({n})")
        signals = {}
        for name in ("EDA", "HR", "TEMP", "BVP"):
            if name in g:
                arr = np.full((n, 1), np.nan)
                arr[idx, 0] = g[name].to_numpy()
                signals[name] = Signal(name, start, fs, arr, [name.lower()])
        if "acc_x" in g:
            if "acc_y" in g:
                arr = np.full((n, 3), np.nan)
                arr[idx] = g[list(ACC_COLUMNS)].to_numpy()
                signals["ACC"] = Signal("ACC", start, fs, arr, ["x", "y", "z"])
            else:
                arr = np.full((n, 1), np.nan)
                arr[idx, 0] = g["acc_x"].to_numpy()
                signals["ACC"] = Signal("ACC", start, fs, arr, ["magnitude"])
        session = Session(session_id=str(sid), participant_id=str(g["pid"].iloc[0]), protocol=protocol,
                          signals=signals, origin=start)
        session.notes.append(f"sampling rate inferred as {fs:.3f} Hz from median timestamp spacing")
        if "label" in g:
            session.phases = _phases_from_labels(t_rel, g)
        if "calibration" in g:
            cal = pd.to_numeric(g["calibration"], errors="coerce").fillna(0).astype(int).to_numpy()
            if cal.any():
                # Personal baseline = the leading calibration block only (strictly before evaluation data).
                first_non = np.argmax(cal == 0) if (cal == 0).any() else len(cal)
                session.calibration_end_s = float(t_rel[first_non - 1]) if first_non > 0 else None
        sessions.append(session)
    if not sessions:
        raise IngestionError("no session with at least two valid samples")
    return sessions


def _phases_from_labels(t_rel: np.ndarray, g: pd.DataFrame) -> list[Phase]:
    labels = pd.to_numeric(g["label"], errors="coerce").to_numpy()
    names = g["phase"].astype(str).to_numpy() if "phase" in g else None
    phases = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start] or (names is not None and names[i] != names[start]):
            lab = labels[start]
            phases.append(Phase(names[start] if names is not None else f"segment_{len(phases)}",
                                float(t_rel[start]), float(t_rel[i - 1]),
                                None if np.isnan(lab) else int(lab)))
            start = i
    return phases


def load_csv(path: str | Path, protocol: str = "UPLOAD") -> list[Session]:
    try:
        df = pd.read_csv(path, low_memory=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise IngestionError(f"cannot parse CSV: {exc}") from exc
    return load_frame(df, protocol=protocol)
