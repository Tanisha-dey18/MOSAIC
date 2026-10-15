"""Window segmentation and feature extraction.

Every session is cut into fixed-length windows on a common time axis
(seconds since the session origin). For each window and modality we check
data quality and compute a small set of interpretable features. Feature names
and order are fixed by FEATURE_SPECS so that training and inference always see
identical columns.

Windows are assigned to protocol phases only if they lie *entirely* inside one
phase, so no window straddles a calibration/evaluation or stressor/rest boundary.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import signal as sps

from ml.config import FeatureConfig
from ml.data_ingestion.session import LABEL_REST, Session
from ml.preprocessing.filters import preprocess
from ml.preprocessing.quality import invalid_mask

# (feature name, modality, unit, description)
FEATURE_SPECS: list[tuple[str, str, str, str]] = [
    ("eda_mean", "EDA", "uS", "mean skin conductance level"),
    ("eda_std", "EDA", "uS", "standard deviation of skin conductance"),
    ("eda_slope", "EDA", "uS/min", "linear trend of skin conductance"),
    ("eda_scr_rate", "EDA", "1/min", "skin-conductance-response peaks per minute"),
    ("hr_mean", "HR", "bpm", "mean heart rate"),
    ("hr_std", "HR", "bpm", "heart-rate variability within window (std)"),
    ("hr_slope", "HR", "bpm/min", "linear trend of heart rate"),
    ("temp_mean", "TEMP", "degC", "mean skin temperature"),
    ("temp_slope", "TEMP", "degC/min", "linear trend of skin temperature"),
    ("bvp_amp", "BVP", "a.u.", "pulse amplitude (inter-quartile range of filtered BVP)"),
    ("bvp_rate", "BVP", "bpm", "dominant pulse frequency of BVP (Welch spectrum, 0.7-3.5 Hz)"),
    ("acc_mean", "ACC", "g", "mean acceleration magnitude"),
    ("acc_std", "ACC", "g", "variability of acceleration magnitude (movement)"),
    ("eda_hr_corr", "EDA+HR", "r", "Pearson correlation of EDA and HR within the window"),
]
FEATURE_NAMES = [f[0] for f in FEATURE_SPECS]
FEATURE_MODALITY = {f[0]: f[1] for f in FEATURE_SPECS}

META_COLUMNS = [
    "session_id", "participant_id", "protocol", "protocol_version", "window_index", "start_s", "end_s",
    "start_time", "phase", "label", "is_calibration", "is_eval", "excluded", "calibration_end_s",
]


def features_for_modalities(modalities: tuple[str, ...] | list[str]) -> list[str]:
    """Feature columns available when only `modalities` are used (keeps canonical order)."""
    mods = set(modalities)
    return [f for f in FEATURE_NAMES if set(FEATURE_MODALITY[f].split("+")) <= mods]


def _slope_per_min(x: np.ndarray, fs: float) -> float:
    ok = np.isfinite(x)
    if ok.sum() < 3:
        return np.nan
    t = np.arange(x.size)[ok] / fs / 60.0
    return float(np.polyfit(t, x[ok], 1)[0])


def _scr_rate(x: np.ndarray, fs: float) -> float:
    ok = x[np.isfinite(x)]
    if ok.size < fs * 5:
        return np.nan
    # A skin-conductance response: a rise of >= 0.01 uS above the surrounding signal, >= 1 s apart.
    peaks, _ = sps.find_peaks(ok, prominence=0.01, distance=max(int(fs), 1))
    return float(len(peaks) / (ok.size / fs / 60.0))


def _dominant_rate(x: np.ndarray, fs: float) -> float:
    ok = x[np.isfinite(x)]
    if ok.size < fs * 8:
        return np.nan
    f, p = sps.welch(ok - ok.mean(), fs=fs, nperseg=min(ok.size, int(fs * 8)))
    band = (f >= 0.7) & (f <= 3.5)
    if not band.any() or p[band].max() <= 0:
        return np.nan
    return float(f[band][np.argmax(p[band])] * 60.0)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.size, b.size)
    if n < 5:
        return np.nan
    a, b = a[:n], b[:n]
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or np.std(a[ok]) < 1e-9 or np.std(b[ok]) < 1e-9:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _modality_features(name: str, x: np.ndarray, fs: float) -> dict[str, float]:
    if name == "EDA":
        return {"eda_mean": float(np.nanmean(x)), "eda_std": float(np.nanstd(x)),
                "eda_slope": _slope_per_min(x, fs), "eda_scr_rate": _scr_rate(x, fs)}
    if name == "HR":
        return {"hr_mean": float(np.nanmean(x)), "hr_std": float(np.nanstd(x)), "hr_slope": _slope_per_min(x, fs)}
    if name == "TEMP":
        return {"temp_mean": float(np.nanmean(x)), "temp_slope": _slope_per_min(x, fs)}
    if name == "BVP":
        q75, q25 = np.nanpercentile(x, [75, 25])
        return {"bvp_amp": float(q75 - q25), "bvp_rate": _dominant_rate(x, fs)}
    if name == "ACC":
        return {"acc_mean": float(np.nanmean(x)), "acc_std": float(np.nanstd(x))}
    return {}


def _resample_1hz(x: np.ndarray, fs: float) -> np.ndarray:
    """Mean per whole second (used for cross-signal correlation)."""
    k = int(round(fs))
    if k <= 1:
        return x
    n = x.size // k
    if n == 0:
        return np.array([])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN seconds -> NaN is intended
        return np.nanmean(x[: n * k].reshape(n, k), axis=1)


def extract_session_features(session: Session, cfg: FeatureConfig) -> pd.DataFrame:
    """Return one row per window with metadata, per-modality quality and features."""
    duration = session.duration_s
    starts = np.arange(0.0, max(duration - cfg.window_s, -1) + 1e-9, cfg.step_s)
    cleaned: dict[str, tuple[np.ndarray, np.ndarray, float, float]] = {}
    for name in cfg.modalities:
        if name not in session.signals or name in session.invalid_modalities:
            continue
        sig = session.signals[name]
        bad = invalid_mask(name, sig.values)
        vals = sig.values.astype(float).copy()
        vals[bad] = np.nan
        cleaned[name] = (preprocess(name, vals, sig.fs), bad, sig.fs, session.offset_s(name))

    rows = []
    for i, s in enumerate(starts):
        e = s + cfg.window_s
        row: dict = {f: np.nan for f in FEATURE_NAMES}
        segs = {}
        for name, (clean, bad, fs, off) in cleaned.items():
            i0, i1 = int(np.floor((s - off) * fs)), int(np.floor((e - off) * fs))
            expected = max(i1 - i0, 1)
            i0c, i1c = max(i0, 0), min(i1, clean.size)
            if i1c <= i0c:
                row[f"q_{name}"] = 0.0
                continue
            seg = clean[i0c:i1c]
            valid = float((np.isfinite(seg) & ~bad[i0c:i1c]).sum() / expected)
            row[f"q_{name}"] = round(valid, 4)
            if valid >= cfg.min_valid_fraction:
                row.update(_modality_features(name, seg, fs))
                segs[name] = (seg, fs)
        for name in cfg.modalities:
            row.setdefault(f"q_{name}", 0.0)
        if "EDA" in segs and "HR" in segs:
            row["eda_hr_corr"] = _corr(_resample_1hz(*segs["EDA"]), _resample_1hz(*segs["HR"]))

        phase = session.phase_at(s, e)
        label = np.nan
        if phase is not None and phase.label is not None:
            label = float(phase.label)
            if phase.label == LABEL_REST and s < phase.start_s + cfg.rest_guard_s:
                label = np.nan  # recovery period right after a stressor: ambiguous
        cal_end = session.calibration_end_s
        is_cal = bool(cal_end is not None and e <= cal_end and s >= cal_end - cfg.calibration_s)
        is_eval = bool(cal_end is not None and s >= cal_end and not np.isnan(label) and not session.excluded)
        row.update({
            "session_id": session.session_id, "participant_id": session.participant_id,
            "protocol": session.protocol, "protocol_version": session.protocol_version,
            "window_index": i, "start_s": float(s), "end_s": float(e),
            "start_time": (session.origin + pd.Timedelta(seconds=float(s))).isoformat() if session.origin is not None else None,
            "phase": phase.name if phase else None, "label": label,
            "is_calibration": is_cal and not session.excluded, "is_eval": is_eval, "excluded": session.excluded,
            "calibration_end_s": cal_end if cal_end is not None else np.nan,
        })
        rows.append(row)
    cols = META_COLUMNS + [f"q_{m}" for m in cfg.modalities] + FEATURE_NAMES
    return pd.DataFrame(rows, columns=cols)


def extract_features(sessions: list[Session], cfg: FeatureConfig) -> pd.DataFrame:
    frames = [extract_session_features(s, cfg) for s in sessions]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=META_COLUMNS + FEATURE_NAMES)
    return pd.concat(frames, ignore_index=True)


def downsample_signals(session: Session, rate_hz: float = 1.0) -> pd.DataFrame:
    """Per-second summary of each cleaned signal, used for charts (not for modelling)."""
    duration = int(np.ceil(session.duration_s))
    out = pd.DataFrame({"t_s": np.arange(duration, dtype=float)})
    for name, sig in session.signals.items():
        bad = invalid_mask(name, sig.values)
        vals = sig.values.astype(float).copy()
        vals[bad] = np.nan
        clean = preprocess(name, vals, sig.fs)
        k = max(int(round(sig.fs / rate_hz)), 1)
        n = clean.size // k
        if n == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            blocks = clean[: n * k].reshape(n, k)
            series = np.nanstd(blocks, axis=1) if name == "BVP" else np.nanmean(blocks, axis=1)
        off = int(round(session.offset_s(name)))
        col = np.full(duration, np.nan)
        end = min(off + n, duration)
        if end > off >= 0:
            col[off:end] = series[: end - off]
        out[name] = col
        if name in session.invalid_modalities:
            out[name] = np.nan
    return out
