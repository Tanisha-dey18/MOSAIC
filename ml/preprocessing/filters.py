"""Signal-specific filtering.

Each modality gets its own preprocessing because the signals differ in
bandwidth and noise characteristics:

* EDA (4 Hz): 4th-order low-pass Butterworth at 1 Hz removes quantisation and
  high-frequency noise while keeping skin-conductance responses (0.05-1 Hz).
* BVP (64 Hz): band-pass 0.5-8 Hz keeps the cardiac pulse (and harmonics) and
  removes baseline wander and high-frequency noise.
* TEMP (4 Hz): median filter (5 s) removes isolated spikes; temperature changes slowly.
* HR (1 Hz, derived by Empatica from BVP): median filter (5 s) removes spikes.
* ACC (32 Hz): Euclidean magnitude in g, then a 5 Hz low-pass filter.

Filters are zero-phase (filtfilt), which uses samples on both sides of a point.
That is acceptable for *offline* window feature extraction where the whole
window is available; it is noted as a limitation for genuine real-time use.
Missing samples are linearly interpolated before filtering (gaps up to 5 s)
and restored to NaN afterwards so that quality accounting stays honest.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps


def _interpolate_gaps(x: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).copy()
    nan = ~np.isfinite(x)
    if not nan.any():
        return x, nan
    if nan.all():
        return x, nan
    idx = np.arange(x.size)
    x[nan] = np.interp(idx[nan], idx[~nan], x[~nan])
    # Gaps that are too long are not trusted: keep them as NaN.
    long_gap = np.zeros_like(nan)
    run_start = None
    for i, flag in enumerate(np.append(nan, False)):
        if flag and run_start is None:
            run_start = i
        elif not flag and run_start is not None:
            if i - run_start > max_gap:
                long_gap[run_start:i] = True
            run_start = None
    x[long_gap] = np.nan
    return x, nan


def _butter(x: np.ndarray, fs: float, btype: str, cutoff, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    wn = np.atleast_1d(cutoff) / nyq
    if np.any(wn >= 1.0) or np.any(wn <= 0):
        return x  # filter not meaningful at this sampling rate
    sos = sps.butter(order, wn if wn.size > 1 else wn[0], btype=btype, output="sos")
    padlen = 3 * (2 * len(sos) + 1)
    if x.size <= padlen:
        return x
    finite = np.isfinite(x)
    if not finite.all():
        return x
    return sps.sosfiltfilt(sos, x)


def _median(x: np.ndarray, fs: float, seconds: float) -> np.ndarray:
    k = max(int(seconds * fs) // 2 * 2 + 1, 3)
    if x.size < k:
        return x
    return sps.medfilt(x, kernel_size=k)


def preprocess(name: str, values: np.ndarray, fs: float) -> np.ndarray:
    """Return a 1-D cleaned signal for the given modality."""
    name = name.upper()
    values = np.asarray(values, dtype=float)
    if name == "ACC":
        if values.ndim == 2 and values.shape[1] == 3:
            x = np.linalg.norm(values / 64.0, axis=1)  # magnitude in g
        else:
            x = values.ravel() / 64.0
    else:
        x = values[:, 0] if values.ndim == 2 else values

    if x.size == 0:
        return x
    x, was_nan = _interpolate_gaps(x, max_gap=int(5 * fs))
    if np.isnan(x).any():
        # Long gaps: filter each finite segment separately would be ideal; for
        # simplicity we skip filtering and rely on window-level quality gating.
        out = x
    elif name == "EDA":
        out = _butter(x, fs, "lowpass", 1.0)
    elif name == "BVP":
        out = _butter(x, fs, "bandpass", [0.5, 8.0], order=2)
    elif name in ("TEMP", "HR"):
        out = _median(x, fs, 5.0)
    elif name == "ACC":
        out = _butter(x, fs, "lowpass", 5.0)
    else:
        out = x
    out = np.asarray(out, dtype=float)
    out[was_nan] = np.nan
    return out
