"""Sensor-specific data-quality checks.

Plausibility ranges are deliberately wide: they flag values that are physically
implausible for an Empatica E4 worn on the wrist (e.g. sensor not in contact),
not values that are "abnormal" for a person. They are documented in docs/.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

# (min, max) plausible values per signal; None = no bound.
PLAUSIBLE_RANGES: dict[str, tuple[float | None, float | None]] = {
    "EDA": (0.0, 60.0),  # microsiemens; E4 range is 0.01-100 uS, >60 is almost always artefact
    "TEMP": (20.0, 42.0),  # deg C at the skin surface; <20 suggests the sensor is not on skin
    "HR": (30.0, 220.0),  # bpm
    "BVP": (-2000.0, 2000.0),  # arbitrary PPG units; extreme values = motion/clipping artefact
    "ACC": (-128.0, 128.0),  # 1/64 g, E4 range is +-2 g = +-128
}

# A signal is "flat" when its rolling std is ~0 for this many seconds (sensor disconnected / dock on).
FLATLINE_SECONDS = 30.0


@dataclass
class SignalQuality:
    signal: str
    n_samples: int
    fs: float
    duration_s: float
    missing_fraction: float
    out_of_range_fraction: float
    flatline_fraction: float
    constant_signal: bool
    status: str  # "good" | "degraded" | "unusable"
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def invalid_mask(name: str, values: np.ndarray) -> np.ndarray:
    """Boolean mask (n_samples,) of samples that are missing or outside plausible ranges."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    mask = np.isnan(values).any(axis=1)
    lo, hi = PLAUSIBLE_RANGES.get(name.upper(), (None, None))
    with np.errstate(invalid="ignore"):
        if lo is not None:
            mask |= (values < lo).any(axis=1)
        if hi is not None:
            mask |= (values > hi).any(axis=1)
    return mask


def flatline_mask(values: np.ndarray, fs: float, seconds: float = FLATLINE_SECONDS, tol: float = 1e-6) -> np.ndarray:
    """Mark samples inside runs of identical values longer than `seconds`."""
    values = np.asarray(values, dtype=float)
    if values.ndim > 1:
        values = values[:, 0]
    n = values.size
    mask = np.zeros(n, dtype=bool)
    min_run = max(int(seconds * fs), 2)
    if n < min_run:
        return mask
    diffs = np.abs(np.diff(values)) <= tol
    run_start = 0
    for i in range(1, n):
        if not diffs[i - 1]:
            if i - run_start >= min_run:
                mask[run_start:i] = True
            run_start = i
    if n - run_start >= min_run:
        mask[run_start:n] = True
    return mask


def assess_signal(name: str, values: np.ndarray, fs: float, issues: list[str] | None = None) -> SignalQuality:
    """Summarise the quality of a whole signal."""
    values = np.asarray(values, dtype=float)
    n = int(values.shape[0]) if values.size else 0
    issues = list(issues or [])
    if n == 0:
        return SignalQuality(name, 0, fs, 0.0, 1.0, 0.0, 0.0, False, "unusable", issues + ["no samples"])

    missing = float(np.isnan(values).any(axis=1).mean()) if values.ndim > 1 else float(np.isnan(values).mean())
    bad = invalid_mask(name, values)
    out_of_range = float(bad.mean() - missing) if bad.mean() > missing else 0.0
    flat = flatline_mask(values, fs)
    flat_frac = float(flat.mean())
    finite = values[np.isfinite(values).all(axis=1)] if values.ndim > 1 else values[np.isfinite(values)]
    constant = bool(finite.size > 0 and np.nanstd(finite) < 1e-9)

    if missing > 0:
        issues.append(f"{missing:.1%} missing samples")
    if out_of_range > 0:
        issues.append(f"{out_of_range:.1%} samples outside plausible range {PLAUSIBLE_RANGES.get(name.upper())}")
    if flat_frac > 0:
        issues.append(f"{flat_frac:.1%} samples in flat-line segments (> {FLATLINE_SECONDS:.0f} s)")
    if constant:
        issues.append("signal is constant")

    unusable_frac = float((bad | flat).mean())
    if constant or unusable_frac > 0.5:
        status = "unusable"
    elif unusable_frac > 0.05:
        status = "degraded"
    else:
        status = "good"
    return SignalQuality(name, n, fs, n / fs, missing, out_of_range, flat_frac, constant, status, issues)


def window_valid_fraction(name: str, values: np.ndarray, fs: float) -> float:
    """Fraction of valid samples within a window (used to accept/reject windows)."""
    if values.size == 0:
        return 0.0
    bad = invalid_mask(name, values)
    return float(1.0 - bad.mean())
