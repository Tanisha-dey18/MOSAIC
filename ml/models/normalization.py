"""Reference baselines used to normalise features before anomaly scoring.

Three normalisation references are compared (this is the experimental variable
for the personalisation research question):

* ``global``   - one population median/scale learned from the *training participants'*
                 calibration windows. Between-person differences remain in the data.
* ``personal`` - each participant's own median/scale from their calibration period
                 (the last `calibration_s` seconds before the first stressor). Only past
                 data of that participant is used, so there is no temporal leakage.
* ``rolling``  - an adaptive personal baseline: median of the participant's windows
                 that ended within the preceding `rolling_s` seconds (strictly past).

Personal scales are estimated from only ~10 effective windows, so they are shrunk
towards the typical within-person scale (s_w) of the training participants:

    s_p^2 = (n_eff * mad_p^2 + k * s_w^2) / (n_eff + k)

where n_eff = n_windows * step / window corrects for overlapping windows and
k (`prior_windows`) is the prior strength. This is a simple empirical-Bayes-style
shrinkage that prevents division by near-zero scales.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ml.config import FeatureConfig

MAD_TO_SD = 1.4826  # makes the MAD a consistent estimator of the SD for Gaussian data
EPS = 1e-6


def robust_center_scale(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column-wise median and MAD*1.4826, ignoring NaNs."""
    with np.errstate(all="ignore"):
        center = np.nanmedian(X, axis=0)
        mad = np.nanmedian(np.abs(X - center), axis=0) * MAD_TO_SD
    return np.nan_to_num(center), np.nan_to_num(mad)


@dataclass
class ParticipantBaseline:
    participant_id: str
    status: str  # "ok" | "insufficient"
    n_windows: int
    center: list[float]
    scale: list[float]
    session_id: str | None = None
    period_start_s: float | None = None
    period_end_s: float | None = None
    period_start_time: str | None = None
    period_end_time: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReferenceStats:
    """Statistics learned from training participants only."""

    features: list[str]
    global_center: list[float]
    global_scale: list[float]
    within_scale: list[float]  # median of training participants' personal MADs (s_w)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ReferenceStats":
        return cls(**d)


def complete_calibration_rows(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    cal = df[df["is_calibration"].astype(bool)]
    return cal.dropna(subset=features)


def fit_reference(train: pd.DataFrame, features: list[str], cfg: FeatureConfig) -> ReferenceStats:
    """Learn global and within-person reference statistics from training participants."""
    cal = complete_calibration_rows(train, features)
    if cal.empty:
        raise ValueError("no complete calibration windows in the training data")
    center, scale = robust_center_scale(cal[features].to_numpy(float))
    per_person = []
    for _, g in cal.groupby("participant_id"):
        if len(g) >= cfg.min_calibration_windows:
            per_person.append(robust_center_scale(g[features].to_numpy(float))[1])
    within = np.nanmedian(np.vstack(per_person), axis=0) if per_person else scale
    scale = np.maximum(scale, EPS)
    within = np.maximum(within, EPS)
    return ReferenceStats(features, center.tolist(), scale.tolist(), within.tolist())


def shrunk_scale(mad: np.ndarray, n: int, ref: ReferenceStats, cfg: FeatureConfig, prior_windows: float) -> np.ndarray:
    n_eff = n * min(cfg.step_s / cfg.window_s, 1.0)
    s_w = np.asarray(ref.within_scale)
    var = (n_eff * mad**2 + prior_windows * s_w**2) / (n_eff + prior_windows)
    return np.maximum(np.sqrt(var), EPS)


def build_baselines(df: pd.DataFrame, ref: ReferenceStats, cfg: FeatureConfig,
                    prior_windows: float = 5.0) -> dict[str, ParticipantBaseline]:
    """Personal baselines for every participant present in `df` (uses calibration windows only)."""
    features = ref.features
    out: dict[str, ParticipantBaseline] = {}
    for pid, g in df.groupby("participant_id"):
        cal = complete_calibration_rows(g, features)
        any_cal = g[g["is_calibration"].astype(bool)]
        if len(cal) < cfg.min_calibration_windows:
            reason = (f"{len(cal)} complete calibration windows (< {cfg.min_calibration_windows} required)"
                      if len(any_cal) else "no calibration period before the first stressor")
            if len(any_cal) and len(cal) < len(any_cal):
                reason += "; some modalities missing or low quality"
            out[str(pid)] = ParticipantBaseline(str(pid), "insufficient", int(len(cal)), [], [], reason=reason)
            continue
        X = cal[features].to_numpy(float)
        center, mad = robust_center_scale(X)
        scale = shrunk_scale(mad, len(cal), ref, cfg, prior_windows)
        out[str(pid)] = ParticipantBaseline(
            str(pid), "ok", int(len(cal)), center.tolist(), scale.tolist(),
            session_id=str(cal["session_id"].iloc[0]),
            period_start_s=float(cal["start_s"].min()), period_end_s=float(cal["end_s"].max()),
            period_start_time=str(cal.loc[cal["start_s"].idxmin(), "start_time"]),
            period_end_time=str(pd.Timestamp(cal.loc[cal["end_s"].idxmax(), "start_time"])
                                + pd.Timedelta(seconds=cfg.window_s)),
        )
    return out


def normalize(df: pd.DataFrame, mode: str, ref: ReferenceStats, cfg: FeatureConfig,
              baselines: dict[str, ParticipantBaseline] | None = None,
              rolling_s: float = 300.0, prior_windows: float = 5.0) -> np.ndarray:
    """Return Z (n, d). Rows that cannot be normalised (no baseline) are all-NaN.

    Individual missing features stay NaN; scorers treat them as "no evidence".
    """
    features = ref.features
    X = df[features].to_numpy(float)
    if mode == "global":
        return (X - np.asarray(ref.global_center)) / np.asarray(ref.global_scale)
    if mode == "personal":
        if baselines is None:
            raise ValueError("personal normalisation requires baselines")
        Z = np.full_like(X, np.nan)
        pids = df["participant_id"].astype(str).to_numpy()
        for pid in np.unique(pids):
            b = baselines.get(pid)
            if b is None or b.status != "ok":
                continue
            m = pids == pid
            Z[m] = (X[m] - np.asarray(b.center)) / np.asarray(b.scale)
        return Z
    if mode == "rolling":
        Z = np.full_like(X, np.nan)
        positions = {idx: k for k, idx in enumerate(df.index)}
        for _, g in df.groupby("session_id"):
            g = g.sort_values("start_s")
            starts, ends = g["start_s"].to_numpy(), g["end_s"].to_numpy()
            Xg = g[features].to_numpy(float)
            for j in range(len(g)):
                past = (ends <= starts[j]) & (ends > starts[j] - rolling_s)
                Xp = Xg[past]
                Xp = Xp[~np.isnan(Xp).any(axis=1)]
                if len(Xp) < cfg.min_calibration_windows // 2:
                    continue
                c, mad = robust_center_scale(Xp)
                s = shrunk_scale(mad, len(Xp), ref, cfg, prior_windows)
                Z[positions[g.index[j]]] = (Xg[j] - c) / s
        return Z
    raise ValueError(f"unknown normalisation mode: {mode}")
