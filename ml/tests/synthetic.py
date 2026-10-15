"""SYNTHETIC TEST DATA GENERATOR - for automated software tests only.

Writes a small directory tree in the *format* of the PhysioNet wearable dataset
(Empatica CSV headers, tags.csv, Stress_Level files) filled with simulated
signals. These values are NOT real observations, must never be presented as
experimental results, and are not used anywhere outside the test-suite.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

START = pd.Timestamp("2020-01-01 10:00:00")
# Relative tag times (s) for the two STRESS protocol versions (stressor = between tags, see physionet_wearable).
V1_TAGS = [60, 240, 300, 420, 660, 840, 1080, 1120, 1160, 1200, 1240, 1280, 1340]
V2_TAGS = [120, 300, 480, 780, 820, 860, 900, 1200, 1240]
V1_STRESS = [(300, 420), (660, 840), (1080, 1120), (1160, 1200), (1240, 1280)]
V2_STRESS = [(300, 480), (780, 820), (860, 900), (1200, 1240)]


def _stress_mask(t: np.ndarray, spans) -> np.ndarray:
    m = np.zeros_like(t, dtype=bool)
    for a, b in spans:
        m |= (t >= a) & (t < b)
    return m


def _write_signal(path: Path, start: pd.Timestamp, fs: float, values: np.ndarray) -> None:
    values = np.atleast_2d(values.T).T if values.ndim == 1 else values
    k = values.shape[1]
    lines = [",".join([start.strftime("%Y-%m-%d %H:%M:%S")] * k), ",".join([f"{fs:.1f}"] * k)]
    lines += [",".join(f"{v:.6f}" for v in row) for row in values]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_session(folder: Path, version: str, seed: int, duration_s: int = 1500, stress_effect: float = 1.0) -> None:
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    tags = V1_TAGS if version == "V1" else V2_TAGS
    spans = V1_STRESS if version == "V1" else V2_STRESS
    hr0, eda0, temp0 = rng.uniform(62, 85), rng.uniform(0.5, 6), rng.uniform(31.5, 34)

    t4 = np.arange(duration_s * 4) / 4
    s4 = _stress_mask(t4, spans)
    eda = eda0 + 0.0005 * t4 + stress_effect * 1.2 * s4 + 0.05 * rng.standard_normal(t4.size)
    temp = temp0 + 0.0003 * t4 - stress_effect * 0.25 * s4 + 0.02 * rng.standard_normal(t4.size)
    t1 = np.arange(duration_s - 10, dtype=float) + 10
    hr = hr0 + stress_effect * 14 * _stress_mask(t1, spans) + 1.5 * rng.standard_normal(t1.size)
    t64 = np.arange(duration_s * 64) / 64
    amp = 60 * (1 - 0.3 * stress_effect * _stress_mask(t64, spans))
    bvp = amp * np.sin(2 * np.pi * (hr0 / 60) * t64) + 5 * rng.standard_normal(t64.size)
    t32 = np.arange(duration_s * 32)
    acc = np.column_stack([rng.normal(2, 1, t32.size), rng.normal(-3, 1, t32.size), rng.normal(63, 1, t32.size)])

    _write_signal(folder / "EDA.csv", START, 4.0, eda)
    _write_signal(folder / "TEMP.csv", START, 4.0, temp)
    _write_signal(folder / "HR.csv", START + pd.Timedelta(seconds=10), 1.0, hr)
    _write_signal(folder / "BVP.csv", START, 64.0, bvp)
    _write_signal(folder / "ACC.csv", START, 32.0, np.round(acc))
    (folder / "IBI.csv").write_text(f"{START.strftime('%Y-%m-%d %H:%M:%S')},IBI\n12.0,0.8\n12.8,0.81\n", encoding="utf-8")
    (folder / "tags.csv").write_text(
        "\n".join((START + pd.Timedelta(seconds=x)).strftime("%Y-%m-%d %H:%M:%S") for x in tags) + "\n", encoding="utf-8")


def write_physionet_tree(root: Path, v1=("S01", "S02", "S03", "S04"), v2=("f01", "f02", "f03"), seed: int = 0) -> Path:
    """Create <root>/Wearable_Dataset/STRESS/<subject>/... plus Stress_Level files. Returns root."""
    for i, s in enumerate(v1):
        write_session(root / "Wearable_Dataset" / "STRESS" / s, "V1", seed + i)
    for i, s in enumerate(v2):
        write_session(root / "Wearable_Dataset" / "STRESS" / s, "V2", seed + 100 + i)
    pd.DataFrame({"Baseline": 2, "Stroop": 5, "First Rest": 3, "TMCT": 6, "Second Rest": 3, "Real Opinion": 5,
                  "Opposite Opinion": 6, "Subtract": 6}, index=list(v1)).to_csv(root / "Stress_Level_v1.csv")
    pd.DataFrame({"Baseline": 2, "TMCT": 6, "First Rest": 3, "Real Opinion": 5, "Opposite Opinion": 6,
                  "Second Rest": 3, "Subtract": 6}, index=list(v2)).to_csv(root / "Stress_Level_v2.csv")
    return root


def generic_csv_frame(n_participants: int = 2, minutes: int = 12, seed: int = 0) -> pd.DataFrame:
    """Long-format CSV content (1 Hz) with a leading calibration block and 0/1 labels."""
    rng = np.random.default_rng(seed)
    frames = []
    for p in range(n_participants):
        n = minutes * 60
        t = pd.date_range("2021-05-01 08:00:00", periods=n, freq="s", tz="UTC")
        label = np.where(np.arange(n) >= n // 2, 1, 0)
        frames.append(pd.DataFrame({
            "timestamp": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "participant_id": f"P{p + 1:02d}",
            "eda": 2 + 0.8 * label + 0.05 * rng.standard_normal(n), "hr": 70 + 12 * label + rng.standard_normal(n),
            "temp": 33 + 0.02 * rng.standard_normal(n), "label": label,
            "calibration": (np.arange(n) < 300).astype(int),
        }))
    return pd.concat(frames, ignore_index=True)
