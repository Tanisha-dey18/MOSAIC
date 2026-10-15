"""Quality checks, filtering, windowing, feature generation and leakage rules."""
import numpy as np
import pandas as pd
import pytest

from ml.config import FeatureConfig
from ml.data_ingestion.empatica import Signal
from ml.data_ingestion.session import Phase, Session
from ml.features.extract import FEATURE_NAMES, extract_session_features, features_for_modalities
from ml.preprocessing.filters import preprocess
from ml.preprocessing.quality import assess_signal, flatline_mask, invalid_mask


def test_invalid_mask_ranges():
    vals = np.array([[0.5], [np.nan], [-1.0], [80.0], [3.0]])
    assert invalid_mask("EDA", vals).tolist() == [False, True, True, True, False]
    assert invalid_mask("TEMP", np.array([[15.0], [33.0]])).tolist() == [True, False]


def test_flatline_and_constant_signal():
    x = np.concatenate([np.random.default_rng(0).normal(size=40), np.ones(200)])
    assert flatline_mask(x, fs=4.0).sum() >= 120  # 30 s at 4 Hz
    q = assess_signal("TEMP", np.full((400, 1), 33.0), 4.0)
    assert q.constant_signal and q.status == "unusable"


def test_assess_empty_signal():
    q = assess_signal("EDA", np.empty((0, 1)), 4.0)
    assert q.status == "unusable" and q.missing_fraction == 1.0


def test_filters_preserve_length_and_missing_positions():
    rng = np.random.default_rng(1)
    eda = 2 + 0.1 * rng.standard_normal(400)
    eda[50:55] = np.nan
    out = preprocess("EDA", eda[:, None], 4.0)
    assert out.shape == (400,) and np.isnan(out[50:55]).all() and np.isfinite(out[:50]).all()
    acc = np.tile([0, 0, 64], (320, 1)).astype(float)
    assert np.allclose(preprocess("ACC", acc, 32.0), 1.0, atol=1e-6)  # magnitude in g


def test_feature_order_is_canonical(features):
    assert [c for c in features.columns if c in FEATURE_NAMES] == FEATURE_NAMES
    assert features_for_modalities(["EDA"]) == ["eda_mean", "eda_std", "eda_slope", "eda_scr_rate"]
    assert "eda_hr_corr" in features_for_modalities(["EDA", "HR"])


def test_windows_never_cross_phase_or_calibration_boundaries(sessions, features):
    by_id = {s.session_id: s for s in sessions}
    for _, row in features[features["label"].notna()].iterrows():
        ph = by_id[row["session_id"]].phase_at(row["start_s"], row["end_s"])
        assert ph is not None and ph.label == row["label"]
    cal, ev = features[features["is_calibration"]], features[features["is_eval"]]
    for sid, g in cal.groupby("session_id"):
        assert (g["end_s"] <= by_id[sid].calibration_end_s).all()
    for sid, g in ev.groupby("session_id"):
        assert (g["start_s"] >= by_id[sid].calibration_end_s).all()  # no temporal leakage


def test_rest_guard_excludes_recovery(sessions, features):
    cfg = FeatureConfig()
    by_id = {s.session_id: s for s in sessions}
    rest = features[(features["label"] == 0) & features["is_eval"]]
    for _, r in rest.iterrows():
        ph = by_id[r["session_id"]].phase_at(r["start_s"], r["end_s"])
        assert r["start_s"] >= ph.start_s + cfg.rest_guard_s


def test_excluded_session_has_no_training_windows(features):
    s02 = features[features["session_id"] == "STRESS_S02"]
    assert not s02.empty and not s02["is_calibration"].any() and not s02["is_eval"].any()


def _session_with(signals: dict[str, Signal], invalid=None) -> Session:
    s = Session("X", "P1", "STRESS", signals, phases=[Phase("Rest", 0, 1000, 0)], calibration_end_s=120)
    s.invalid_modalities = invalid or {}
    return s


def _sig(name, fs, values):
    return Signal(name, pd.Timestamp("2020-01-01", tz="UTC"), fs, values.reshape(len(values), -1))


def test_missing_and_invalid_modalities_give_nan_features():
    rng = np.random.default_rng(0)
    eda = _sig("EDA", 4.0, 2 + 0.05 * rng.standard_normal(4 * 300))
    hr = _sig("HR", 1.0, 70 + rng.standard_normal(300))
    feats = extract_session_features(_session_with({"EDA": eda, "HR": hr}, invalid={"HR": "test"}), FeatureConfig())
    assert feats["eda_mean"].notna().all()
    assert feats[["hr_mean", "temp_mean", "bvp_amp", "acc_mean"]].isna().all().all()
    assert (feats["q_HR"] == 0).all()


def test_low_quality_window_is_rejected():
    eda_vals = 2 + 0.01 * np.random.default_rng(0).standard_normal(4 * 120)
    eda_vals[:4 * 20] = 200.0  # implausible values for the first 20 s
    feats = extract_session_features(_session_with({"EDA": _sig("EDA", 4.0, eda_vals)}), FeatureConfig())
    first = feats.iloc[0]
    assert first["q_EDA"] < 0.8 and np.isnan(first["eda_mean"])
    assert feats.iloc[-1]["eda_mean"] == pytest.approx(2, abs=0.05)


def test_constant_signal_features_are_finite():
    feats = extract_session_features(_session_with({"TEMP": _sig("TEMP", 4.0, np.full(4 * 120, 33.0))}), FeatureConfig())
    # Flat-lines are physically valid values; std is 0 and the slope 0, nothing breaks.
    assert np.allclose(feats["temp_mean"].dropna(), 33.0)
