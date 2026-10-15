"""Detectors, personal baselines, serialisation and artifact validation."""
import json

import numpy as np
import pytest

from ml.config import FeatureConfig
from ml.features.extract import FEATURE_NAMES
from ml.models.detectors import ArtifactError, Detector, ModelNotReadyError, parse_method
from ml.models.normalization import build_baselines, fit_reference, normalize

CFG = FeatureConfig()


def _train(features):
    return features[~features["excluded"]]


@pytest.mark.parametrize("method", ["robust_z_global", "robust_z_personal", "robust_z_rolling", "arousal_z_personal",
                                    "iforest_global", "iforest_personal", "ocsvm_global", "ocsvm_personal",
                                    "autoencoder_global", "autoencoder_personal", "hr_rule"])
def test_fit_and_score_all_methods(features, method):
    scorer, norm = parse_method(method)
    det = Detector(scorer, norm, FEATURE_NAMES, CFG, {"epochs": 10} if scorer == "autoencoder" else None)
    data = _train(features)
    det.fit(data)
    s = det.score(data)
    assert s.shape == (len(data),)
    assert np.isfinite(s[data["is_eval"].to_numpy()]).mean() > 0.9
    c = det.contributions(data)
    if scorer == "hr_rule":
        assert c is None
    else:
        assert c.shape == (len(data), len(FEATURE_NAMES))


def test_untrained_model_refuses_to_score(features):
    det = Detector("iforest", "global", FEATURE_NAMES, CFG)
    with pytest.raises(ModelNotReadyError):
        det.score(features)
    with pytest.raises(ModelNotReadyError):
        det.save("unused")


def test_personal_baseline_uses_only_calibration_windows(features):
    data = _train(features)
    ref = fit_reference(data, FEATURE_NAMES, CFG)
    b1 = build_baselines(data, ref, CFG)
    tampered = data.copy()
    tampered.loc[~tampered["is_calibration"], FEATURE_NAMES] *= 100  # change only future/evaluation data
    b2 = build_baselines(tampered, ref, CFG)
    assert b1["S01"].center == b2["S01"].center and b1["S01"].scale == b2["S01"].scale
    assert b1["S01"].period_end_s <= 300


def test_insufficient_history_is_reported_and_not_scored(features):
    data = _train(features).copy()
    data.loc[data["participant_id"] == "S03", "is_calibration"] = False
    det = Detector("robust_z", "personal", FEATURE_NAMES, CFG).fit(data[data["participant_id"] != "S03"])
    b = build_baselines(data, det.ref, CFG)
    assert b["S03"].status == "insufficient" and "no calibration period" in b["S03"].reason
    scores = det.score(data)
    assert np.isnan(scores[(data["participant_id"] == "S03").to_numpy()]).all()


def test_robust_z_contributions_sum_to_squared_score(features):
    data = _train(features)
    det = Detector("robust_z", "global", FEATURE_NAMES, CFG).fit(data)
    s, c = det.score(data), det.contributions(data)
    ok = np.isfinite(s)
    full = data[FEATURE_NAMES].notna().all(axis=1).to_numpy() & ok
    assert np.allclose(c[full].sum(axis=1), s[full] ** 2, rtol=1e-6)


def test_arousal_direction_is_one_sided(features):
    data = _train(features)
    det = Detector("arousal_z", "personal", FEATURE_NAMES, CFG).fit(data)
    m = data["is_eval"].to_numpy()
    # Shift only evaluation windows; the personal baseline (calibration windows) stays the same.
    lowered = data.copy()
    lowered.loc[m, "hr_mean"] -= 30  # strong *decrease* in heart rate
    raised = data.copy()
    raised.loc[m, "hr_mean"] += 30
    assert np.nanmean(det.score(raised)[m]) > np.nanmean(det.score(lowered)[m])


def test_global_vs_personal_normalisation_differ(features):
    data = _train(features)
    ref = fit_reference(data, FEATURE_NAMES, CFG)
    zg = normalize(data, "global", ref, CFG)
    zp = normalize(data, "personal", ref, CFG, build_baselines(data, ref, CFG))
    cal = data["is_calibration"].to_numpy()
    # Personal z of a participant's own calibration windows is centred at 0.
    assert abs(np.nanmedian(zp[cal][:, FEATURE_NAMES.index("hr_mean")])) < 0.2
    assert not np.allclose(np.nan_to_num(zg), np.nan_to_num(zp))


def test_deterministic_with_seed(features):
    data = _train(features)
    a = Detector("iforest", "global", FEATURE_NAMES, CFG, seed=7).fit(data).score(data)
    b = Detector("iforest", "global", FEATURE_NAMES, CFG, seed=7).fit(data).score(data)
    assert np.allclose(a, b, equal_nan=True)


@pytest.mark.parametrize("method", ["robust_z_personal", "iforest_global", "ocsvm_personal", "autoencoder_global", "hr_rule"])
def test_save_load_roundtrip(features, tmp_path, method):
    scorer, norm = parse_method(method)
    data = _train(features)
    det = Detector(scorer, norm, FEATURE_NAMES, CFG, {"epochs": 5} if scorer == "autoencoder" else None).fit(data)
    det.set_threshold_from(det.score(data), 0.95)
    det.save(tmp_path / method)
    loaded = Detector.load(tmp_path / method)
    assert loaded.threshold == det.threshold and loaded.features == det.features
    assert np.allclose(loaded.score(data), det.score(data), equal_nan=True)


def test_missing_and_corrupted_artifacts_are_rejected(features, tmp_path):
    data = _train(features)
    det = Detector("iforest", "global", FEATURE_NAMES, CFG).fit(data)
    det.save(tmp_path / "m")
    with pytest.raises(ArtifactError):
        Detector.load(tmp_path / "does_not_exist")
    blob = (tmp_path / "m" / "model.bin").read_bytes()
    (tmp_path / "m" / "model.bin").write_bytes(blob[:-10] + b"0123456789")
    with pytest.raises(ArtifactError, match="checksum"):
        Detector.load(tmp_path / "m")
    (tmp_path / "m" / "model.bin").unlink()
    with pytest.raises(ArtifactError, match="missing"):
        Detector.load(tmp_path / "m")


def test_feature_schema_tampering_is_rejected(features, tmp_path):
    det = Detector("robust_z", "global", FEATURE_NAMES, CFG).fit(_train(features))
    det.save(tmp_path / "m")
    meta = json.loads((tmp_path / "m" / "metadata.json").read_text())
    meta["features"] = list(reversed(meta["features"]))
    (tmp_path / "m" / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ArtifactError, match="hash"):
        Detector.load(tmp_path / "m")
    (tmp_path / "m" / "metadata.json").write_text("{not json")
    with pytest.raises(ArtifactError, match="corrupted"):
        Detector.load(tmp_path / "m")


def test_constant_features_do_not_break_scaling(features):
    data = _train(features).copy()
    data["acc_mean"] = 1.0
    data["acc_std"] = 0.0
    det = Detector("ocsvm", "personal", FEATURE_NAMES, CFG).fit(data)
    assert np.isfinite(det.score(data)[data["is_eval"].to_numpy()]).all()


def test_autoencoder_early_stopping_history(features):
    det = Detector("autoencoder", "global", FEATURE_NAMES, CFG, {"epochs": 300, "patience": 3})
    data = _train(features)
    det.fit(data[data["participant_id"] != "f03"], data[data["participant_id"] == "f03"])
    h = det.train_info["history"]
    assert len(h["val_loss"]) <= 300 and h["best_epoch"] <= len(h["val_loss"])


def test_feature_table_sorts_and_compares_with_reference():
    from ml.explainability.contributions import feature_table
    feats = ["eda_mean", "hr_mean", "temp_mean"]
    table = feature_table(feats, [3.0, 90.0, float("nan")], contributions=[0.1, 0.9, 0.0],
                          center=[2.0, 70.0, 33.0], scale=[0.5, 5.0, 0.2])
    assert [r["feature"] for r in table][:2] == ["hr_mean", "eda_mean"]
    assert table[0]["z"] == pytest.approx(4.0) and table[0]["contribution_share"] == pytest.approx(0.9)
    nan_row = next(r for r in table if r["feature"] == "temp_mean")
    assert nan_row["value"] is None and nan_row["z"] is None  # missing value -> no fabricated z
