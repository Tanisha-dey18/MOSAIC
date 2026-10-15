"""Cross-validation leakage rules and metric correctness."""
import numpy as np
import pandas as pd
import pytest

from ml.evaluation.cross_validation import eligibility, make_folds, mask_modalities, run_cv
from ml.evaluation.metrics import (
    bootstrap_ci, event_sensitivity, false_alarms_per_hour, paired_comparison, summarize, threshold_metrics,
)
from ml.experiments.train import train_and_evaluate
from ml.features.extract import FEATURE_NAMES


def test_folds_are_disjoint_and_complete():
    pids = [f"P{i}" for i in range(11)]
    folds = make_folds(pids, 5, seed=1)
    flat = [p for f in folds for p in f]
    assert sorted(flat) == sorted(pids) and len(set(flat)) == len(flat)
    assert make_folds(pids, 5, seed=1) == folds  # reproducible


def test_eligibility_excludes_documented_sessions(features):
    eligible, reasons = eligibility(features, FEATURE_NAMES, 10)
    assert "S02" not in eligible and "S02" in reasons
    assert len(eligible) == 6


def test_run_cv_has_no_participant_leakage(features, exp_cfg):
    res = run_cv(features, ["robust_z_personal", "iforest_global"], FEATURE_NAMES, exp_cfg)
    pred = res.predictions
    # each participant appears in exactly one test fold
    assert pred.groupby("participant_id")["fold"].nunique().max() == 1
    for fi, test_pids in enumerate(res.folds):
        assert set(pred[pred["fold"] == fi]["participant_id"]) <= set(test_pids)
    # identical evaluation windows for all methods
    assert pred["score__robust_z_personal"].notna().sum() == pred["score__iforest_global"].notna().sum()
    assert set(pred["label"].unique()) <= {0.0, 1.0}


def test_sensor_dropout_masks_only_post_calibration(features):
    masked = mask_modalities(features, ["HR"])
    before = masked["start_s"] < masked["calibration_end_s"]
    assert masked.loc[~before & masked["calibration_end_s"].notna(), "hr_mean"].isna().all()
    assert masked.loc[before, "hr_mean"].notna().any()


def test_threshold_metrics_known_values():
    y = np.array([0, 0, 1, 1, 1])
    s = np.array([0.1, 0.9, 0.8, 0.2, 0.7])
    m = threshold_metrics(y, s, 0.5)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 1, 1, 1)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)


def test_event_and_false_alarm_metrics():
    df = pd.DataFrame({"session_id": ["a"] * 6, "phase": ["T", "T", "R", "R", "U", "U"],
                       "label": [1, 1, 0, 0, 1, 1], "flag": [True, False, True, False, False, False]})
    assert event_sensitivity(df, "flag") == 0.5
    assert false_alarms_per_hour(df, "flag", step_s=10) == pytest.approx(1 / (2 * 10 / 3600))


def test_summary_bootstrap_and_paired(features, exp_cfg):
    res = run_cv(features, ["robust_z_global", "robust_z_personal"], FEATURE_NAMES, exp_cfg)
    p = res.predictions
    s = summarize(p, "score__robust_z_personal", "thr__robust_z_personal", 10)
    assert 0 <= s["roc_auc"] <= 1 and s["n_participants"] == 6
    ci = bootstrap_ci(p, "score__robust_z_personal", "thr__robust_z_personal", n_boot=30)
    assert ci["roc_auc"][0] <= ci["roc_auc"][1]
    c = paired_comparison(p, "score__robust_z_global", "score__robust_z_personal", n_boot=30)
    assert c["n_participants"] == 6 and len(c["pooled_auc_diff_ci95"]) == 2


def test_train_and_evaluate_writes_valid_artifact(features, exp_cfg, tmp_path):
    from ml.models.detectors import Detector
    res = train_and_evaluate(features[~features["excluded"]], "iforest_personal", exp_cfg, artifact_dir=tmp_path / "a")
    assert res["evaluation"]["metrics"]["n_participants"] == 6
    det = Detector.load(tmp_path / "a")
    assert det.threshold == pytest.approx(res["threshold"])


def test_train_without_labels_reports_no_evaluation(features, exp_cfg):
    df = features[~features["excluded"]].copy()
    df["label"] = np.nan
    df["is_eval"] = False
    res = train_and_evaluate(df, "robust_z_personal", exp_cfg)
    assert res["evaluation"]["kind"] == "none"
