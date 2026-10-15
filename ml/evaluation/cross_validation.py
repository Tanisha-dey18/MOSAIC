"""Participant-grouped cross-validation without participant or temporal leakage.

Protocol
--------
1. Eligible participants: >= `min_calibration_windows` complete calibration
   windows and >= 1 stressor and >= 1 rest evaluation window.
2. Participants are shuffled (seeded) and split into k folds. A participant's
   windows are never in both the training and the test side of a fold.
3. In each fold every method is trained on the *training participants'*
   calibration windows only. The alert threshold is the q-quantile of scores
   on the training participants' rest windows.
4. Test participants are scored. Personal baselines for a test participant
   come only from that participant's own calibration period, which ends
   before the first stressor and never overlaps an evaluation window.
5. All methods are evaluated on exactly the same test windows.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ml.config import ExperimentConfig
from ml.features.extract import FEATURE_MODALITY
from ml.models.detectors import Detector, parse_method

THRESHOLD_GRID = (0.80, 0.85, 0.90, 0.95, 0.975, 0.99)
PRED_META = ["participant_id", "session_id", "protocol_version", "window_index", "start_s", "end_s", "phase", "label",
             "acc_std"]


@dataclass
class CVResult:
    predictions: pd.DataFrame
    folds: list[list[str]]
    eligible: list[str]
    ineligible: dict[str, str]
    timing: dict[str, dict] = field(default_factory=dict)
    threshold_grid: dict[str, list[dict]] = field(default_factory=dict)  # method -> per-fold {q: thr}
    histories: dict[str, list] = field(default_factory=dict)


def eligibility(df: pd.DataFrame, features: list[str], min_cal: int,
                require_labels: bool = True) -> tuple[list[str], dict[str, str]]:
    eligible, reasons = [], {}
    for pid, g in df.groupby("participant_id"):
        if g["excluded"].all():
            reasons[str(pid)] = "session excluded (see data_constraints.txt)"
            continue
        n_cal = int(g[g["is_calibration"]].dropna(subset=features).shape[0])
        ev = g[g["is_eval"]]
        if n_cal < min_cal:
            reasons[str(pid)] = f"insufficient complete calibration windows ({n_cal} < {min_cal})"
        elif require_labels and (not (ev["label"] == 1).any() or not (ev["label"] == 0).any()):
            reasons[str(pid)] = "no labelled stressor/rest evaluation windows"
        else:
            eligible.append(str(pid))
    return sorted(eligible), reasons


def make_folds(pids: list[str], n_folds: int, seed: int) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(sorted(pids)))
    k = min(n_folds, len(shuffled))
    return [sorted(shuffled[i::k]) for i in range(k)]


def mask_modalities(df: pd.DataFrame, modalities: list[str]) -> pd.DataFrame:
    """Simulate sensor drop-out *after* calibration: set those modalities' features to NaN in evaluation windows."""
    df = df.copy()
    cols = [f for f, m in FEATURE_MODALITY.items() if set(m.split("+")) & set(modalities) and f in df]
    after = df["start_s"] >= df["calibration_end_s"].fillna(np.inf)
    df.loc[after, cols] = np.nan
    return df


def run_cv(df: pd.DataFrame, methods: list[str], features: list[str], cfg: ExperimentConfig,
           params: dict[str, dict] | None = None, drop_modalities: list[str] | None = None,
           eligible: list[str] | None = None, progress=None) -> CVResult:
    params = params or {}
    df = df[~df["excluded"]].copy()
    if eligible is None:
        eligible, ineligible = eligibility(df, features, cfg.features.min_calibration_windows)
    else:
        ineligible = {}
    if len(eligible) < 2:
        raise ValueError(f"need at least 2 eligible participants for cross-validation, found {len(eligible)}")
    df = df[df["participant_id"].isin(eligible)]
    folds = make_folds(eligible, cfg.n_folds, cfg.seed)
    preds = []
    timing = {m: {"fit_seconds": [], "infer_seconds_per_window": []} for m in methods}
    grid = {m: [] for m in methods}
    histories = {m: [] for m in methods}
    for fi, test_pids in enumerate(folds):
        train = df[~df["participant_id"].isin(test_pids)]
        test = df[df["participant_id"].isin(test_pids)]
        if drop_modalities:
            test = mask_modalities(test, drop_modalities)
        train_pids = sorted(train["participant_id"].unique())
        rng = np.random.default_rng(cfg.seed + fi)
        val_pids = set(rng.choice(train_pids, max(1, len(train_pids) // 5), replace=False))
        test_eval = test[test["is_eval"]]
        fold_pred = test_eval[[c for c in PRED_META if c in test_eval]].copy()
        fold_pred["fold"] = fi
        for m in methods:
            scorer, norm = parse_method(m)
            det = Detector(scorer, norm, features, cfg.features, params.get(scorer), seed=cfg.seed)
            if scorer == "autoencoder":
                det.params.setdefault("epochs", cfg.autoencoder_epochs)
                det.fit(train[~train["participant_id"].isin(val_pids)], train[train["participant_id"].isin(val_pids)])
                histories[m].append(det.train_info.get("history"))
            else:
                if scorer == "ocsvm":
                    det.params.setdefault("max_train", cfg.ocsvm_max_train)
                det.fit(train)
            timing[m]["fit_seconds"].append(det.train_info.get("fit_seconds", 0.0))
            train_scores = pd.Series(det.score(train), index=train.index)
            rest_mask = train["is_eval"] & (train["label"] == 0)
            rest_scores = train_scores[rest_mask].to_numpy()
            thr = det.set_threshold_from(rest_scores, cfg.threshold_quantile)
            finite = rest_scores[np.isfinite(rest_scores)]
            grid[m].append({str(q): (float(np.quantile(finite, q)) if scorer != "hr_rule" else thr) for q in THRESHOLD_GRID})
            t0 = time.perf_counter()
            test_scores = pd.Series(det.score(test), index=test.index)
            timing[m]["infer_seconds_per_window"].append((time.perf_counter() - t0) / max(len(test), 1))
            fold_pred[f"score__{m}"] = test_scores.loc[test_eval.index].to_numpy()
            fold_pred[f"thr__{m}"] = thr
            if progress:
                progress(fi, m)
        preds.append(fold_pred)
    predictions = pd.concat(preds, ignore_index=True)
    timing_summary = {m: {"fit_seconds_mean": float(np.mean(t["fit_seconds"])),
                          "infer_ms_per_window": float(np.mean(t["infer_seconds_per_window"]) * 1000)}
                      for m, t in timing.items()}
    return CVResult(predictions, folds, eligible, ineligible, timing_summary, grid, histories)
