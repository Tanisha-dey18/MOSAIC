"""Evaluation metrics for window-level anomaly scores against protocol-phase labels.

Positive class = window inside a stressor task (label 1); negative = rest (label 0).
Labels are protocol-derived proxies, not clinical anomaly annotations.

Because windows overlap and windows from one participant are correlated,
confidence intervals are obtained with a *participant-level* (cluster) bootstrap:
participants are resampled with replacement and all their windows come along.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve


def _safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def _safe_ap(y: np.ndarray, s: np.ndarray) -> float:
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def threshold_metrics(y: np.ndarray, s: np.ndarray, threshold: float | np.ndarray) -> dict:
    ok = np.isfinite(s)
    y, s = y[ok].astype(int), s[ok]
    thr = threshold[ok] if isinstance(threshold, np.ndarray) else threshold
    pred = (s > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall and np.isfinite(precision) else 0.0
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    return {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn), "precision": float(precision),
            "recall": float(recall), "f1": float(f1), "fpr": float(fpr), "specificity": float(1 - fpr)}


def false_alarms_per_hour(pred_df: pd.DataFrame, flagged_col: str, step_s: float) -> float:
    """Flagged rest windows per hour of rest monitoring (each window advances by `step_s`)."""
    rest = pred_df[pred_df["label"] == 0]
    hours = len(rest) * step_s / 3600.0
    if hours == 0:
        return float("nan")
    return float(rest[flagged_col].sum() / hours)


def event_sensitivity(pred_df: pd.DataFrame, flagged_col: str) -> float:
    """Fraction of stressor *events* (one phase of one session) with >= 1 flagged window."""
    pos = pred_df[pred_df["label"] == 1]
    if pos.empty:
        return float("nan")
    hits = pos.groupby(["session_id", "phase"])[flagged_col].any()
    return float(hits.mean())


def per_participant_auc(pred_df: pd.DataFrame, score_col: str) -> pd.Series:
    out = {}
    for pid, g in pred_df.groupby("participant_id"):
        out[pid] = _safe_auc(g["label"].to_numpy(int), g[score_col].to_numpy(float))
    return pd.Series(out, name=score_col)


def summarize(pred_df: pd.DataFrame, score_col: str, thr_col: str, step_s: float) -> dict:
    y = pred_df["label"].to_numpy(int)
    s = pred_df[score_col].to_numpy(float)
    thr = pred_df[thr_col].to_numpy(float)
    flagged = np.isfinite(s) & (s > thr)
    tmp = pred_df.assign(_flag=flagged)
    pp = per_participant_auc(pred_df, score_col)
    res = {
        "n_windows": int(np.isfinite(s).sum()), "n_unscored": int((~np.isfinite(s)).sum()),
        "n_participants": int(pred_df["participant_id"].nunique()),
        "prevalence": float(y.mean()) if len(y) else float("nan"),
        "roc_auc": _safe_auc(y, s), "pr_auc": _safe_ap(y, s),
        "participant_auc_median": float(np.nanmedian(pp)) if pp.notna().any() else float("nan"),
        "participant_auc_iqr": [float(np.nanpercentile(pp, 25)), float(np.nanpercentile(pp, 75))] if pp.notna().any() else None,
        "false_alarms_per_hour": false_alarms_per_hour(tmp, "_flag", step_s),
        "event_sensitivity": event_sensitivity(tmp, "_flag"),
    }
    res.update(threshold_metrics(y, s, thr))
    return res


def bootstrap_ci(pred_df: pd.DataFrame, score_col: str, thr_col: str, n_boot: int = 1000, seed: int = 42,
                 alpha: float = 0.05) -> dict:
    """Participant-level bootstrap CIs for ROC-AUC, PR-AUC and F1."""
    rng = np.random.default_rng(seed)
    groups = {pid: g for pid, g in pred_df.groupby("participant_id")}
    pids = np.array(list(groups))
    vals = {"roc_auc": [], "pr_auc": [], "f1": []}
    for _ in range(n_boot):
        sample = pd.concat([groups[p] for p in rng.choice(pids, len(pids), replace=True)], ignore_index=True)
        y, s = sample["label"].to_numpy(int), sample[score_col].to_numpy(float)
        vals["roc_auc"].append(_safe_auc(y, s))
        vals["pr_auc"].append(_safe_ap(y, s))
        vals["f1"].append(threshold_metrics(y, s, sample[thr_col].to_numpy(float))["f1"])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: [float(np.nanpercentile(v, lo)), float(np.nanpercentile(v, hi))] for k, v in vals.items()}


def paired_comparison(pred_df: pd.DataFrame, col_a: str, col_b: str, n_boot: int = 1000, seed: int = 42) -> dict:
    """Compare two methods on the *same* windows.

    * Wilcoxon signed-rank test on per-participant ROC-AUC (b - a).
    * Participant-level paired bootstrap CI of the pooled ROC-AUC difference (b - a).
    """
    a, b = per_participant_auc(pred_df, col_a), per_participant_auc(pred_df, col_b)
    both = pd.concat([a, b], axis=1).dropna()
    diff = both.iloc[:, 1] - both.iloc[:, 0]
    if len(diff) >= 5 and (diff != 0).any():
        w = stats.wilcoxon(diff)
        wilcoxon = {"statistic": float(w.statistic), "p_value": float(w.pvalue)}
    else:
        wilcoxon = None
    rng = np.random.default_rng(seed)
    groups = {pid: g for pid, g in pred_df.groupby("participant_id")}
    pids = np.array(list(groups))
    deltas = []
    for _ in range(n_boot):
        sample = pd.concat([groups[p] for p in rng.choice(pids, len(pids), replace=True)], ignore_index=True)
        y = sample["label"].to_numpy(int)
        deltas.append(_safe_auc(y, sample[col_b].to_numpy(float)) - _safe_auc(y, sample[col_a].to_numpy(float)))
    y = pred_df["label"].to_numpy(int)
    return {
        "a": col_a, "b": col_b, "n_participants": int(len(diff)),
        "pooled_auc_diff": _safe_auc(y, pred_df[col_b].to_numpy(float)) - _safe_auc(y, pred_df[col_a].to_numpy(float)),
        "pooled_auc_diff_ci95": [float(np.nanpercentile(deltas, 2.5)), float(np.nanpercentile(deltas, 97.5))],
        "participant_auc_diff_median": float(diff.median()) if len(diff) else float("nan"),
        "n_participants_b_better": int((diff > 0).sum()),
        "wilcoxon": wilcoxon,
    }


def curves(pred_df: pd.DataFrame, score_col: str) -> dict:
    ok = np.isfinite(pred_df[score_col].to_numpy(float))
    y = pred_df["label"].to_numpy(int)[ok]
    s = pred_df[score_col].to_numpy(float)[ok]
    if len(np.unique(y)) < 2:
        return {}
    fpr, tpr, _ = roc_curve(y, s)
    prec, rec, _ = precision_recall_curve(y, s)

    def thin(*arrs, n=200):
        idx = np.unique(np.linspace(0, len(arrs[0]) - 1, min(n, len(arrs[0]))).astype(int))
        return [a[idx].round(4).tolist() for a in arrs]

    fpr, tpr = thin(fpr, tpr)
    prec, rec = thin(prec, rec)
    return {"roc": {"fpr": fpr, "tpr": tpr}, "pr": {"precision": prec, "recall": rec}}
