"""Matplotlib figures for the experiment report (saved as PNG)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PALETTE = ["#2a6f97", "#e07a5f", "#3d405b", "#81b29a", "#f2cc8f", "#6d597a", "#b56576", "#468faf", "#9a8c98",
           "#bc6c25", "#588157", "#c9184a", "#7f5539"]


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path.name


def roc_pr(curves: dict[str, dict], out: Path, title_suffix: str = "") -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, (m, c) in enumerate(curves.items()):
        if not c:
            continue
        col = PALETTE[i % len(PALETTE)]
        axes[0].plot(c["roc"]["fpr"], c["roc"]["tpr"], label=m, color=col, lw=1.4)
        axes[1].plot(c["pr"]["recall"], c["pr"]["precision"], label=m, color=col, lw=1.4)
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title=f"ROC curves{title_suffix}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title=f"Precision-recall curves{title_suffix}")
    axes[0].legend(fontsize=7, loc="lower right")
    return _save(fig, out)


def score_distributions(pred: pd.DataFrame, methods: list[str], out: Path) -> str:
    n = len(methods)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    for ax, m in zip(np.atleast_1d(axes).ravel(), methods):
        s = pred[f"score__{m}"]
        thr = pred[f"thr__{m}"].median()
        lo, hi = np.nanpercentile(s, [1, 99])
        bins = np.linspace(lo, hi, 40)
        ax.hist(s[pred["label"] == 0].clip(lo, hi), bins=bins, alpha=0.6, density=True, label="rest", color=PALETTE[0])
        ax.hist(s[pred["label"] == 1].clip(lo, hi), bins=bins, alpha=0.6, density=True, label="stressor", color=PALETTE[1])
        ax.axvline(thr, color="k", ls="--", lw=1, label="threshold (median over folds)")
        ax.set_title(m, fontsize=9)
        ax.set_xlabel("anomaly score")
    for ax in np.atleast_1d(axes).ravel()[n:]:
        ax.axis("off")
    np.atleast_1d(axes).ravel()[0].legend(fontsize=7)
    return _save(fig, out)


def threshold_sensitivity(table: pd.DataFrame, methods: list[str], out: Path) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, m in enumerate(methods):
        t = table[table["method"] == m].sort_values("quantile")
        col = PALETTE[i % len(PALETTE)]
        axes[0].plot(t["quantile"], t["f1"], marker="o", ms=3, label=m, color=col)
        axes[1].plot(t["quantile"], t["recall"], marker="o", ms=3, color=col)
        axes[2].plot(t["quantile"], t["false_alarms_per_hour"], marker="o", ms=3, color=col)
    axes[0].set(xlabel="threshold quantile q (training rest windows)", ylabel="F1", title="F1 vs threshold")
    axes[1].set(xlabel="q", ylabel="recall (window level)", title="Recall vs threshold")
    axes[2].set(xlabel="q", ylabel="false alarms per hour of rest", title="False alarms vs threshold")
    axes[0].legend(fontsize=7)
    return _save(fig, out)


def participant_auc(pp: pd.DataFrame, out: Path) -> str:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    data = [pp[c].dropna().to_numpy() for c in pp.columns]
    ax.boxplot(data, showfliers=True)
    ax.set_xticks(range(1, len(pp.columns) + 1), pp.columns, rotation=40, ha="right", fontsize=8)
    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    ax.set(ylabel="ROC-AUC within participant", title="Per-participant ROC-AUC (stressor vs rest windows)")
    return _save(fig, out)


def paired_lines(pp: pd.DataFrame, a: str, b: str, out: Path) -> str:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    both = pp[[a, b]].dropna()
    for _, r in both.iterrows():
        ax.plot([0, 1], [r[a], r[b]], color="grey", alpha=0.5, lw=0.8)
    ax.scatter(np.zeros(len(both)), both[a], color=PALETTE[0], s=12)
    ax.scatter(np.ones(len(both)), both[b], color=PALETTE[1], s=12)
    ax.set_xticks([0, 1], [a, b], fontsize=8)
    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    ax.set(ylabel="per-participant ROC-AUC", title="Paired comparison")
    return _save(fig, out)


def grouped_bars(table: pd.DataFrame, x: str, hue: str, y: str, out: Path, title: str, ylabel: str) -> str:
    piv = table.pivot(index=x, columns=hue, values=y)
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(piv)), 4.5))
    piv.plot(kind="bar", ax=ax, color=PALETTE[: len(piv.columns)], width=0.8)
    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    ax.set(title=title, ylabel=ylabel, xlabel="")
    ax.legend(fontsize=7)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return _save(fig, out)


def line_plot(table: pd.DataFrame, x: str, hue: str, y: str, out: Path, title: str, xlabel: str, ylabel: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (m, g) in enumerate(table.groupby(hue)):
        g = g.sort_values(x)
        ax.plot(g[x], g[y], marker="o", label=m, color=PALETTE[i % len(PALETTE)])
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.legend(fontsize=7)
    return _save(fig, out)


def confusion_grid(summaries: dict[str, dict], out: Path) -> str:
    n = len(summaries)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.8 * rows))
    for ax, (m, s) in zip(np.atleast_1d(axes).ravel(), summaries.items()):
        mat = np.array([[s["tn"], s["fp"]], [s["fn"], s["tp"]]])
        ax.imshow(mat, cmap="Blues")
        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=9,
                    color="white" if v > mat.max() / 2 else "black")
        ax.set_xticks([0, 1], ["pred rest", "pred flag"], fontsize=7)
        ax.set_yticks([0, 1], ["rest", "stressor"], fontsize=7)
        ax.set_title(m, fontsize=8)
    for ax in np.atleast_1d(axes).ravel()[n:]:
        ax.axis("off")
    return _save(fig, out)


def example_timeline(pred: pd.DataFrame, pid: str, methods: list[str], out: Path) -> str:
    g = pred[pred["participant_id"] == pid].sort_values("start_s")
    fig, axes = plt.subplots(len(methods), 1, figsize=(12, 2.2 * len(methods)), sharex=True)
    for ax, m in zip(np.atleast_1d(axes), methods):
        ax.scatter(g["start_s"] / 60, g[f"score__{m}"], c=np.where(g["label"] == 1, PALETTE[1], PALETTE[0]), s=8)
        ax.axhline(g[f"thr__{m}"].iloc[0], color="k", ls="--", lw=0.8)
        ax.set_ylabel(m, fontsize=7)
    np.atleast_1d(axes)[-1].set_xlabel("minutes since session start (orange = stressor window, blue = rest)")
    np.atleast_1d(axes)[0].set_title(f"Out-of-fold anomaly scores for participant {pid}")
    return _save(fig, out)
