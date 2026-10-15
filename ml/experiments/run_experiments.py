"""Reproducible research experiment suite for MOSAIC.

    python -m ml.experiments.run_experiments --features-dir storage/processed/physionet_wearable_1.0.1

Analyses (all on the STRESS protocol, participant-grouped 5-fold CV, seed 42):

  E1  PRIMARY (configuration fixed before any result was seen):
      hr_rule, {robust_z, iforest, ocsvm, autoencoder} x {global, personal}, robust_z_rolling
  E2  SECONDARY / POST HOC (motivated by E1 diagnostics):
      arousal_z x {global, personal, rolling}
  E3  Threshold sensitivity (q = 0.80 ... 0.99)
  E4  Modality ablation (single modality and leave-one-modality-out)
  E5  Calibration-length ablation for personal baselines (1-5 min)
  E6  Robustness to sensor drop-out after calibration
  E7  Agreement with self-reported stress (Spearman), exploratory
  E8  Error analysis (detection rate by phase; motion in false positives)
  E9  Computational cost (fit time, inference latency)

Outputs: <out>/results.json, predictions.parquet, summary.md, figures/*.png
Nothing in this script contains or hard-codes results.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from scipy import stats

from ml import __version__ as ML_VERSION
from ml.config import ExperimentConfig
from ml.evaluation import plots
from ml.evaluation.cross_validation import THRESHOLD_GRID, eligibility, run_cv
from ml.evaluation.metrics import (
    bootstrap_ci, curves, event_sensitivity, false_alarms_per_hour, paired_comparison, per_participant_auc, summarize,
    threshold_metrics,
)
from ml.features.extract import FEATURE_NAMES, features_for_modalities
from ml.models.detectors import AROUSAL_DIRECTION
from ml.pipeline import load_features

PRIMARY = ["hr_rule", "robust_z_global", "robust_z_personal", "robust_z_rolling", "iforest_global", "iforest_personal",
           "ocsvm_global", "ocsvm_personal", "autoencoder_global", "autoencoder_personal"]
SECONDARY = ["arousal_z_global", "arousal_z_personal", "arousal_z_rolling"]
PAIRS = [("robust_z_global", "robust_z_personal"), ("iforest_global", "iforest_personal"),
         ("ocsvm_global", "ocsvm_personal"), ("autoencoder_global", "autoencoder_personal"),
         ("robust_z_global", "robust_z_rolling"), ("arousal_z_global", "arousal_z_personal"),
         ("arousal_z_personal", "arousal_z_rolling"), ("hr_rule", "arousal_z_personal")]
MODALITY_SETS = {"all": ["EDA", "HR", "TEMP", "BVP", "ACC"], "EDA only": ["EDA"], "HR only": ["HR"],
                 "TEMP only": ["TEMP"], "BVP only": ["BVP"], "ACC only": ["ACC"],
                 "without EDA": ["HR", "TEMP", "BVP", "ACC"], "without HR": ["EDA", "TEMP", "BVP", "ACC"],
                 "without TEMP": ["EDA", "HR", "BVP", "ACC"], "without BVP": ["EDA", "HR", "TEMP", "ACC"],
                 "without ACC": ["EDA", "HR", "TEMP", "BVP"]}
ABLATION_METHODS = ["robust_z_global", "robust_z_personal", "iforest_global", "iforest_personal",
                    "arousal_z_global", "arousal_z_personal"]
DROPOUT_SCENARIOS = {"EDA lost": ["EDA"], "PPG lost (HR+BVP)": ["HR", "BVP"], "TEMP lost": ["TEMP"], "ACC lost": ["ACC"]}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _r(x, n=4):
    if isinstance(x, float):
        return None if np.isnan(x) else round(x, n)
    if isinstance(x, dict):
        return {k: _r(v, n) for k, v in x.items()}
    if isinstance(x, list):
        return [_r(v, n) for v in x]
    return x


def method_table(pred: pd.DataFrame, methods: list[str], cfg: ExperimentConfig, ci: bool = True) -> dict:
    out = {}
    for m in methods:
        s = summarize(pred, f"score__{m}", f"thr__{m}", cfg.features.step_s)
        if ci:
            s["ci95"] = bootstrap_ci(pred, f"score__{m}", f"thr__{m}", cfg.bootstrap_iterations, cfg.seed)
        out[m] = s
    return out


def threshold_table(pred: pd.DataFrame, grid: dict[str, list[dict]], methods: list[str], step_s: float) -> pd.DataFrame:
    rows = []
    for m in methods:
        for q in THRESHOLD_GRID:
            thr = pred["fold"].map(lambda f: grid[m][f][str(q)]).to_numpy(float)
            s = pred[f"score__{m}"].to_numpy(float)
            flag = np.isfinite(s) & (s > thr)
            tm = threshold_metrics(pred["label"].to_numpy(int), s, thr)
            tmp = pred.assign(_f=flag)
            rows.append({"method": m, "quantile": q, **{k: tm[k] for k in ("precision", "recall", "f1", "fpr")},
                         "false_alarms_per_hour": false_alarms_per_hour(tmp, "_f", step_s),
                         "event_sensitivity": event_sensitivity(tmp, "_f")})
    return pd.DataFrame(rows)


def self_report_analysis(pred: pd.DataFrame, levels: pd.DataFrame, methods: list[str]) -> dict:
    if levels.empty:
        return {"available": False}
    out = {"available": True, "note": "Baseline phase excluded (it is the calibration period). Exploratory only.",
           "methods": {}}
    for m in methods:
        ph = pred.groupby(["participant_id", "phase"])[f"score__{m}"].mean().reset_index()
        merged = ph.merge(levels, on=["participant_id", "phase"]).dropna()
        if len(merged) < 10:
            continue
        rho, p = stats.spearmanr(merged[f"score__{m}"], merged["level"])
        within = [stats.spearmanr(g[f"score__{m}"], g["level"])[0] for _, g in merged.groupby("participant_id")
                  if len(g) >= 4 and g["level"].nunique() > 1]
        out["methods"][m] = {"n_pairs": int(len(merged)), "pooled_spearman_rho": float(rho), "pooled_p_value": float(p),
                             "within_participant_rho_median": float(np.nanmedian(within)) if within else None,
                             "n_participants_within": len(within)}
    return out


def error_analysis(pred: pd.DataFrame, methods: list[str]) -> dict:
    out = {}
    for m in methods:
        s, thr = pred[f"score__{m}"], pred[f"thr__{m}"]
        flag = (s > thr) & s.notna()
        by_phase = pred.assign(_f=flag).groupby(["protocol_version", "phase"])["_f"].mean().round(4)
        fp = pred[flag & (pred["label"] == 0)]
        tn = pred[~flag & (pred["label"] == 0)]
        pp = per_participant_auc(pred, f"score__{m}").dropna().sort_values()
        out[m] = {
            "flag_rate_by_phase": {f"{v} {p}": float(r) for (v, p), r in by_phase.items()},
            "median_acc_std_false_positives": float(fp["acc_std"].median()) if len(fp) else None,
            "median_acc_std_true_negatives": float(tn["acc_std"].median()) if len(tn) else None,
            "worst_participants": {k: round(float(v), 3) for k, v in pp.head(3).items()},
            "best_participants": {k: round(float(v), 3) for k, v in pp.tail(3).items()},
        }
    return out


def md_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a Markdown table (no extra dependency)."""
    head = "| " + " | ".join([str(df.index.name or "")] + [str(c) for c in df.columns]) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    body = ["| " + " | ".join([str(i)] + ["n/a" if pd.isna(v) else str(v) for v in row]) + " |"
            for i, row in zip(df.index, df.to_numpy())]
    return "\n".join([head, sep, *body])


def write_summary(res: dict, out: Path) -> None:
    L = []
    L.append("# MOSAIC experiment results\n")
    L.append(f"Run: `{res['run_id']}`  |  Generated: {res['generated_at']}  |  Dataset: {res['dataset']}\n")
    L.append("Labels are protocol phases (stressor task = 1, rest = 0) used as a proxy for deviation from a resting "
             "state; they are not clinical anomaly annotations. Scores are not probabilities.\n")
    el = res["eligibility"]
    L.append(f"Eligible participants: {len(el['eligible'])}. Ineligible: "
             + (", ".join(f"{k} ({v})" for k, v in el["ineligible"].items()) or "none") + "\n")
    for title, key in (("E1 Primary comparison (pre-declared)", "primary"), ("E2 Secondary, post-hoc (arousal-directed)", "secondary")):
        L.append(f"\n## {title}\n")
        L.append("| Method | ROC-AUC [95% CI] | PR-AUC [95% CI] | Median participant AUC | Precision | Recall | F1 | False alarms / h | Event sensitivity |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for m, s in res[key]["metrics"].items():
            ci = s.get("ci95", {})
            fmt = lambda v: "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"
            L.append(f"| {m} | {fmt(s['roc_auc'])} [{fmt(ci.get('roc_auc', [None])[0])}, {fmt(ci.get('roc_auc', [None, None])[1])}] "
                     f"| {fmt(s['pr_auc'])} [{fmt(ci.get('pr_auc', [None])[0])}, {fmt(ci.get('pr_auc', [None, None])[1])}] "
                     f"| {fmt(s['participant_auc_median'])} | {fmt(s['precision'])} | {fmt(s['recall'])} | {fmt(s['f1'])} "
                     f"| {fmt(s['false_alarms_per_hour'])} | {fmt(s['event_sensitivity'])} |")
        L.append(f"\nPrevalence of stressor windows (PR-AUC chance level): {list(res[key]['metrics'].values())[0]['prevalence']:.3f}\n")
    L.append("\n## Paired comparisons (same windows)\n")
    L.append("| A | B | Pooled AUC diff (B-A) [95% CI] | Median participant AUC diff | B better in | Wilcoxon p |")
    L.append("|---|---|---|---|---|---|")
    for c in res["paired"]:
        w = c["wilcoxon"]["p_value"] if c["wilcoxon"] else float("nan")
        L.append(f"| {c['a']} | {c['b']} | {c['pooled_auc_diff']:+.3f} [{c['pooled_auc_diff_ci95'][0]:+.3f}, "
                 f"{c['pooled_auc_diff_ci95'][1]:+.3f}] | {c['participant_auc_diff_median']:+.3f} | "
                 f"{c['n_participants_b_better']}/{c['n_participants']} | {w:.4f} |")
    L.append("\nNo correction for multiple comparisons is applied; with 8 comparisons, a Bonferroni threshold would be p < 0.00625.\n")
    L.append("\n## E4 Modality ablation (pooled ROC-AUC / median participant AUC)\n")
    abl = pd.DataFrame(res["modality_ablation"])
    if not abl.empty:
        L.append(abl.pivot(index="feature_set", columns="method", values="roc_auc").round(3).pipe(md_table))
        L.append("\nMedian per-participant AUC:\n")
        L.append(abl.pivot(index="feature_set", columns="method", values="participant_auc_median").round(3).pipe(md_table))
    L.append("\n\n## E5 Calibration length (personal baselines)\n")
    cal = pd.DataFrame(res["calibration_length"])
    if not cal.empty:
        L.append(cal.pivot(index="calibration_s", columns="method", values="roc_auc").round(3).pipe(md_table))
    L.append("\n\n## E6 Sensor drop-out after calibration (pooled ROC-AUC)\n")
    dr = pd.DataFrame(res["dropout"])
    if not dr.empty:
        L.append(dr.pivot(index="scenario", columns="method", values="roc_auc").round(3).pipe(md_table))
    L.append("\n\n## E7 Self-reported stress agreement (exploratory)\n")
    sr = res["self_report"]
    if sr.get("available"):
        L.append("| Method | n | pooled Spearman rho | p | median within-participant rho |")
        L.append("|---|---|---|---|---|")
        for m, v in sr["methods"].items():
            w = v["within_participant_rho_median"]
            L.append(f"| {m} | {v['n_pairs']} | {v['pooled_spearman_rho']:.3f} | {v['pooled_p_value']:.4f} | "
                     f"{'n/a' if w is None else f'{w:.3f}'} |")
    L.append("\n\n## E9 Computational cost\n")
    L.append("| Method | mean fit time per fold (s) | inference (ms / window) |")
    L.append("|---|---|---|")
    for m, t in res["timing"].items():
        L.append(f"| {m} | {t['fit_seconds_mean']:.3f} | {t['infer_ms_per_window']:.4f} |")
    L.append(f"\n\nEnvironment: {res['environment']}\n")
    (out / "summary.md").write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-dir", default="storage/processed/physionet_wearable_1.0.1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true", help="fewer bootstrap iterations / epochs (for smoke tests)")
    args = ap.parse_args()
    cfg = ExperimentConfig.load(args.config) if args.config else ExperimentConfig()
    if args.quick:
        cfg.bootstrap_iterations, cfg.autoencoder_epochs = 100, 30
    torch.set_num_threads(1)
    run_id = datetime.now(timezone.utc).strftime("exp_%Y%m%dT%H%M%SZ")
    out = Path(args.out or f"reports/experiments/{run_id}")
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    features_dir = Path(args.features_dir)
    manifest = json.loads((features_dir / "manifest.json").read_text(encoding="utf-8"))
    df = load_features(features_dir)
    df = df[(df["protocol"] == "STRESS") & (~df["excluded"])].copy()
    eligible, ineligible = eligibility(df, FEATURE_NAMES, cfg.features.min_calibration_windows)
    log(f"{len(eligible)} eligible participants; ineligible: {ineligible}")

    res: dict = {"run_id": run_id, "generated_at": datetime.now(timezone.utc).isoformat(), "ml_version": ML_VERSION,
                 "dataset": f"{manifest['dataset'].get('name')} v{manifest['dataset'].get('version')}",
                 "dataset_doi": manifest["dataset"].get("doi"), "config": cfg.to_dict(),
                 "features": FEATURE_NAMES, "arousal_direction": AROUSAL_DIRECTION,
                 "eligibility": {"eligible": eligible, "ineligible": ineligible},
                 "environment": f"Python {platform.python_version()}, scikit-learn {sklearn.__version__}, "
                                f"torch {torch.__version__}, {platform.platform()}"}

    # ---- E1 + E2 -------------------------------------------------------
    log("E1/E2: cross-validated comparison")
    methods = PRIMARY + SECONDARY
    cv = run_cv(df, methods, FEATURE_NAMES, cfg, eligible=eligible, progress=lambda f, m: None)
    pred = cv.predictions
    pred.to_parquet(out / "predictions.parquet", index=False)
    res["folds"] = cv.folds
    res["n_eval_windows"] = int(len(pred))
    res["label_counts"] = {"stressor": int((pred["label"] == 1).sum()), "rest": int((pred["label"] == 0).sum())}
    log("bootstrap CIs")
    res["primary"] = {"metrics": method_table(pred, PRIMARY, cfg)}
    res["secondary"] = {"metrics": method_table(pred, SECONDARY, cfg)}
    res["curves"] = {m: curves(pred, f"score__{m}") for m in methods}
    pp = pd.concat([per_participant_auc(pred, f"score__{m}").rename(m) for m in methods], axis=1)
    res["per_participant_auc"] = pp.round(4).to_dict(orient="index")
    res["paired"] = []
    for a, b in PAIRS:
        c = paired_comparison(pred, f"score__{a}", f"score__{b}", cfg.bootstrap_iterations, cfg.seed)
        c["a"], c["b"] = a, b
        res["paired"].append(c)
    res["timing"] = cv.timing
    res["autoencoder_histories"] = {m: cv.histories[m] for m in methods if m.startswith("autoencoder")}
    figures = {
        "roc_pr_primary": plots.roc_pr({m: res["curves"][m] for m in PRIMARY}, figs / "roc_pr_primary.png", " (E1)"),
        "roc_pr_secondary": plots.roc_pr({m: res["curves"][m] for m in SECONDARY + ["hr_rule"]},
                                         figs / "roc_pr_secondary.png", " (E2)"),
        "score_distributions": plots.score_distributions(pred, methods, figs / "score_distributions.png"),
        "participant_auc": plots.participant_auc(pp, figs / "participant_auc.png"),
        "paired_robust_z": plots.paired_lines(pp, "robust_z_global", "robust_z_personal", figs / "paired_robust_z.png"),
        "paired_arousal_z": plots.paired_lines(pp, "arousal_z_global", "arousal_z_personal", figs / "paired_arousal_z.png"),
        "confusion": plots.confusion_grid({m: (res["primary"]["metrics"] | res["secondary"]["metrics"])[m] for m in methods},
                                          figs / "confusion_matrices.png"),
    }
    example = pp["arousal_z_personal"].sub(pp["arousal_z_personal"].median()).abs().idxmin()
    figures["example_timeline"] = plots.example_timeline(pred, example, ["hr_rule", "robust_z_global", "robust_z_personal",
                                                                         "arousal_z_personal"], figs / "example_timeline.png")

    # ---- E3 ------------------------------------------------------------
    log("E3: threshold sensitivity")
    thr_tab = threshold_table(pred, cv.threshold_grid, methods, cfg.features.step_s)
    res["threshold_sensitivity"] = thr_tab.round(4).to_dict(orient="records")
    figures["threshold_sensitivity"] = plots.threshold_sensitivity(
        thr_tab, [m for m in methods if m != "hr_rule"], figs / "threshold_sensitivity.png")

    # ---- E4 ------------------------------------------------------------
    log("E4: modality ablation")
    abl_rows = []
    for name, mods in MODALITY_SETS.items():
        feats = features_for_modalities(mods)
        ms = [m for m in ABLATION_METHODS if not (m.startswith("arousal") and not any(f in AROUSAL_DIRECTION for f in feats))]
        r = run_cv(df, ms, feats, cfg, eligible=eligible)
        for m in ms:
            s = summarize(r.predictions, f"score__{m}", f"thr__{m}", cfg.features.step_s)
            abl_rows.append({"feature_set": name, "method": m, "n_features": len(feats), "roc_auc": s["roc_auc"],
                             "pr_auc": s["pr_auc"], "participant_auc_median": s["participant_auc_median"], "f1": s["f1"]})
    res["modality_ablation"] = abl_rows
    figures["modality_ablation"] = plots.grouped_bars(pd.DataFrame(abl_rows), "feature_set", "method", "participant_auc_median",
                                                      figs / "modality_ablation.png", "Modality ablation",
                                                      "median per-participant ROC-AUC")

    # ---- E5 ------------------------------------------------------------
    log("E5: calibration length")
    cal_rows = []
    for L in (60, 120, 180, 240, 300):
        d2 = df.copy()
        d2["is_calibration"] = (d2["end_s"] <= d2["calibration_end_s"]) & (d2["start_s"] >= d2["calibration_end_s"] - L)
        c2 = ExperimentConfig.from_dict(cfg.to_dict())
        c2.features.min_calibration_windows = 3
        c2.features.calibration_s = float(L)
        ms = ["robust_z_personal", "iforest_personal", "arousal_z_personal"]
        r = run_cv(d2, ms, FEATURE_NAMES, c2, eligible=eligible)
        for m in ms:
            s = summarize(r.predictions, f"score__{m}", f"thr__{m}", c2.features.step_s)
            cal_rows.append({"calibration_s": L, "method": m, "roc_auc": s["roc_auc"],
                             "participant_auc_median": s["participant_auc_median"], "f1": s["f1"],
                             "false_alarms_per_hour": s["false_alarms_per_hour"]})
    res["calibration_length"] = cal_rows
    figures["calibration_length"] = plots.line_plot(pd.DataFrame(cal_rows), "calibration_s", "method", "roc_auc",
                                                    figs / "calibration_length.png", "Calibration length",
                                                    "seconds of personal baseline", "pooled ROC-AUC")

    # ---- E6 ------------------------------------------------------------
    log("E6: sensor drop-out")
    drop_methods = ["robust_z_global", "robust_z_personal", "iforest_global", "iforest_personal", "ocsvm_personal",
                    "autoencoder_personal", "arousal_z_personal"]
    dr_rows = []
    base = method_table(pred, drop_methods, cfg, ci=False)
    for m in drop_methods:
        dr_rows.append({"scenario": "none", "method": m, "roc_auc": base[m]["roc_auc"], "f1": base[m]["f1"]})
    for name, mods in DROPOUT_SCENARIOS.items():
        r = run_cv(df, drop_methods, FEATURE_NAMES, cfg, drop_modalities=mods, eligible=eligible)
        for m in drop_methods:
            s = summarize(r.predictions, f"score__{m}", f"thr__{m}", cfg.features.step_s)
            dr_rows.append({"scenario": name, "method": m, "roc_auc": s["roc_auc"], "f1": s["f1"]})
    res["dropout"] = dr_rows
    figures["dropout"] = plots.grouped_bars(pd.DataFrame(dr_rows), "scenario", "method", "roc_auc", figs / "dropout.png",
                                            "Sensor drop-out after calibration", "pooled ROC-AUC")

    # ---- E7, E8 ----------------------------------------------------------
    log("E7/E8: self-report and error analysis")
    levels_path = features_dir / "stress_levels.csv"
    levels = pd.read_csv(levels_path) if levels_path.exists() else pd.DataFrame()
    res["self_report"] = self_report_analysis(pred, levels, methods)
    res["error_analysis"] = error_analysis(pred, methods)
    res["figures"] = figures
    res["runtime_seconds"] = round(time.perf_counter() - t_start, 1)

    clean = _r(json.loads(json.dumps(res, default=lambda o: o.item() if hasattr(o, "item") else str(o))))
    (out / "results.json").write_text(json.dumps(clean, indent=1), encoding="utf-8")
    write_summary(clean, out)
    log(f"done in {res['runtime_seconds']} s -> {out}")


if __name__ == "__main__":
    main()
