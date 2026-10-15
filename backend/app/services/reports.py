"""Generate a self-contained HTML research report from measured results only.

The report embeds: dataset provenance, methodology, the results of a chosen
experiment run (tables + figures, all read from results.json), the models
trained through the API with their stored evaluation metrics, and limitations.
Nothing is hard-coded: if no experiment run exists, the report says so.
"""
from __future__ import annotations

import base64
import hashlib
import html
import math
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.database.models import Dataset, EvaluationResult, ModelVersion, ResearchReport
from app.services import experiments

CSS = """
body{font-family:Segoe UI,Helvetica,Arial,sans-serif;max-width:1050px;margin:32px auto;padding:0 20px;color:#1f2933;line-height:1.55}
h1{color:#134e6f;margin-bottom:4px}h2{color:#134e6f;border-bottom:2px solid #d9e2ec;padding-bottom:4px;margin-top:36px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0}th,td{border:1px solid #d9e2ec;padding:5px 7px;text-align:left}
th{background:#f0f4f8}.note{background:#fff8e6;border-left:4px solid #f0b429;padding:10px 14px;margin:14px 0}
img{max-width:100%;border:1px solid #d9e2ec;margin:8px 0}.muted{color:#627d98;font-size:13px}code{background:#f0f4f8;padding:1px 4px}
"""


def _f(v, n=3):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{n}f}"
    return html.escape(str(v))


def _table(headers: list[str], rows: list[list]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{_f(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _img(run_id: str, name: str | None, caption: str) -> str:
    if not name:
        return ""
    try:
        data = experiments.figure_path(run_id, name).read_bytes()
    except Exception:
        return f"<p class='muted'>Figure {html.escape(name)} not available.</p>"
    return (f"<figure><img alt='{html.escape(caption)}' src='data:image/png;base64,{base64.b64encode(data).decode()}'/>"
            f"<figcaption class='muted'>{html.escape(caption)}</figcaption></figure>")


def _metrics_rows(metrics: dict) -> list[list]:
    rows = []
    for m, s in metrics.items():
        ci = s.get("ci95") or {}
        auc_ci = ci.get("roc_auc") or [None, None]
        rows.append([m, s.get("roc_auc"), f"[{_f(auc_ci[0])}, {_f(auc_ci[1])}]", s.get("pr_auc"),
                     s.get("participant_auc_median"), s.get("precision"), s.get("recall"), s.get("f1"),
                     s.get("false_alarms_per_hour"), s.get("event_sensitivity")])
    return rows


def _experiment_section(run: dict) -> str:
    rid = run["run_id"]
    figs = run.get("figures", {})
    parts = [f"<h2>Results (experiment run <code>{html.escape(rid)}</code>)</h2>",
             f"<p class='muted'>Generated {html.escape(str(run.get('generated_at')))}; "
             f"{len(run['eligibility']['eligible'])} eligible participants; {run.get('n_eval_windows')} evaluation windows "
             f"({run['label_counts']['stressor']} stressor, {run['label_counts']['rest']} rest). "
             f"Ineligible: {html.escape(', '.join(f'{k} ({v})' for k, v in run['eligibility']['ineligible'].items()) or 'none')}.</p>",
             "<div class='note'>Labels are protocol phases (stressor task vs rest) used as a proxy for deviation from a "
             "resting state, not clinical anomaly annotations. Confidence intervals use a participant-level bootstrap.</div>"]
    hdr = ["Method", "ROC-AUC", "95% CI", "PR-AUC", "Median participant AUC", "Precision", "Recall", "F1",
           "False alarms/h", "Event sensitivity"]
    parts.append("<h3>E1 Primary comparison (pre-declared)</h3>" + _table(hdr, _metrics_rows(run["primary"]["metrics"])))
    parts.append("<h3>E2 Secondary analysis (post hoc, arousal-directed scoring)</h3>"
                 "<p>This analysis was added after inspecting E1 and must be read as exploratory.</p>"
                 + _table(hdr, _metrics_rows(run["secondary"]["metrics"])))
    parts.append(_img(rid, figs.get("roc_pr_primary"), "ROC and precision-recall curves, primary methods"))
    parts.append(_img(rid, figs.get("roc_pr_secondary"), "ROC and precision-recall curves, secondary methods"))
    parts.append("<h3>Paired comparisons on identical windows</h3>")
    parts.append(_table(["A", "B", "Pooled AUC diff (B-A)", "95% CI", "Median participant AUC diff", "B better in",
                         "Wilcoxon p"],
                        [[c["a"], c["b"], c["pooled_auc_diff"], f"[{_f(c['pooled_auc_diff_ci95'][0])}, {_f(c['pooled_auc_diff_ci95'][1])}]",
                          c["participant_auc_diff_median"], f"{c['n_participants_b_better']}/{c['n_participants']}",
                          (c.get("wilcoxon") or {}).get("p_value")] for c in run["paired"]]))
    parts.append("<p class='muted'>No multiple-comparison correction applied (Bonferroni threshold for 8 tests: 0.00625).</p>")
    parts.append(_img(rid, figs.get("participant_auc"), "Per-participant ROC-AUC"))
    parts.append(_img(rid, figs.get("score_distributions"), "Score distributions for rest and stressor windows"))
    parts.append(_img(rid, figs.get("threshold_sensitivity"), "Threshold sensitivity"))
    parts.append(_img(rid, figs.get("confusion"), "Confusion matrices at the calibrated threshold"))
    abl = run.get("modality_ablation", [])
    if abl:
        parts.append("<h3>E4 Modality ablation</h3>" + _table(
            ["Feature set", "Method", "#features", "ROC-AUC", "Median participant AUC", "F1"],
            [[r["feature_set"], r["method"], r["n_features"], r["roc_auc"], r["participant_auc_median"], r["f1"]] for r in abl]))
        parts.append(_img(rid, figs.get("modality_ablation"), "Modality ablation"))
    cal = run.get("calibration_length", [])
    if cal:
        parts.append("<h3>E5 Calibration length</h3>" + _table(
            ["Calibration (s)", "Method", "ROC-AUC", "Median participant AUC", "F1", "False alarms/h"],
            [[r["calibration_s"], r["method"], r["roc_auc"], r["participant_auc_median"], r["f1"], r["false_alarms_per_hour"]] for r in cal]))
    dr = run.get("dropout", [])
    if dr:
        parts.append("<h3>E6 Sensor drop-out after calibration</h3>" + _table(
            ["Scenario", "Method", "ROC-AUC", "F1"], [[r["scenario"], r["method"], r["roc_auc"], r["f1"]] for r in dr]))
    sr = run.get("self_report", {})
    if sr.get("available"):
        parts.append("<h3>E7 Agreement with self-reported stress (exploratory)</h3>" + _table(
            ["Method", "n", "Pooled Spearman rho", "p", "Median within-participant rho"],
            [[m, v["n_pairs"], v["pooled_spearman_rho"], v["pooled_p_value"], v["within_participant_rho_median"]]
             for m, v in sr["methods"].items()]))
    parts.append("<h3>E9 Computational cost</h3>" + _table(
        ["Method", "Mean fit time per fold (s)", "Inference (ms/window)"],
        [[m, t["fit_seconds_mean"], t["infer_ms_per_window"]] for m, t in run.get("timing", {}).items()]))
    parts.append(f"<p class='muted'>Environment: {html.escape(run.get('environment', ''))}</p>")
    return "\n".join(parts)


def generate(db: Session, title: str, run_id: str | None, include_models: bool = True) -> ResearchReport:
    settings = get_settings()
    ds = db.scalars(select(Dataset).where(Dataset.status == "ready").order_by(Dataset.id.desc())).first()
    run = experiments.get_run(run_id) if run_id else None
    now = datetime.now(timezone.utc)
    body = [f"<h1>{html.escape(title)}</h1>",
            "<p class='muted'>MOSAIC - Multimodal Observation System for Anomaly Identification and Clinical Trends. "
            f"Report generated {now.isoformat()}.</p>",
            "<div class='note'><b>Research use only.</b> MOSAIC is not a medical device, is not clinically validated and "
            "must not be used for diagnosis, prediction of medical emergencies or treatment decisions.</div>",
            "<h2>Research question</h2><p>Does personalised baseline modelling improve physiological anomaly detection "
            "compared with global statistical thresholds and non-personalised machine-learning methods? This is treated "
            "as a hypothesis to be tested, not an established result.</p>"]
    if ds is not None:
        man = ds.manifest or {}
        meta = man.get("dataset", {})
        body.append("<h2>Dataset and provenance</h2>" + _table(
            ["Field", "Value"],
            [["Name", ds.name], ["Version", ds.version], ["DOI", ds.doi], ["License", ds.license],
             ["Citation", meta.get("citation")], ["Sessions / participants / windows",
                                                  f"{ds.n_sessions} / {ds.n_participants} / {ds.n_windows}"],
             ["Processed at", man.get("created_at")], ["Feature configuration", man.get("feature_config")]]))
    else:
        body.append("<h2>Dataset</h2><p>No dataset has been ingested.</p>")
    body.append("""<h2>Methodology (summary)</h2>
<ul>
<li>Signals: Empatica E4 EDA (4 Hz), BVP (64 Hz), HR (1 Hz), skin temperature (4 Hz), 3-axis accelerometer (32 Hz).</li>
<li>Signal-specific preprocessing (EDA 1 Hz low-pass; BVP 0.5-8 Hz band-pass; TEMP/HR median filter; ACC magnitude with 5 Hz low-pass); implausible values and flat-lines are masked.</li>
<li>30 s windows with a 10 s hop; 14 interpretable features; a modality's features are discarded when &lt;80% of its samples in the window are valid.</li>
<li>Protocol phases from the dataset's button tags: stressor tasks (label 1) vs rest phases (label 0, first 60 s of each rest excluded).</li>
<li>Personal baseline: the participant's last 5 minutes before the first stressor (never overlapping evaluation windows), with a shrunk scale estimate.</li>
<li>Participant-grouped 5-fold cross-validation; models trained only on training participants' calibration windows; thresholds set at the 95th percentile of training participants' rest-window scores.</li>
</ul>""")
    if run is not None:
        body.append(_experiment_section(run))
    else:
        body.append("<h2>Results</h2><div class='note'>No experiment run is available, so no results are reported. "
                    "Run <code>python -m ml.experiments.run_experiments</code> or start a run from the Model Laboratory.</div>")
    if include_models:
        mvs = db.scalars(select(ModelVersion).order_by(ModelVersion.id)).all()
        if mvs:
            rows = []
            for mv in mvs:
                ev = db.scalars(select(EvaluationResult).where(EvaluationResult.model_version_id == mv.id)).first()
                m = (ev.metrics or {}) if ev else {}
                rows.append([f"{mv.method} v{mv.version}", mv.created_at.isoformat() if mv.created_at else None,
                             mv.threshold, m.get("roc_auc"), m.get("pr_auc"), m.get("participant_auc_median"), m.get("f1")])
            body.append("<h2>Models trained through the application</h2>" + _table(
                ["Model", "Trained", "Threshold", "ROC-AUC (CV)", "PR-AUC (CV)", "Median participant AUC", "F1"], rows))
    body.append("""<h2>Limitations</h2><ul>
<li>Stressor/rest phases are experimental proxies, not clinical anomaly labels; "anomaly" here means deviation from a resting reference.</li>
<li>Healthy young volunteers (18-30) in one laboratory; results may not generalise to other populations, devices or free-living conditions.</li>
<li>Small sample (about 33 eligible participants); confidence intervals are wide.</li>
<li>The personal baseline covers only ~5 minutes; slow physiological drift over a session affects fixed baselines.</li>
<li>The arousal-directed analysis was defined after inspecting the primary results and is exploratory.</li>
<li>Explanations describe model computations (association), not causes.</li></ul>""")
    doc = f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{CSS}</style></head><body>{''.join(body)}</body></html>"
    data = doc.encode("utf-8")
    out_dir = settings.generated_reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = ResearchReport(title=title, experiment_run=run_id, dataset_id=ds.id if ds else None, file_path="",
                         sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
                         contents={"experiment_run": run_id, "includes_models": include_models,
                                   "dataset": ds.name if ds else None})
    db.add(rep)
    db.flush()
    path = out_dir / f"mosaic_report_{rep.id}.html"
    path.write_bytes(data)
    rep.file_path = str(path.relative_to(settings.storage_dir).as_posix())
    db.commit()
    db.refresh(rep)
    return rep
