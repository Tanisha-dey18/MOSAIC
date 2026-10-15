import { useEffect, useState } from "react";
import { ApiError, api } from "../lib/api";
import { useApi } from "../lib/hooks";
import { dateTime, fmt, methodLabel } from "../lib/format";
import { Badge, Card, ErrorState, Field, Loading, Notice, PageHeader } from "../components/ui";

const FIGURE_TITLES: Record<string, string> = {
  roc_pr_primary: "ROC / PR curves - primary methods",
  roc_pr_secondary: "ROC / PR curves - secondary analysis",
  participant_auc: "Per-participant ROC-AUC",
  paired_robust_z: "Paired: robust z global vs personal",
  paired_arousal_z: "Paired: arousal z global vs personal",
  score_distributions: "Score distributions (rest vs stressor)",
  threshold_sensitivity: "Threshold sensitivity",
  modality_ablation: "Modality ablation",
  calibration_length: "Calibration-length ablation",
  dropout: "Sensor drop-out robustness",
  confusion: "Confusion matrices",
  example_timeline: "Example participant timeline",
};

export default function Reports() {
  const runs = useApi(() => api.experiments(), []);
  const reports = useApi(() => api.reports(), []);
  const completed = (runs.data?.runs ?? []).filter((r) => r.status === "completed");
  const [runId, setRunId] = useState("");
  const [title, setTitle] = useState("MOSAIC research report");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { if (!runId && completed.length) setRunId(completed[0].run_id); }, [completed, runId]);
  const exp = useApi(() => (runId ? api.experiment(runId) : Promise.resolve(null)), [runId]);

  const generate = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true); setErr(null);
    try { await api.createReport(title, runId || undefined); reports.reload(); } catch (e2) { setErr(e2 instanceof ApiError ? e2.message : String(e2)); } finally { setBusy(false); }
  };

  return (
    <div className="space-y-6">
      <PageHeader title="Research reports" subtitle="Generate a downloadable report containing the measured experiment results, figures, dataset provenance, methodology and limitations. Reports contain only results produced by the pipeline." />
      <div className="grid gap-6 lg:grid-cols-3">
        <Card title="Generate report">
          <form className="space-y-4" onSubmit={generate}>
            <Field label="Title" htmlFor="rt"><input id="rt" className="input" value={title} minLength={3} maxLength={200} onChange={(e) => setTitle(e.target.value)} /></Field>
            <Field label="Experiment run" htmlFor="rr" hint={completed.length ? undefined : "No completed run: the report will state that no results are available."}>
              <select id="rr" className="input" value={runId} onChange={(e) => setRunId(e.target.value)}>
                <option value="">None</option>
                {completed.map((r) => <option key={r.run_id} value={r.run_id}>{r.run_id}</option>)}
              </select>
            </Field>
            <button className="btn-primary w-full" disabled={busy}>{busy ? "Generating…" : "Generate HTML report"}</button>
            {err && <div className="rounded-lg bg-rose-50 p-3 text-sm text-rose-800">{err}</div>}
          </form>
        </Card>
        <Card title="Generated reports" className="lg:col-span-2">
          {reports.loading ? <Loading /> : reports.error ? <ErrorState error={reports.error} /> : reports.data!.items.length === 0 ? <p className="text-sm text-ink-soft">No reports generated yet.</p> : (
            <div className="overflow-x-auto"><table className="table-base">
              <thead><tr><th>#</th><th>Title</th><th>Experiment</th><th>Size</th><th>Created</th><th /></tr></thead>
              <tbody>
                {reports.data!.items.map((r) => (
                  <tr key={r.id}>
                    <td>{r.id}</td><td>{r.title}</td><td className="text-xs">{r.experiment_run ?? "-"}</td><td>{(r.bytes / 1024).toFixed(0)} KB</td><td className="text-xs">{dateTime(r.created_at)}</td>
                    <td><a className="btn-secondary px-2 py-1 text-xs" href={api.reportUrl(r.id)} download>Download</a></td>
                  </tr>
                ))}
              </tbody>
            </table></div>
          )}
        </Card>
      </div>

      {runs.loading ? <Loading /> : exp.data && (
        <>
          <Card title={`Experiment ${exp.data.run_id}`} subtitle={`${exp.data.dataset} · ${exp.data.eligibility.eligible.length} participants · runtime ${Math.round(exp.data.runtime_seconds / 60)} min`}>
            <Notice>Labels are protocol phases (stressor vs rest) used as a proxy, not clinical annotations. The E2 arousal-directed analysis was defined after inspecting E1 and is exploratory.</Notice>
            <div className="mt-4 grid gap-6 lg:grid-cols-2">
              <div className="min-w-0">
                <h3 className="mb-2 text-sm font-semibold">E4 · Modality ablation (median participant AUC)</h3>
                <AblationTable rows={exp.data.modality_ablation} />
              </div>
              <div className="min-w-0 space-y-6">
                <div>
                  <h3 className="mb-2 text-sm font-semibold">E5 · Calibration length (pooled ROC-AUC)</h3>
                  <Pivot rows={exp.data.calibration_length.map((r) => ({ row: `${r.calibration_s} s`, col: r.method, v: r.roc_auc }))} />
                </div>
                <div>
                  <h3 className="mb-2 text-sm font-semibold">E6 · Sensor drop-out after calibration (pooled ROC-AUC)</h3>
                  <Pivot rows={exp.data.dropout.map((r) => ({ row: r.scenario, col: r.method, v: r.roc_auc }))} />
                </div>
              </div>
            </div>
            {exp.data.self_report.available && exp.data.self_report.methods && (
              <div className="mt-6">
                <h3 className="mb-2 text-sm font-semibold">E7 · Agreement with self-reported stress (exploratory)</h3>
                <div className="overflow-x-auto"><table className="table-base">
                  <thead><tr><th>Method</th><th>n (participant × phase)</th><th>Pooled Spearman ρ</th><th>p</th><th>Median within-participant ρ</th></tr></thead>
                  <tbody>
                    {Object.entries(exp.data.self_report.methods).map(([m, v]) => (
                      <tr key={m}><td>{methodLabel(m)}</td><td>{v.n_pairs}</td><td>{fmt(v.pooled_spearman_rho)}</td><td>{fmt(v.pooled_p_value, 4)}</td><td>{fmt(v.within_participant_rho_median)}</td></tr>
                    ))}
                  </tbody>
                </table></div>
              </div>
            )}
          </Card>
          <Card title="Figures" subtitle="Generated by the experiment pipeline (matplotlib)">
            <div className="grid gap-6 lg:grid-cols-2">
              {Object.entries(exp.data.figures).map(([key, name]) => (
                <figure key={key} className="rounded-lg border border-slate-200 p-2">
                  <img src={api.figureUrl(exp.data!.run_id, name)} alt={FIGURE_TITLES[key] ?? key} loading="lazy" className="w-full" />
                  <figcaption className="mt-1 text-xs text-ink-soft">{FIGURE_TITLES[key] ?? key} <Badge tone="pending">{name}</Badge></figcaption>
                </figure>
              ))}
            </div>
          </Card>
        </>
      )}
    </div>
  );
}

function Pivot({ rows }: { rows: { row: string; col: string; v: number | null }[] }) {
  const cols = [...new Set(rows.map((r) => r.col))];
  const rowKeys = [...new Set(rows.map((r) => r.row))];
  return (
    <div className="overflow-x-auto">
      <table className="table-base">
        <thead><tr><th /> {cols.map((c) => <th key={c}>{methodLabel(c)}</th>)}</tr></thead>
        <tbody>
          {rowKeys.map((rk) => (
            <tr key={rk}><td className="font-medium">{rk}</td>{cols.map((c) => <td key={c}>{fmt(rows.find((r) => r.row === rk && r.col === c)?.v ?? null)}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function AblationTable({ rows }: { rows: { feature_set: string; method: string; participant_auc_median: number }[] }) {
  return <Pivot rows={rows.map((r) => ({ row: r.feature_set, col: r.method, v: r.participant_auc_median }))} />;
}
