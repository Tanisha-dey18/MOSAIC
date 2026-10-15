import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ApiError, api } from "../lib/api";
import { useApi, usePolling } from "../lib/hooks";
import { dateTime, fmt, methodLabel } from "../lib/format";
import type { Metrics } from "../lib/types";
import { CurveChart } from "../components/charts";
import { Badge, Card, ErrorState, Field, KeyValue, Loading, Notice, PageHeader } from "../components/ui";

const MODALITIES = ["EDA", "HR", "TEMP", "BVP", "ACC"];

export default function ModelLab() {
  const methods = useApi(() => api.methods(), []);
  const runs = useApi(() => api.trainingRuns(), []);
  const models = useApi(() => api.models(), []);
  const [selected, setSelected] = useState<number | null>(null);
  const active = (runs.data?.items ?? []).some((r) => r.status === "queued" || r.status === "running");
  usePolling(() => { runs.reload(); models.reload(); }, 3000, active);
  useEffect(() => { if (selected === null && models.data?.length) setSelected(models.data[0].id); }, [models.data, selected]);

  return (
    <div className="space-y-6">
      <PageHeader title="Model laboratory" subtitle="Train detectors with participant-grouped cross-validation, inspect their measured performance and compare methods. Training runs as a background job; the API stays responsive." />
      <div className="grid gap-6 xl:grid-cols-3">
        <TrainForm methods={methods.data?.methods.map((m) => m.method) ?? []} onStarted={() => runs.reload()} />
        <Card title="Training history" className="xl:col-span-2">
          {runs.loading ? <Loading /> : runs.error ? <ErrorState error={runs.error} /> : runs.data!.items.length === 0 ? <p className="text-sm text-ink-soft">No training runs yet.</p> : (
            <div className="max-h-80 overflow-auto">
              <table className="table-base">
                <thead><tr><th>#</th><th>Method</th><th>Status</th><th>Progress</th><th>Features</th><th>Started</th><th>Finished</th></tr></thead>
                <tbody>
                  {runs.data!.items.map((r) => (
                    <tr key={r.id}>
                      <td>{r.id}</td><td>{methodLabel(r.method)}</td>
                      <td><Badge>{r.status}</Badge>{r.error && <div className="mt-1 max-w-xs text-xs text-rose-700">{r.error}</div>}</td>
                      <td className="text-xs">{r.progress ?? "-"}</td><td className="text-xs">{r.features.length}</td>
                      <td className="text-xs">{dateTime(r.started_at)}</td><td className="text-xs">{dateTime(r.finished_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </div>

      <Card title="Trained model versions">
        {models.loading ? <Loading /> : models.error ? <ErrorState error={models.error} /> : models.data!.length === 0 ? <p className="text-sm text-ink-soft">No trained models yet.</p> : (
          <div className="overflow-x-auto">
            <table className="table-base">
              <thead><tr><th>ID</th><th>Model</th><th>Version</th><th>ROC-AUC</th><th>PR-AUC</th><th>Median participant AUC</th><th>F1</th><th>False alarms/h</th><th>Threshold</th><th>Created</th></tr></thead>
              <tbody>
                {models.data!.map((m) => (
                  <tr key={m.id} className={`cursor-pointer ${selected === m.id ? "bg-brand-50" : "hover:bg-slate-50"}`} onClick={() => setSelected(m.id)}>
                    <td>{m.id}</td><td><button className="font-medium text-brand-800 hover:underline">{methodLabel(m.method)}</button></td><td>v{m.version}</td>
                    <td>{fmt(m.metrics?.roc_auc)}</td><td>{fmt(m.metrics?.pr_auc)}</td><td>{fmt(m.metrics?.participant_auc_median)}</td>
                    <td>{fmt(m.metrics?.f1)}</td><td>{fmt(m.metrics?.false_alarms_per_hour, 1)}</td><td>{fmt(m.threshold)}</td><td className="text-xs">{dateTime(m.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
      {selected !== null && <ModelDetail id={selected} />}
      <ExperimentPanel />
      <Card title="Method catalogue">
        {methods.data && (
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {methods.data.methods.map((m) => (
              <div key={m.method} className="rounded-lg border border-slate-200 p-3">
                <div className="flex items-center justify-between gap-2"><span className="font-medium">{methodLabel(m.method)}</span><Badge tone={m.analysis.startsWith("secondary") ? "pending" : "ok"}>{m.analysis}</Badge></div>
                <p className="mt-1 text-xs text-ink-soft">{m.description}</p>
                <p className="mt-1 text-xs text-amber-800"><b>Limitation:</b> {m.limitations}</p>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}

function TrainForm({ methods, onStarted }: { methods: string[]; onStarted: () => void }) {
  const [method, setMethod] = useState("robust_z_personal");
  const [mods, setMods] = useState<string[]>(MODALITIES);
  const [q, setQ] = useState(0.95);
  const [folds, setFolds] = useState(5);
  const [seed, setSeed] = useState(42);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true); setErr(null); setMsg(null);
    try {
      const r = await api.train({ method, modalities: mods.length === MODALITIES.length ? undefined : mods, threshold_quantile: q, n_folds: folds, seed });
      setMsg(`Training run #${r.id} queued.`);
      onStarted();
    } catch (e2) {
      setErr(e2 instanceof ApiError ? e2.message : String(e2));
    } finally {
      setBusy(false);
    }
  };
  return (
    <Card title="Train a model">
      <form className="space-y-4" onSubmit={submit}>
        <Field label="Method" htmlFor="method">
          <select id="method" className="input" value={method} onChange={(e) => setMethod(e.target.value)}>
            {methods.map((m) => <option key={m} value={m}>{methodLabel(m)}</option>)}
          </select>
        </Field>
        <fieldset>
          <legend className="label">Modalities</legend>
          <div className="flex flex-wrap gap-3">
            {MODALITIES.map((m) => (
              <label key={m} className="flex items-center gap-1.5 text-sm">
                <input type="checkbox" className="accent-brand-700" checked={mods.includes(m)} onChange={(e) => setMods(e.target.checked ? [...mods, m] : mods.filter((x) => x !== m))} />{m}
              </label>
            ))}
          </div>
        </fieldset>
        <div className="grid grid-cols-3 gap-3">
          <Field label="Threshold q" htmlFor="q"><input id="q" className="input" type="number" min={0.51} max={0.999} step={0.005} value={q} onChange={(e) => setQ(Number(e.target.value))} /></Field>
          <Field label="CV folds" htmlFor="folds"><input id="folds" className="input" type="number" min={2} max={10} value={folds} onChange={(e) => setFolds(Number(e.target.value))} /></Field>
          <Field label="Seed" htmlFor="seed"><input id="seed" className="input" type="number" min={0} value={seed} onChange={(e) => setSeed(Number(e.target.value))} /></Field>
        </div>
        <p className="text-xs text-ink-soft">The threshold is the q-quantile of scores on training participants' rest windows (target false-positive rate 1 − q).</p>
        <button className="btn-primary w-full" disabled={busy || mods.length === 0}>{busy ? "Submitting…" : "Start training"}</button>
        {msg && <Notice>{msg}</Notice>}
        {err && <div className="rounded-lg bg-rose-50 p-3 text-sm text-rose-800" role="alert">{err}</div>}
      </form>
    </Card>
  );
}

function ModelDetail({ id }: { id: number }) {
  const m = useApi(() => api.modelMetrics(id), [id]);
  if (m.loading) return <Loading />;
  if (m.error) return <ErrorState error={m.error} />;
  const ev = m.data!.evaluations[0];
  const met = ev?.metrics as Metrics | null;
  const pp = Object.entries(ev?.per_participant ?? {}).map(([p, v]) => ({ p, v })).sort((a, b) => a.v - b.v);
  const details = (ev?.details ?? {}) as { timing?: { fit_seconds_mean: number; infer_ms_per_window: number }; ineligible?: Record<string, string> };
  return (
    <Card title={`${methodLabel(m.data!.method)} v${m.data!.version} - evaluation`} subtitle={ev?.kind}>
      <Notice>{m.data!.description} <b>Limitation:</b> {m.data!.limitations}</Notice>
      {!met ? <p className="mt-4 text-sm text-ink-soft">No labelled evaluation is available for this model ({String((ev?.details as Record<string, unknown> | null)?.reason ?? "unsupervised")}).</p> : (
        <div className="mt-4 grid gap-6 lg:grid-cols-3">
          <KeyValue items={[
            ["ROC-AUC (pooled)", fmt(met.roc_auc)], ["PR-AUC", `${fmt(met.pr_auc)} (chance ${fmt(met.prevalence)})`],
            ["Median participant AUC", fmt(met.participant_auc_median)], ["Precision / Recall", `${fmt(met.precision)} / ${fmt(met.recall)}`],
            ["F1", fmt(met.f1)], ["False alarms / hour of rest", fmt(met.false_alarms_per_hour, 1)], ["Event sensitivity", fmt(met.event_sensitivity)],
            ["Confusion (TP/FP/TN/FN)", `${met.tp}/${met.fp}/${met.tn}/${met.fn}`], ["Participants / windows", `${met.n_participants} / ${met.n_windows}`],
            ["Fit time (mean per fold)", details.timing ? `${fmt(details.timing.fit_seconds_mean, 3)} s` : "n/a"],
            ["Inference", details.timing ? `${fmt(details.timing.infer_ms_per_window, 4)} ms/window` : "n/a"],
          ]} />
          <div>
            <div className="label">ROC curve (out-of-fold)</div>
            {ev.curves?.roc ? <CurveChart x={ev.curves.roc.fpr} y={ev.curves.roc.tpr} xLabel="false-positive rate" yLabel="TPR" diagonal /> : <p className="text-sm text-ink-soft">n/a</p>}
          </div>
          <div>
            <div className="label">Precision-recall curve</div>
            {ev.curves?.pr ? <CurveChart x={ev.curves.pr.recall} y={ev.curves.pr.precision} xLabel="recall" yLabel="precision" /> : <p className="text-sm text-ink-soft">n/a</p>}
          </div>
        </div>
      )}
      {pp.length > 0 && (
        <div className="mt-6">
          <div className="label">Per-participant ROC-AUC</div>
          <div style={{ height: 220 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={pp} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="p" fontSize={10} interval={0} angle={-60} textAnchor="end" height={45} />
                <YAxis domain={[0, 1]} fontSize={11} width={36} />
                <ReferenceLine y={0.5} stroke="#94a3b8" strokeDasharray="4 4" />
                <Tooltip formatter={(v: number) => fmt(v)} />
                <Bar dataKey="v" fill="#2a8c80" name="AUC" isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </Card>
  );
}

function ExperimentPanel() {
  const list = useApi(() => api.experiments(), []);
  const latest = list.data?.runs.find((r) => r.status === "completed");
  const exp = useApi(() => (latest ? api.experiment(latest.run_id) : Promise.resolve(null)), [latest?.run_id]);
  const running = (list.data?.runs ?? []).some((r) => r.status === "running");
  usePolling(() => list.reload(), 5000, running);
  const [err, setErr] = useState<string | null>(null);
  const launch = async (quick: boolean) => {
    setErr(null);
    try { await api.runExperiments(quick); list.reload(); } catch (e) { setErr(e instanceof ApiError ? e.message : String(e)); }
  };
  const rows = exp.data ? [...Object.entries(exp.data.primary.metrics).map(([k, v]) => ({ k, v, a: "primary" })), ...Object.entries(exp.data.secondary.metrics).map(([k, v]) => ({ k, v, a: "secondary (post hoc)" }))] : [];
  return (
    <Card title="Method comparison (research experiment suite)" subtitle="Identical evaluation windows for all methods; 95% CIs from a participant-level bootstrap."
      actions={<div className="flex gap-2"><button className="btn-secondary" disabled={running} onClick={() => launch(true)}>Quick run</button><button className="btn-primary" disabled={running} onClick={() => launch(false)}>Run full suite</button></div>}>
      {err && <div className="mb-3 rounded-lg bg-rose-50 p-3 text-sm text-rose-800">{err}</div>}
      {running && <div className="mb-3"><Notice>An experiment run is in progress (it runs as a separate process and takes several minutes). <pre className="mt-2 whitespace-pre-wrap text-xs">{list.data?.runs.find((r) => r.status === "running")?.log_tail}</pre></Notice></div>}
      {list.loading || exp.loading ? <Loading /> : !exp.data ? <p className="text-sm text-ink-soft">No completed experiment run yet.</p> : (
        <>
          <p className="mb-3 text-xs text-ink-soft">Run <code>{exp.data.run_id}</code> · {exp.data.eligibility.eligible.length} participants · {exp.data.label_counts.stressor} stressor / {exp.data.label_counts.rest} rest windows · generated {dateTime(exp.data.generated_at)}</p>
          <div className="overflow-x-auto">
            <table className="table-base">
              <thead><tr><th>Method</th><th>Analysis</th><th>ROC-AUC [95% CI]</th><th>PR-AUC</th><th>Median participant AUC</th><th>F1</th><th>False alarms/h</th><th>Event sens.</th></tr></thead>
              <tbody>
                {rows.map(({ k, v, a }) => (
                  <tr key={k}>
                    <td className="font-medium">{methodLabel(k)}</td><td className="text-xs">{a}</td>
                    <td>{fmt(v.roc_auc)} <span className="text-xs text-ink-faint">[{fmt(v.ci95?.roc_auc?.[0])}, {fmt(v.ci95?.roc_auc?.[1])}]</span></td>
                    <td>{fmt(v.pr_auc)}</td><td>{fmt(v.participant_auc_median)}</td><td>{fmt(v.f1)}</td><td>{fmt(v.false_alarms_per_hour, 1)}</td><td>{fmt(v.event_sensitivity)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <h3 className="mb-2 mt-6 text-sm font-semibold">Paired comparisons (global vs personalised)</h3>
          <div className="overflow-x-auto">
            <table className="table-base">
              <thead><tr><th>A</th><th>B</th><th>Pooled AUC diff B−A [95% CI]</th><th>Median participant diff</th><th>B better</th><th>Wilcoxon p</th></tr></thead>
              <tbody>
                {exp.data.paired.map((c) => (
                  <tr key={`${c.a}-${c.b}`}>
                    <td>{methodLabel(c.a)}</td><td>{methodLabel(c.b)}</td>
                    <td>{fmt(c.pooled_auc_diff)} <span className="text-xs text-ink-faint">[{fmt(c.pooled_auc_diff_ci95[0])}, {fmt(c.pooled_auc_diff_ci95[1])}]</span></td>
                    <td>{fmt(c.participant_auc_diff_median)}</td><td>{c.n_participants_b_better}/{c.n_participants}</td><td>{fmt(c.wilcoxon?.p_value, 4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-xs text-ink-faint">No multiple-comparison correction is applied. Full results, ablations and figures are in Research Reports.</p>
        </>
      )}
    </Card>
  );
}
