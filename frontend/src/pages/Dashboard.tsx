import { Link } from "react-router-dom";
import { api } from "../lib/api";
import { useApi } from "../lib/hooks";
import { dateTime, fmt, int, methodLabel } from "../lib/format";
import { Badge, Card, ErrorState, Loading, NoDataset, Notice, PageHeader, Stat } from "../components/ui";

export default function Dashboard() {
  const meta = useApi(() => api.metadata(), []);
  const quality = useApi(() => (meta.data?.configured ? api.quality() : Promise.resolve(null)), [meta.data?.configured]);
  const models = useApi(() => api.models(), []);
  const runs = useApi(() => api.trainingRuns(), []);
  const anomalies = useApi(() => api.anomalies({ page_size: 8 }), []);
  const experiments = useApi(() => api.experiments(), []);

  if (meta.loading) return <Loading />;
  if (meta.error) return <ErrorState error={meta.error} onRetry={meta.reload} />;
  const m = meta.data!;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Research dashboard"
        subtitle="Status of the dataset, data quality, trained models and recent offline detections. All figures are read from the backend; nothing on this page is simulated."
      />
      {!m.configured ? (
        <>
          {m.latest_attempt && (
            <Notice tone={m.latest_attempt.status === "failed" ? "warn" : "info"}>
              Latest ingestion attempt: <b>{m.latest_attempt.status}</b> {m.latest_attempt.progress ? `(${m.latest_attempt.progress})` : ""}
              {m.latest_attempt.error ? ` - ${m.latest_attempt.error}` : ""}
            </Notice>
          )}
          <NoDataset />
        </>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            <Stat label="Participants" value={int(m.dataset!.n_participants)} hint={`${int(m.dataset!.n_sessions)} recording sessions`} />
            <Stat label="Windows processed" value={int(m.dataset!.n_windows)} hint="30 s windows, 10 s hop" />
            <Stat label="Sensor modalities" value={m.modalities?.length ?? 0} hint={m.modalities?.join(", ")} />
            <Stat
              label="Trained models"
              value={models.data ? models.data.length : "…"}
              hint={models.data?.length ? `latest: ${methodLabel(models.data[0].method)}` : "none yet - see Model Laboratory"}
              tone={models.data?.length ? "good" : "warn"}
            />
          </div>

          <div className="grid gap-6 lg:grid-cols-3">
            <Card title="Dataset" className="lg:col-span-2" actions={<Badge>{m.dataset!.status}</Badge>}>
              <p className="text-sm font-medium">{m.dataset!.name}</p>
              <p className="mt-1 text-xs text-ink-soft">
                Version {m.dataset!.version} · DOI {m.dataset!.doi ?? "n/a"} · {m.dataset!.license} · processed {dateTime(m.processed_at)}
              </p>
              <div className="mt-4 grid gap-3 sm:grid-cols-3">
                {Object.entries(m.sampling_rates ?? {}).map(([k, v]) => (
                  <div key={k} className="rounded-lg bg-slate-50 px-3 py-2 text-sm">
                    <div className="font-semibold">{k}</div>
                    <div className="text-xs text-ink-soft">{v.join(", ")} Hz</div>
                  </div>
                ))}
              </div>
              <p className="mt-4 text-xs text-ink-soft">Protocols: {m.protocols?.join(", ")}</p>
            </Card>

            <Card title="Data quality" subtitle="Per-session status from sensor-specific checks">
              {quality.loading ? <Loading /> : quality.error ? <ErrorState error={quality.error} /> : quality.data && (
                <div className="space-y-3 text-sm">
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(quality.data.status_counts).map(([k, v]) => <Badge key={k} tone={k}>{`${k}: ${v}`}</Badge>)}
                  </div>
                  <div>
                    <div className="label">Mean valid fraction per modality</div>
                    {Object.entries(quality.data.per_modality).map(([mod, q]) => (
                      <div key={mod} className="flex items-center gap-2 py-0.5">
                        <span className="w-12 text-xs font-medium">{mod}</span>
                        <div className="h-2 flex-1 rounded bg-slate-100">
                          <div className="h-2 rounded bg-brand-500" style={{ width: `${q.mean_valid_fraction * 100}%` }} />
                        </div>
                        <span className="w-12 text-right text-xs">{(q.mean_valid_fraction * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                  <p className="text-xs text-ink-soft">{quality.data.excluded_sessions.length} sessions excluded from modelling (documented dataset constraints).</p>
                  <Link className="text-xs font-medium text-brand-700 hover:underline" to="/data">Open Data Explorer →</Link>
                </div>
              )}
            </Card>
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            <Card title="Model readiness" actions={<Link className="text-xs font-medium text-brand-700 hover:underline" to="/models">Model Laboratory →</Link>}>
              {models.loading ? <Loading /> : models.error ? <ErrorState error={models.error} /> : models.data!.length === 0 ? (
                <p className="text-sm text-ink-soft">No model has been trained. Train one in the Model Laboratory; until then detection endpoints return a "model not ready" error rather than any prediction.</p>
              ) : (
                <div className="overflow-x-auto">
                  <table className="table-base">
                    <thead><tr><th>Model</th><th>ROC-AUC (CV)</th><th>Median participant AUC</th><th>Artifact</th></tr></thead>
                    <tbody>
                      {models.data!.slice(0, 6).map((mv) => (
                        <tr key={mv.id}>
                          <td>{methodLabel(mv.method)} <span className="text-xs text-ink-faint">v{mv.version}</span></td>
                          <td>{fmt(mv.metrics?.roc_auc)}</td>
                          <td>{fmt(mv.metrics?.participant_auc_median)}</td>
                          <td>{mv.artifact_ok ? <Badge tone="ok">valid</Badge> : <Badge tone="failed">missing</Badge>}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              {runs.data && runs.data.items.length > 0 && (
                <p className="mt-3 text-xs text-ink-soft">
                  Latest training run: #{runs.data.items[0].id} {methodLabel(runs.data.items[0].method)} - <Badge>{runs.data.items[0].status}</Badge> ({dateTime(runs.data.items[0].created_at)})
                </p>
              )}
            </Card>

            <Card title="Recent detected anomalies" subtitle="Windows above threshold from offline analyses" actions={<Link className="text-xs font-medium text-brand-700 hover:underline" to="/monitoring">Monitoring →</Link>}>
              {anomalies.loading ? <Loading /> : anomalies.error ? <ErrorState error={anomalies.error} /> : anomalies.data!.items.length === 0 ? (
                <p className="text-sm text-ink-soft">No detections yet. Run an analysis on a session in Anomaly Monitoring.</p>
              ) : (
                <ul className="divide-y divide-slate-100 text-sm">
                  {anomalies.data!.items.map((a) => (
                    <li key={a.id} className="flex items-center justify-between gap-3 py-2">
                      <div>
                        <div className="font-medium">{a.session_key} · window {a.window_index}</div>
                        <div className="text-xs text-ink-soft">{a.phase ?? "unlabelled phase"} · score {fmt(a.score)} (threshold {fmt(a.threshold)})</div>
                      </div>
                      <Link className="btn-secondary px-2 py-1 text-xs" to={`/explain?anomaly=${a.id}`}>Explain</Link>
                    </li>
                  ))}
                </ul>
              )}
            </Card>
          </div>

          <Card title="Research experiments" actions={<Link className="text-xs font-medium text-brand-700 hover:underline" to="/reports">Reports →</Link>}>
            {experiments.loading ? <Loading /> : experiments.error ? <ErrorState error={experiments.error} /> : experiments.data!.runs.length === 0 ? (
              <p className="text-sm text-ink-soft">No experiment run yet. Results will appear here after running the evaluation pipeline.</p>
            ) : (
              <ul className="text-sm">
                {experiments.data!.runs.slice(0, 3).map((r) => (
                  <li key={r.run_id} className="flex flex-wrap items-center gap-3 py-1">
                    <code className="text-xs">{r.run_id}</code> <Badge>{r.status}</Badge>
                    {r.n_participants && <span className="text-xs text-ink-soft">{r.n_participants} participants · {int(r.n_eval_windows)} evaluation windows · {dateTime(r.generated_at)}</span>}
                  </li>
                ))}
              </ul>
            )}
          </Card>
        </>
      )}
    </div>
  );
}
