import { useEffect, useMemo, useState } from "react";
import { ApiError, api } from "../lib/api";
import { useApi, usePolling } from "../lib/hooks";
import { dateTime, int, minutes, pct } from "../lib/format";
import type { SessionInfo } from "../lib/types";
import { PhaseLegend, SignalChart } from "../components/charts";
import { Badge, Card, EmptyState, ErrorState, Field, KeyValue, Loading, Notice, PageHeader } from "../components/ui";

export default function DataExplorer() {
  const meta = useApi(() => api.metadata(), []);
  const datasets = useApi(() => api.datasets(), []);
  const busy = (datasets.data ?? []).some((d) => d.status === "pending" || d.status === "processing");
  usePolling(() => {
    datasets.reload();
    meta.reload();
  }, 3000, busy);

  return (
    <div className="space-y-6">
      <PageHeader
        title="Data Explorer"
        subtitle="Import a dataset, inspect recordings and examine data quality. Signals are shown as 1-second summaries of the cleaned signals; protocol phases come from the dataset's own event tags."
      />
      <ImportPanel onStarted={() => datasets.reload()} />
      {datasets.data && datasets.data.length > 0 && (
        <Card title="Ingestion history">
          <div className="overflow-x-auto">
            <table className="table-base">
              <thead><tr><th>#</th><th>Dataset</th><th>Status</th><th>Progress</th><th>Sessions</th><th>Windows</th><th>Created</th></tr></thead>
              <tbody>
                {datasets.data.map((d) => (
                  <tr key={d.id}>
                    <td>{d.id}</td>
                    <td className="max-w-xs truncate" title={d.name}>{d.name}</td>
                    <td><Badge>{d.status}</Badge>{d.error && <div className="mt-1 max-w-xs text-xs text-rose-700">{d.error}</div>}</td>
                    <td className="text-xs">{d.progress ?? "-"}</td>
                    <td>{int(d.n_sessions)}</td>
                    <td>{int(d.n_windows)}</td>
                    <td className="text-xs">{dateTime(d.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
      {meta.loading ? <Loading /> : meta.error ? <ErrorState error={meta.error} /> : meta.data?.configured ? <Explorer /> : (
        <EmptyState title="No processed dataset yet">Use the import panel above. Charts appear once ingestion finishes.</EmptyState>
      )}
    </div>
  );
}

function ImportPanel({ onStarted }: { onStarted: () => void }) {
  const sources = useApi(() => api.sources(), []);
  const [file, setFile] = useState<File | null>(null);
  const [validation, setValidation] = useState<Record<string, unknown> | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [working, setWorking] = useState(false);

  const run = async (fn: () => Promise<unknown>, ok: string) => {
    setWorking(true);
    setError(null);
    setMessage(null);
    try {
      await fn();
      setMessage(ok);
      onStarted();
    } catch (e) {
      setError(e instanceof ApiError ? `${e.message}${e.details ? ` (${JSON.stringify(e.details)})` : ""}` : String(e));
    } finally {
      setWorking(false);
    }
  };

  return (
    <Card title="Import data" subtitle="Server-side PhysioNet folder, a ZIP of the PhysioNet release, or a long-format CSV">
      <div className="grid gap-6 lg:grid-cols-2">
        <div>
          <h3 className="mb-2 text-sm font-semibold">Server-side datasets (under DATASETS_DIR)</h3>
          {sources.loading ? <Loading /> : sources.error ? <ErrorState error={sources.error} /> : sources.data!.sources.length === 0 ? (
            <p className="text-sm text-ink-soft">No dataset folder found. Run <code>python scripts/download_dataset.py</code> on the server.</p>
          ) : (
            <ul className="space-y-2">
              {sources.data!.sources.map((s) => (
                <li key={s.path} className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-slate-200 p-3">
                  <div>
                    <div className="text-sm font-medium">{s.path}</div>
                    <div className="text-xs text-ink-soft">{s.type} · protocols {s.protocols.join(", ")}</div>
                  </div>
                  <button className="btn-primary" disabled={working} onClick={() => run(() => api.ingest(s.path), "Ingestion started. Progress is shown in the history table.")}>
                    Ingest
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
        <div>
          <h3 className="mb-2 text-sm font-semibold">Upload a file (.csv or .zip)</h3>
          <Field label="File" htmlFor="upload" hint="CSV columns: timestamp, participant_id, and any of eda, hr, temp, bvp, acc_x/acc_y/acc_z (optional: label, phase, calibration).">
            <input id="upload" type="file" accept=".csv,.zip" className="input" onChange={(e) => { setFile(e.target.files?.[0] ?? null); setValidation(null); }} />
          </Field>
          <div className="mt-3 flex flex-wrap gap-2">
            <button className="btn-secondary" disabled={!file || working} onClick={() => run(async () => setValidation(await api.validateFile(file!)), "Validation finished.")}>Validate</button>
            <button className="btn-primary" disabled={!file || working || (validation !== null && validation.valid === false)} onClick={() => run(() => api.uploadFile(file!, file!.name), "Upload accepted; ingestion started.")}>Upload &amp; ingest</button>
          </div>
          {validation && (
            <div className="mt-3 rounded-lg bg-slate-50 p-3 text-xs">
              <div className="mb-1 font-semibold">Validation: {validation.valid ? <Badge tone="ok">valid</Badge> : <Badge tone="failed">invalid</Badge>}</div>
              {(validation.errors as string[] | undefined)?.map((e) => <div key={e} className="text-rose-700">• {e}</div>)}
              {(validation.warnings as string[] | undefined)?.slice(0, 8).map((w) => <div key={w} className="text-amber-800">• {w}</div>)}
              {"participants" in validation && <div className="mt-1">Participants: {String(validation.participants)}</div>}
            </div>
          )}
        </div>
      </div>
      {message && <div className="mt-4"><Notice>{message}</Notice></div>}
      {error && <div className="mt-4 rounded-lg bg-rose-50 p-3 text-sm text-rose-800" role="alert">{error}</div>}
    </Card>
  );
}

function Explorer() {
  const sessions = useApi(() => api.sessions(), []);
  const quality = useApi(() => api.quality(), []);
  const [participant, setParticipant] = useState("");
  const [sessionKey, setSessionKey] = useState("");
  const [selected, setSelected] = useState<string[]>(["EDA", "HR", "TEMP", "ACC"]);
  const [range, setRange] = useState<{ start: string; end: string }>({ start: "", end: "" });

  const participants = useMemo(() => [...new Set((sessions.data ?? []).map((s) => s.participant!))].sort(), [sessions.data]);
  useEffect(() => {
    if (!participant && participants.length) setParticipant(participants[0]);
  }, [participants, participant]);
  const pSessions = (sessions.data ?? []).filter((s) => s.participant === participant);
  useEffect(() => {
    if (pSessions.length && !pSessions.some((s) => s.session_key === sessionKey)) setSessionKey(pSessions[0].session_key);
  }, [participant, pSessions, sessionKey]);
  const session = pSessions.find((s) => s.session_key === sessionKey);

  if (sessions.loading) return <Loading />;
  if (sessions.error) return <ErrorState error={sessions.error} />;

  return (
    <>
      <Card title="Recording browser">
        <div className="grid gap-4 md:grid-cols-4">
          <Field label="Participant" htmlFor="participant">
            <select id="participant" className="input" value={participant} onChange={(e) => setParticipant(e.target.value)}>
              {participants.map((p) => <option key={p}>{p}</option>)}
            </select>
          </Field>
          <Field label="Session" htmlFor="session">
            <select id="session" className="input" value={sessionKey} onChange={(e) => setSessionKey(e.target.value)}>
              {pSessions.map((s) => <option key={s.session_key} value={s.session_key}>{s.session_key} ({s.protocol})</option>)}
            </select>
          </Field>
          <Field label="From (min)" htmlFor="from">
            <input id="from" className="input" type="number" min={0} step={1} value={range.start} placeholder="0" onChange={(e) => setRange({ ...range, start: e.target.value })} />
          </Field>
          <Field label="To (min)" htmlFor="to">
            <input id="to" className="input" type="number" min={0} step={1} value={range.end} placeholder="end" onChange={(e) => setRange({ ...range, end: e.target.value })} />
          </Field>
        </div>
        {session && (
          <fieldset className="mt-4">
            <legend className="label">Signals</legend>
            <div className="flex flex-wrap gap-3">
              {(session.modalities ?? []).map((m) => (
                <label key={m} className="flex items-center gap-1.5 text-sm">
                  <input type="checkbox" className="accent-brand-700" checked={selected.includes(m)} onChange={(e) => setSelected(e.target.checked ? [...selected, m] : selected.filter((x) => x !== m))} />
                  {m}
                </label>
              ))}
            </div>
          </fieldset>
        )}
      </Card>
      {session && <SessionView session={session} signals={selected.filter((s) => session.modalities?.includes(s))} range={range} />}
      {quality.data && (
        <Card title="Missing-value analysis (all windows)" subtitle="Fraction of windows in which a feature could not be computed because its modality had < 80% valid samples or was documented as invalid">
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {Object.entries(quality.data.feature_missing_fraction).map(([f, v]) => (
              <div key={f} className="flex items-center gap-2 text-sm">
                <span className="w-32 truncate font-mono text-xs">{f}</span>
                <div className="h-2 flex-1 rounded bg-slate-100"><div className="h-2 rounded bg-amber-500" style={{ width: `${Math.min(v * 100, 100)}%` }} /></div>
                <span className="w-14 text-right text-xs">{pct(v)}</span>
              </div>
            ))}
          </div>
          {quality.data.excluded_sessions.length > 0 && (
            <div className="mt-4 text-sm">
              <div className="label">Sessions excluded from modelling</div>
              <ul className="list-disc pl-5 text-xs text-ink-soft">
                {quality.data.excluded_sessions.map((e) => <li key={e.session_key}><b>{e.session_key}</b>: {e.reason}</li>)}
              </ul>
            </div>
          )}
        </Card>
      )}
    </>
  );
}

function SessionView({ session, signals, range }: { session: SessionInfo; signals: string[]; range: { start: string; end: string } }) {
  const start = range.start ? Number(range.start) * 60 : undefined;
  const end = range.end ? Number(range.end) * 60 : undefined;
  const data = useApi(
    () => (signals.length ? api.signals(session.session_key, { signals: signals.join(","), start_s: start, end_s: end }) : Promise.resolve(null)),
    [session.session_key, signals.join(","), start, end],
  );
  return (
    <div className="grid gap-6 xl:grid-cols-3">
      <Card title={`Signals · ${session.session_key}`} className="xl:col-span-2" actions={<PhaseLegend />}>
        {signals.length === 0 ? <p className="text-sm text-ink-soft">Select at least one signal.</p> : data.loading ? <Loading /> : data.error ? <ErrorState error={data.error} /> : data.data && (
          <div className="space-y-3">
            {data.data.points.length === 0 ? <p className="text-sm text-ink-soft">No samples in the selected time range.</p> : signals.map((s) => (
              <SignalChart key={s} points={data.data!.points} signal={s} unit={data.data!.units[s] ?? ""} phases={data.data!.phases} calibrationEnd={data.data!.calibration_end_s} />
            ))}
            <p className="text-xs text-ink-faint">Display resolution: {data.data.resolution_s} s per point. Gaps indicate missing or invalid samples.</p>
          </div>
        )}
      </Card>
      <div className="min-w-0 space-y-6">
        <Card title="Session details" actions={<Badge>{session.excluded ? "excluded" : session.quality_status}</Badge>}>
          <KeyValue items={[
            ["Participant", session.participant],
            ["Protocol", session.protocol],
            ["Duration", minutes(session.duration_s)],
            ["Start (shifted UTC)", dateTime(session.origin)],
            ["Windows", `${session.n_windows} total · ${session.n_calibration_windows} baseline · ${session.n_eval_windows} labelled evaluation`],
            ["Baseline ends at", session.calibration_end_s != null ? `${(session.calibration_end_s / 60).toFixed(1)} min` : "n/a (no labelled protocol)"],
          ]} />
          {session.exclusion_reason && <p className="mt-3 text-xs text-rose-700">{session.exclusion_reason}</p>}
          {session.notes && session.notes.length > 0 && <ul className="mt-3 list-disc pl-5 text-xs text-ink-soft">{session.notes.map((n) => <li key={n}>{n}</li>)}</ul>}
        </Card>
        <Card title="Sampling rate and quality">
          <div className="overflow-x-auto">
            <table className="table-base">
              <thead><tr><th>Signal</th><th>Hz</th><th>Missing</th><th>Out of range</th><th>Flat-line</th><th>Status</th></tr></thead>
              <tbody>
                {Object.entries(session.quality ?? {}).map(([k, q]) => (
                  <tr key={k} title={q.issues.join("; ")}>
                    <td className="font-medium">{k}</td><td>{session.sampling_rates?.[k] ?? "-"}</td>
                    <td>{pct(q.missing_fraction)}</td><td>{pct(q.out_of_range_fraction)}</td><td>{pct(q.flatline_fraction)}</td>
                    <td><Badge>{q.status}</Badge></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
        {session.phases && session.phases.length > 0 && (
          <Card title="Protocol phases" subtitle="Derived from the dataset's button tags">
            <ul className="max-h-64 space-y-1 overflow-y-auto text-xs">
              {session.phases.map((p, i) => (
                <li key={i} className="flex justify-between gap-2">
                  <span>{p.name} {p.label === 1 ? <Badge tone="processing">stressor</Badge> : p.label === 0 ? <Badge tone="ok">rest</Badge> : null}</span>
                  <span className="text-ink-soft">{(p.start_s / 60).toFixed(1)}–{(p.end_s / 60).toFixed(1)} min</span>
                </li>
              ))}
            </ul>
          </Card>
        )}
      </div>
    </div>
  );
}
