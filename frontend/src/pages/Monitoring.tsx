import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ApiError, api } from "../lib/api";
import { useApi } from "../lib/hooks";
import { dateTime, fmt, methodLabel, mmss } from "../lib/format";
import type { DetectionResult } from "../lib/types";
import { PhaseLegend, ScoreTimeline, SignalChart } from "../components/charts";
import { Badge, Card, EmptyState, ErrorState, Field, Loading, Notice, PageHeader, Stat } from "../components/ui";

export default function Monitoring() {
  const models = useApi(() => api.models(), []);
  const sessions = useApi(() => api.sessions(), []);
  const history = useApi(() => api.detections(), []);
  const [modelId, setModelId] = useState<number | "">("");
  const [participant, setParticipant] = useState("");
  const [sessionKey, setSessionKey] = useState("");
  const [threshold, setThreshold] = useState("");
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [running, setRunning] = useState(false);
  const navigate = useNavigate();

  const participants = useMemo(() => [...new Set((sessions.data ?? []).map((s) => s.participant!))].sort(), [sessions.data]);
  const pSessions = (sessions.data ?? []).filter((s) => s.participant === participant);
  const model = models.data?.find((m) => m.id === modelId);
  useEffect(() => { if (modelId === "" && models.data?.length) setModelId(models.data[0].id); }, [models.data, modelId]);
  useEffect(() => { if (!participant && participants.length) setParticipant(participants[0]); }, [participants, participant]);
  useEffect(() => {
    if (pSessions.length && !pSessions.some((s) => s.session_key === sessionKey)) {
      setSessionKey((pSessions.find((s) => s.protocol === "STRESS") ?? pSessions[0]).session_key);
    }
  }, [participant, pSessions, sessionKey]);

  const analyse = async () => {
    if (modelId === "" || !sessionKey) return;
    setRunning(true);
    setError(null);
    try {
      setResult(await api.detect(modelId, sessionKey, threshold === "" ? undefined : Number(threshold)));
      history.reload();
    } catch (e) {
      setError(e as ApiError);
    } finally {
      setRunning(false);
    }
  };

  const loadPast = async (id: number) => {
    setError(null);
    try {
      setResult(await api.detection(id));
    } catch (e) {
      setError(e as ApiError);
    }
  };

  if (models.loading || sessions.loading) return <Loading />;
  if (sessions.error) return <ErrorState error={sessions.error} />;

  return (
    <div className="space-y-6">
      <PageHeader title="Anomaly monitoring" subtitle="Retrospective scoring of recorded sessions with a trained model. Every window gets a model-generated score; windows above the threshold are stored as detections." />
      <Notice tone="warn">
        <b>Offline analysis, not real-time monitoring.</b> MOSAIC analyses recordings after they were made. Scores measure deviation from a reference and are not probabilities of any medical condition.
      </Notice>
      {models.data?.length === 0 ? (
        <EmptyState title="No trained model" action={<Link className="btn-primary" to="/models">Train a model</Link>}>
          Detection requires a trained model. MOSAIC never returns predictions from an untrained model.
        </EmptyState>
      ) : (
        <Card title="Analysis setup">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
            <Field label="Model" htmlFor="model">
              <select id="model" className="input" value={modelId} onChange={(e) => { setModelId(Number(e.target.value)); setThreshold(""); }}>
                {models.data!.map((m) => <option key={m.id} value={m.id}>{methodLabel(m.method)} v{m.version} (#{m.id})</option>)}
              </select>
            </Field>
            <Field label="Participant" htmlFor="mp">
              <select id="mp" className="input" value={participant} onChange={(e) => setParticipant(e.target.value)}>
                {participants.map((p) => <option key={p}>{p}</option>)}
              </select>
            </Field>
            <Field label="Session" htmlFor="ms">
              <select id="ms" className="input" value={sessionKey} onChange={(e) => setSessionKey(e.target.value)}>
                {pSessions.map((s) => <option key={s.session_key} value={s.session_key}>{s.session_key}</option>)}
              </select>
            </Field>
            <Field label="Threshold" htmlFor="thr" hint={`Calibrated: ${fmt(model?.threshold)} (q = ${model?.threshold_quantile ?? "fixed"})`}>
              <input id="thr" className="input" type="number" step="any" placeholder={fmt(model?.threshold)} value={threshold} onChange={(e) => setThreshold(e.target.value)} />
            </Field>
            <div className="flex items-end">
              <button className="btn-primary w-full" disabled={running || modelId === "" || !sessionKey} onClick={analyse}>
                {running ? "Analysing…" : "Run offline analysis"}
              </button>
            </div>
          </div>
        </Card>
      )}
      {error && <ErrorState error={error} />}
      {result && <ResultView result={result} onExplain={(w) => navigate(`/explain?model=${result.model.id}&recording=${result.session.session_key}&window=${w}`)} />}

      <Card title="Detection history">
        {history.loading ? <Loading /> : history.error ? <ErrorState error={history.error} /> : history.data!.items.length === 0 ? (
          <p className="text-sm text-ink-soft">No analyses have been run yet.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="table-base">
              <thead><tr><th>#</th><th>Session</th><th>Model</th><th>Threshold</th><th>Flagged / scored</th><th>When</th><th /></tr></thead>
              <tbody>
                {history.data!.items.map((h) => (
                  <tr key={h.id}>
                    <td>{h.id}</td><td>{h.session_key}</td><td>{methodLabel(h.method)} v{h.version}</td><td>{fmt(h.threshold)}</td>
                    <td>{h.n_flagged} / {h.n_scored}</td><td className="text-xs">{dateTime(h.created_at)}</td>
                    <td><button className="btn-secondary px-2 py-1 text-xs" onClick={() => loadPast(h.id)}>Open</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}

function ResultView({ result, onExplain }: { result: DetectionResult; onExplain: (w: number) => void }) {
  const signals = useApi(() => api.signals(result.session.session_key, { signals: "EDA,HR" }).catch(() => null), [result.session.session_key]);
  const lowQuality = result.timeline.filter((p) => Object.entries(p).some(([k, v]) => k.startsWith("q_") && typeof v === "number" && v < 0.8));
  const flagged = result.timeline.filter((p) => p.flagged);
  const labelled = result.timeline.filter((p) => p.label !== null && !p.is_calibration);
  const flagRate = (lab: number) => {
    const sel = labelled.filter((p) => p.label === lab && p.score !== null);
    return sel.length ? sel.filter((p) => p.flagged).length / sel.length : null;
  };
  return (
    <>
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
        <Stat label="Windows scored" value={`${result.n_scored}/${result.n_windows}`} hint="unscored = no baseline or no valid features" />
        <Stat label="Flagged windows" value={result.n_flagged} tone={result.n_flagged ? "warn" : "default"} />
        <Stat label="Threshold" value={fmt(result.threshold)} hint={`median score ${fmt(result.score_summary.median)}`} />
        <Stat label="Flag rate in stressor phases" value={flagRate(1) === null ? "n/a" : `${(flagRate(1)! * 100).toFixed(0)}%`} hint="protocol-labelled windows only" />
        <Stat label="Flag rate in rest phases" value={flagRate(0) === null ? "n/a" : `${(flagRate(0)! * 100).toFixed(0)}%`} hint="false-alarm proxy" />
      </div>
      <Card
        title={`Anomaly-score timeline · ${result.model.method} v${result.model.version} · ${result.session.session_key}`}
        subtitle={result.mode_note}
        actions={<PhaseLegend />}
      >
        {result.reference.status !== "ok" && <div className="mb-3"><Notice tone="warn">Personal baseline unavailable: {String(result.reference.reason)}. Windows of this participant are not scored by personalised models.</Notice></div>}
        {result.reference.warning && <div className="mb-3"><Notice tone="warn">{String(result.reference.warning)}</Notice></div>}
        {result.session.excluded && <div className="mb-3"><Notice tone="warn">This session is excluded from training/evaluation: {result.session.exclusion_reason}</Notice></div>}
        <ScoreTimeline points={result.timeline} threshold={result.threshold} phases={result.session.phases} onSelect={onExplain} />
        <p className="mt-2 text-xs text-ink-soft">Reference: <b>{result.reference.type}</b> {result.reference.description ? `- ${String(result.reference.description)}` : ""}
          {result.reference.period_start_s != null && ` - baseline ${mmss(Number(result.reference.period_start_s))}–${mmss(Number(result.reference.period_end_s))} of ${String(result.reference.session)}`}. Click a point to explain it.</p>
        <div className="mt-4">
          <div className="label">Data-quality indicator (windows with a modality below 80% valid samples)</div>
          <div className="flex h-3 overflow-hidden rounded bg-emerald-100" aria-label={`${lowQuality.length} low-quality windows`}>
            {result.timeline.map((p) => {
              const bad = lowQuality.includes(p);
              return <div key={p.window_index} className={bad ? "bg-amber-500" : ""} style={{ flex: 1 }} title={bad ? `window ${p.window_index}: reduced quality` : undefined} />;
            })}
          </div>
          <p className="mt-1 text-xs text-ink-faint">{lowQuality.length} of {result.timeline.length} windows have reduced data quality (amber).</p>
        </div>
      </Card>
      {signals.data && (
        <Card title="Sensor signals for context">
          <div className="grid gap-4 lg:grid-cols-2">
            {signals.data.signals.map((s) => <SignalChart key={s} points={signals.data!.points} signal={s} unit={signals.data!.units[s]} phases={signals.data!.phases} calibrationEnd={signals.data!.calibration_end_s} height={150} />)}
          </div>
        </Card>
      )}
      <Card title={`Flagged windows (${flagged.length})`}>
        {flagged.length === 0 ? <p className="text-sm text-ink-soft">No window exceeded the threshold.</p> : (
          <div className="max-h-80 overflow-auto">
            <table className="table-base">
              <thead><tr><th>Window</th><th>Time</th><th>Phase</th><th>Score</th><th>Quality</th><th /></tr></thead>
              <tbody>
                {flagged.map((p) => (
                  <tr key={p.window_index}>
                    <td>{p.window_index}</td><td>{mmss(p.start_s)}–{mmss(p.end_s)}</td><td>{p.phase ?? "-"}</td><td>{fmt(p.score)}</td>
                    <td>{lowQuality.includes(p) ? <Badge tone="degraded">reduced</Badge> : <Badge tone="good">good</Badge>}</td>
                    <td><button className="btn-secondary px-2 py-1 text-xs" onClick={() => onExplain(p.window_index)}>Explain</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </>
  );
}
