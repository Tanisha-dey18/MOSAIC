import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, Line, LineChart, ReferenceArea, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ApiError, api } from "../lib/api";
import { useApi } from "../lib/hooks";
import { SIGNAL_COLORS, fmt, methodLabel, mmss, pct } from "../lib/format";
import type { Explanation } from "../lib/types";
import { Badge, Card, EmptyState, ErrorState, Field, KeyValue, Loading, Notice, PageHeader } from "../components/ui";

export default function Explainability() {
  const [params, setParams] = useSearchParams();
  const models = useApi(() => api.models(), []);
  const [form, setForm] = useState({ model: params.get("model") ?? "", session: params.get("recording") ?? "", window: params.get("window") ?? "" });
  const [exp, setExp] = useState<Explanation | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const anomaly = params.get("anomaly");
    const run = async () => {
      setLoading(true);
      setError(null);
      try {
        if (anomaly) {
          const a = await api.anomaly(Number(anomaly));
          setExp(a.explanation);
          setForm({ model: String(a.explanation.model.id), session: a.session_key, window: String(a.explanation.window_index) });
        } else if (params.get("model") && params.get("recording") && params.get("window")) {
          setForm({ model: params.get("model")!, session: params.get("recording")!, window: params.get("window")! });
          setExp(await api.explain(Number(params.get("model")), params.get("recording")!, Number(params.get("window"))));
        }
      } catch (e) {
        setError(e as ApiError);
      } finally {
        setLoading(false);
      }
    };
    run();
  }, [params]);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    setParams({ model: form.model, recording: form.session, window: form.window });
  };

  return (
    <div className="space-y-6">
      <PageHeader title="Explainability" subtitle="Why did a window receive its score? Each explanation shows the observation, its reference, per-feature contributions computed by the model, data quality and the explanation's limitations." />
      <Notice tone="warn">Explanations describe how a model computed a score. They show <b>association, not causation</b>, and do not establish any physiological or medical cause.</Notice>
      <Card title="Select a window">
        <form className="grid gap-4 md:grid-cols-4" onSubmit={submit}>
          <Field label="Model" htmlFor="em">
            <select id="em" className="input" value={form.model} onChange={(e) => setForm({ ...form, model: e.target.value })} required>
              <option value="">Choose…</option>
              {(models.data ?? []).map((m) => <option key={m.id} value={m.id}>{methodLabel(m.method)} v{m.version} (#{m.id})</option>)}
            </select>
          </Field>
          <Field label="Session key" htmlFor="es"><input id="es" className="input" value={form.session} placeholder="e.g. STRESS_S05" pattern="[A-Za-z0-9_\-]+" onChange={(e) => setForm({ ...form, session: e.target.value })} required /></Field>
          <Field label="Window index" htmlFor="ew"><input id="ew" className="input" type="number" min={0} value={form.window} onChange={(e) => setForm({ ...form, window: e.target.value })} required /></Field>
          <div className="flex items-end"><button className="btn-primary w-full">Explain</button></div>
        </form>
        <p className="mt-2 text-xs text-ink-faint">Tip: in Anomaly Monitoring, click any point of the score timeline to open its explanation here.</p>
      </Card>
      {loading && <Loading />}
      {error && <ErrorState error={error} />}
      {!loading && !error && !exp && <EmptyState title="No window selected">Choose a model, a session and a window, or open a detection from Anomaly Monitoring.</EmptyState>}
      {exp && <ExplanationView exp={exp} />}
    </div>
  );
}

function ExplanationView({ exp }: { exp: Explanation }) {
  const sig = useApi(() => api.signals(exp.session_key, { start_s: Math.max(0, exp.start_s - 180), end_s: exp.end_s + 180 }).catch(() => null), [exp.session_key, exp.start_s]);
  const contrib = exp.features.filter((f) => f.contribution !== null).map((f) => ({ f: f.feature, c: f.contribution! }));
  return (
    <>
      <div className="grid gap-6 lg:grid-cols-3">
        <Card title="Window summary" actions={exp.flagged ? <Badge tone="failed">above threshold</Badge> : <Badge tone="ok">below threshold</Badge>}>
          <KeyValue items={[
            ["Model", `${methodLabel(exp.model.method)} v${exp.model.version} (#${exp.model.id})`],
            ["Session / window", `${exp.session_key} / #${exp.window_index}`],
            ["Time", `${mmss(exp.start_s)}–${mmss(exp.end_s)} (${exp.start_time ?? "n/a"})`],
            ["Protocol phase", `${exp.phase ?? "unlabelled"}${exp.label === 1 ? " (stressor)" : exp.label === 0 ? " (rest)" : ""}`],
            ["Anomaly score", fmt(exp.score)],
            ["Applied threshold", fmt(exp.threshold)],
            ["Reference", `${exp.reference.type}${exp.reference.status !== "ok" ? " - unavailable" : ""}`],
          ]} />
          <div className="mt-4">
            <div className="label">Data quality (valid-sample fraction)</div>
            <div className="flex flex-wrap gap-2">
              {Object.entries(exp.quality).map(([m, q]) => <Badge key={m} tone={q !== null && q >= 0.8 ? "good" : "degraded"}>{`${m}: ${pct(q, 0)}`}</Badge>)}
            </div>
          </div>
        </Card>
        <Card title="Feature contributions" className="lg:col-span-2" subtitle={exp.explanation_method}>
          {!exp.contributions_supported || contrib.length === 0 ? <p className="text-sm text-ink-soft">This model does not support feature-level contributions.</p> : (
            <div style={{ height: Math.max(220, contrib.length * 24) }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={contrib} layout="vertical" margin={{ top: 4, right: 16, bottom: 4, left: 16 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis type="number" fontSize={11} />
                  <YAxis type="category" dataKey="f" width={100} fontSize={11} />
                  <Tooltip formatter={(v: number) => fmt(v, 4)} />
                  <Bar dataKey="c" fill="#1f6f67" name="contribution" isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </Card>
      </div>
      <Card title="Observation vs reference" subtitle="z = (value − reference centre) / reference scale">
        <div className="overflow-x-auto">
          <table className="table-base">
            <thead><tr><th>Feature</th><th>Modality</th><th>Value</th><th>Reference centre</th><th>Reference scale</th><th>z</th><th>Contribution share</th></tr></thead>
            <tbody>
              {exp.features.map((f) => (
                <tr key={f.feature} title={f.description}>
                  <td className="font-mono text-xs">{f.feature}</td><td>{f.modality}</td><td>{fmt(f.value)} <span className="text-xs text-ink-faint">{f.unit}</span></td>
                  <td>{fmt(f.reference_center)}</td><td>{fmt(f.reference_scale)}</td><td>{fmt(f.z, 2)}</td><td>{pct(f.contribution_share)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      <Card title="Signals around the window (±3 min)" subtitle="Shaded band = the explained window">
        {sig.loading ? <Loading /> : !sig.data ? <p className="text-sm text-ink-soft">Signals unavailable.</p> : (
          <div className="grid gap-4 md:grid-cols-2">
            {sig.data.signals.map((s) => {
              const data = sig.data!.points.map((p) => ({ t: (p.t_s as number) / 60, v: p[s] }));
              return (
                <figure key={s}>
                  <figcaption className="mb-1 text-xs font-semibold text-ink-soft">{s} ({sig.data!.units[s]})</figcaption>
                  <div style={{ height: 140 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                        <ReferenceArea x1={exp.start_s / 60} x2={exp.end_s / 60} fill="#be123c" fillOpacity={0.12} />
                        <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} fontSize={10} tickFormatter={(v) => v.toFixed(1)} />
                        <YAxis fontSize={10} width={48} domain={["auto", "auto"]} tickFormatter={(v) => fmt(v, 2)} />
                        <Tooltip formatter={(v: number) => fmt(v, 3)} labelFormatter={(l: number) => `${l.toFixed(2)} min`} />
                        <Line dataKey="v" stroke={SIGNAL_COLORS[s]} dot={false} isAnimationActive={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </figure>
              );
            })}
          </div>
        )}
      </Card>
      <Card title="Limitations of this explanation">
        <ul className="list-disc space-y-1 pl-5 text-sm text-ink-soft">
          <li>{exp.model.limitations}</li>
          {exp.caveats.map((c) => <li key={c}>{c}</li>)}
        </ul>
      </Card>
    </>
  );
}
