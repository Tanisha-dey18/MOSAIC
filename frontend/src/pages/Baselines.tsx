import { useEffect, useState } from "react";
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../lib/api";
import { useApi } from "../lib/hooks";
import { fmt, methodLabel, mmss } from "../lib/format";
import { PhaseLegend, phaseAreas } from "../components/charts";
import { Badge, Card, ErrorState, Field, KeyValue, Loading, Notice, PageHeader } from "../components/ui";

export default function Baselines() {
  const parts = useApi(() => api.participants(), []);
  const models = useApi(() => api.models(), []);
  const [pid, setPid] = useState("");
  const [modelId, setModelId] = useState<number | "">("");
  useEffect(() => { if (!pid && parts.data?.participants.length) setPid(parts.data.participants[0].participant); }, [parts.data, pid]);
  const personalModels = (models.data ?? []).filter((m) => m.normalization === "personal");

  if (parts.loading) return <Loading />;
  if (parts.error) return <ErrorState error={parts.error} onRetry={parts.reload} />;
  const list = parts.data!.participants;
  const counts = list.reduce<Record<string, number>>((acc, p) => ({ ...acc, [p.baseline_status]: (acc[p.baseline_status] ?? 0) + 1 }), {});

  return (
    <div className="space-y-6">
      <PageHeader title="Personalised baselines" subtitle="Each participant's own reference state, built only from data recorded before their first stressor task. Later observations are compared against it; nothing from the evaluation period is used to construct it." />
      <div className="flex flex-wrap gap-2 text-sm">
        {Object.entries(counts).map(([k, v]) => <Badge key={k} tone={k}>{`${k.replace(/_/g, " ")}: ${v}`}</Badge>)}
      </div>
      <div className="grid gap-6 lg:grid-cols-3">
        <Card title="Participants" subtitle="Baseline availability (all features complete)">
          <div className="max-h-[560px] overflow-y-auto">
            <table className="table-base">
              <thead><tr><th>ID</th><th>Baseline windows</th><th>Status</th></tr></thead>
              <tbody>
                {list.map((p) => (
                  <tr key={p.participant} className={`cursor-pointer ${p.participant === pid ? "bg-brand-50" : "hover:bg-slate-50"}`} onClick={() => setPid(p.participant)}>
                    <td><button className="font-medium text-brand-800 hover:underline" onClick={() => setPid(p.participant)}>{p.participant}</button><div className="text-xs text-ink-faint">{p.total_minutes} min recorded</div></td>
                    <td>{p.complete_calibration_windows} <span className="text-xs text-ink-faint">/ ≥{p.min_calibration_windows}</span></td>
                    <td><Badge tone={p.baseline_status}>{p.baseline_status.replace(/_/g, " ")}</Badge></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
        <div className="min-w-0 space-y-6 lg:col-span-2">
          <Card title="Reference model">
            <Field label="Personal model (optional)" htmlFor="pm" hint="Without a trained personal model, descriptive statistics from all participants are used for the shrinkage prior.">
              <select id="pm" className="input" value={modelId} onChange={(e) => setModelId(e.target.value === "" ? "" : Number(e.target.value))}>
                <option value="">Latest personal model (or descriptive)</option>
                {personalModels.map((m) => <option key={m.id} value={m.id}>{methodLabel(m.method)} v{m.version} (#{m.id})</option>)}
              </select>
            </Field>
          </Card>
          {pid && <BaselineDetail pid={pid} modelId={modelId === "" ? undefined : modelId} />}
        </div>
      </div>
    </div>
  );
}

function BaselineDetail({ pid, modelId }: { pid: string; modelId?: number }) {
  const b = useApi(() => api.baseline(pid, modelId), [pid, modelId]);
  const stressSession = b.data?.coverage.find((c) => c.protocol === "STRESS" && !c.excluded)?.session_key;
  const sess = useApi(() => (stressSession ? api.sessions({ participant: pid }) : Promise.resolve([])), [pid, stressSession]);
  if (b.loading) return <Loading />;
  if (b.error) return <ErrorState error={b.error} />;
  const d = b.data!;
  const phases = sess.data?.find((s) => s.session_key === stressSession)?.phases;
  const trend = (d.deviation_trend ?? []).filter((t) => t.session_key === stressSession).map((t) => ({ t: t.start_s / 60, v: t.deviation, cal: t.is_calibration }));

  return (
    <>
      <Card title={`Participant ${d.participant}`} actions={<Badge tone={d.status}>{d.status === "ok" ? "baseline available" : "insufficient history"}</Badge>}>
        {d.status !== "ok" && <div className="mb-4"><Notice tone="warn"><b>Insufficient history.</b> {d.reason}. {d.warning ?? ""}</Notice></div>}
        <KeyValue items={[
          ["Reference source", d.reference_source],
          ["Construction rule", d.construction_rule],
          ...(d.period ? [["Baseline period", `${mmss(d.period.start_s)}–${mmss(d.period.end_s)} of ${d.baseline_session} (${d.n_windows} windows)`] as [string, string]] : []),
          ["Stored with model", d.stored_profile ? "yes" : "no (computed on request)"],
        ]} />
        <div className="mt-4 overflow-x-auto">
          <table className="table-base">
            <thead><tr><th>Session</th><th>Protocol</th><th>Duration</th><th>Baseline windows</th><th>Evaluation windows</th></tr></thead>
            <tbody>
              {d.coverage.map((c) => (
                <tr key={c.session_key}><td>{c.session_key} {c.excluded && <Badge tone="excluded">excluded</Badge>}</td><td>{c.protocol}</td><td>{c.duration_min} min</td><td>{c.n_calibration_windows}</td><td>{c.n_eval_windows}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      {d.status === "ok" && d.features && (
        <>
          <Card title="Baseline statistics vs later observations" subtitle="z = (median after baseline - baseline median) / baseline scale. Descriptive comparison only.">
            <div className="overflow-x-auto">
              <table className="table-base">
                <thead><tr><th>Feature</th><th>Baseline median</th><th>Baseline scale</th><th>Population median</th><th>After-baseline median</th><th>z</th></tr></thead>
                <tbody>
                  {d.features.map((f) => (
                    <tr key={f.feature}>
                      <td className="font-mono text-xs">{f.feature} <span className="text-ink-faint">({f.unit})</span></td>
                      <td>{fmt(f.baseline_median)}</td><td>{fmt(f.baseline_scale)}</td><td>{fmt(f.population_median)}</td><td>{fmt(f.after_calibration_median)}</td>
                      <td className={f.after_calibration_z !== null && Math.abs(f.after_calibration_z) > 2 ? "font-semibold text-amber-700" : ""}>{fmt(f.after_calibration_z, 2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
          <Card title={`Deviation from personal baseline over time · ${stressSession}`} subtitle={d.deviation_definition} actions={<PhaseLegend />}>
            <div style={{ height: 240 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={trend} margin={{ top: 8, right: 12, bottom: 8, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  {phaseAreas(phases)}
                  {d.period && <ReferenceLine x={d.period.end_s / 60} stroke="#475569" strokeDasharray="4 3" />}
                  <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} fontSize={11} tickFormatter={(v) => `${Math.round(v)}`} />
                  <YAxis fontSize={11} width={44} />
                  <Tooltip formatter={(v: number) => fmt(v, 2)} labelFormatter={(l: number) => `${l.toFixed(1)} min`} />
                  <Line dataKey="v" stroke="#1f6f67" dot={false} strokeWidth={1.5} isAnimationActive={false} name="RMS personal z" />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <p className="mt-2 text-xs text-ink-faint">Minutes since session start. The dashed line marks the end of the baseline period.</p>
          </Card>
        </>
      )}
    </>
  );
}
