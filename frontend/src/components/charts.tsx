import {
  CartesianGrid, Line, LineChart, ReferenceArea, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis,
  YAxis, ZAxis,
} from "recharts";
import type { Phase } from "../lib/types";
import { SIGNAL_COLORS, fmt } from "../lib/format";

/** Shades protocol phases: stressor tasks in orange, rest phases in blue. */
export function phaseAreas(phases: Phase[] | null | undefined, xScale = 60) {
  return (phases ?? [])
    .filter((p) => p.label !== null)
    .map((p, i) => (
      <ReferenceArea
        key={`${p.name}-${i}`}
        x1={p.start_s / xScale}
        x2={p.end_s / xScale}
        fill={p.label === 1 ? "#fb923c" : "#38bdf8"}
        fillOpacity={0.12}
        ifOverflow="hidden"
      />
    ));
}

export function PhaseLegend() {
  return (
    <div className="flex flex-wrap items-center gap-4 text-xs text-ink-soft">
      <span className="flex items-center gap-1.5"><span className="h-3 w-3 rounded-sm bg-orange-300/60" /> stressor task (protocol)</span>
      <span className="flex items-center gap-1.5"><span className="h-3 w-3 rounded-sm bg-sky-300/60" /> rest / baseline (protocol)</span>
      <span className="flex items-center gap-1.5"><span className="h-0.5 w-4 bg-slate-500" /> end of personal-baseline period</span>
    </div>
  );
}

interface SignalChartProps {
  points: Record<string, number | null>[];
  signal: string;
  unit: string;
  phases?: Phase[] | null;
  calibrationEnd?: number | null;
  height?: number;
}

export function SignalChart({ points, signal, unit, phases, calibrationEnd, height = 170 }: SignalChartProps) {
  const data = points.map((p) => ({ t: (p.t_s as number) / 60, v: p[signal] }));
  return (
    <figure>
      <figcaption className="mb-1 flex items-center justify-between text-xs font-semibold text-ink-soft">
        <span>{signal}</span>
        <span className="font-normal">{unit}</span>
      </figcaption>
      <div style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 4, right: 12, bottom: 4, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            {phaseAreas(phases)}
            {calibrationEnd != null && <ReferenceLine x={calibrationEnd / 60} stroke="#475569" strokeDasharray="4 3" />}
            <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} tickFormatter={(v) => `${Math.round(v)}`} fontSize={11}
              label={{ value: "minutes since session start", position: "insideBottom", offset: -2, fontSize: 10, fill: "#7b8794" }} height={32} />
            <YAxis fontSize={11} width={52} domain={["auto", "auto"]} tickFormatter={(v) => fmt(v, Math.abs(v) < 10 ? 2 : 0)} />
            <Tooltip formatter={(v: number) => [`${fmt(v, 3)} ${unit}`, signal]} labelFormatter={(l: number) => `t = ${l.toFixed(2)} min`} />
            <Line type="monotone" dataKey="v" stroke={SIGNAL_COLORS[signal] ?? "#334155"} dot={false} strokeWidth={1.4} connectNulls={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </figure>
  );
}

export function ScoreTimeline({ points, threshold, phases, onSelect, height = 260 }: {
  points: { start_s: number; score: number | null; flagged: boolean; window_index: number }[];
  threshold: number; phases?: Phase[] | null; onSelect?: (windowIndex: number) => void; height?: number;
}) {
  const normal = points.filter((p) => p.score !== null && !p.flagged).map((p) => ({ t: p.start_s / 60, s: p.score, w: p.window_index }));
  const flagged = points.filter((p) => p.flagged).map((p) => ({ t: p.start_s / 60, s: p.score, w: p.window_index }));
  return (
    <div style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 8, right: 12, bottom: 8, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          {phaseAreas(phases)}
          <XAxis dataKey="t" type="number" name="time" unit=" min" domain={["dataMin", "dataMax"]} fontSize={11} tickFormatter={(v) => `${Math.round(v)}`} />
          <YAxis dataKey="s" type="number" name="score" fontSize={11} width={52} tickFormatter={(v) => fmt(v, 2)} />
          <ZAxis range={[18, 18]} />
          <ReferenceLine y={threshold} stroke="#be123c" strokeDasharray="5 4" label={{ value: "threshold", position: "right", fontSize: 10, fill: "#be123c" }} />
          <Tooltip cursor={{ strokeDasharray: "3 3" }} formatter={(v: number, n: string) => [n === "time" ? `${v.toFixed(2)} min` : fmt(v, 3), n]} />
          <Scatter name="below threshold" data={normal} fill="#2a8c80" onClick={(d: { w: number }) => onSelect?.(d.w)} />
          <Scatter name="flagged" data={flagged} fill="#be123c" onClick={(d: { w: number }) => onSelect?.(d.w)} />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

export function CurveChart({ x, y, xLabel, yLabel, diagonal = false, height = 240 }: { x: number[]; y: number[]; xLabel: string; yLabel: string; diagonal?: boolean; height?: number }) {
  const data = x.map((v, i) => ({ x: v, y: y[i] }));
  return (
    <div style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 12, bottom: 18, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis dataKey="x" type="number" domain={[0, 1]} fontSize={11} label={{ value: xLabel, position: "insideBottom", offset: -8, fontSize: 11 }} />
          <YAxis domain={[0, 1]} fontSize={11} width={40} label={{ value: yLabel, angle: -90, position: "insideLeft", fontSize: 11 }} />
          {diagonal && <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#94a3b8" strokeDasharray="4 4" />}
          <Tooltip formatter={(v: number) => fmt(v, 3)} />
          <Line dataKey="y" stroke="#1f6f67" dot={false} strokeWidth={1.8} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
