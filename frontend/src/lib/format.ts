export const fmt = (v: number | null | undefined, digits = 3): string =>
  v === null || v === undefined || Number.isNaN(v) ? "n/a" : v.toFixed(digits);

export const pct = (v: number | null | undefined, digits = 1): string =>
  v === null || v === undefined || Number.isNaN(v) ? "n/a" : `${(v * 100).toFixed(digits)}%`;

export const int = (v: number | null | undefined): string =>
  v === null || v === undefined ? "n/a" : v.toLocaleString();

export const minutes = (seconds: number): string => `${(seconds / 60).toFixed(1)} min`;

export const mmss = (seconds: number): string => {
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
};

export const dateTime = (iso: string | null | undefined): string => {
  if (!iso) return "n/a";
  const d = new Date(iso.endsWith("Z") || iso.includes("+") ? iso : `${iso}Z`);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
};

/** Human-readable method label, e.g. "iforest_personal" -> "Isolation Forest · personal". */
export function methodLabel(m: string): string {
  const names: Record<string, string> = {
    hr_rule: "HR > 100 bpm rule",
    robust_z: "Robust z-score",
    arousal_z: "Arousal-directed z",
    iforest: "Isolation Forest",
    ocsvm: "One-Class SVM",
    autoencoder: "Autoencoder",
  };
  if (m === "hr_rule") return names.hr_rule;
  const idx = m.lastIndexOf("_");
  const scorer = m.slice(0, idx);
  const norm = m.slice(idx + 1);
  return `${names[scorer] ?? scorer} · ${norm}`;
}

export const SIGNAL_COLORS: Record<string, string> = {
  EDA: "#2a8c80",
  HR: "#c2410c",
  TEMP: "#7c3aed",
  BVP: "#0369a1",
  ACC: "#64748b",
};
