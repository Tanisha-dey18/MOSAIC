// Typed client for the MOSAIC FastAPI backend.
// The backend address is resolved at runtime (see backend.ts): link parameter, remembered choice,
// VITE_API_BASE_URL, or localhost during development.
import type {
  AnomalyEvent, BaselineResponse, DatasetMetadata, DatasetRecord, DetectionResult, Experiment, ExperimentListItem,
  Explanation, Health, MethodInfo, ModelMetrics, ModelVersion, Page, ParticipantsResponse, QualityResponse, Report,
  SessionInfo, SignalsResponse, TrainingRun,
} from "./types";

import { getApiBase } from "./backend";

/** Base URL for links (figures, downloads); empty when no backend is configured. */
const base = () => getApiBase() ?? "";

export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string, public details?: unknown) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const apiBase = getApiBase();
  if (!apiBase) {
    throw new ApiError(0, "backend_not_configured", "This site is not connected to a MOSAIC backend yet.");
  }
  let res: Response;
  try {
    res = await fetch(`${apiBase}${path}`, init);
  } catch {
    throw new ApiError(0, "network_error", `Cannot reach the MOSAIC API at ${apiBase}. Is the backend running?`);
  }
  if (!res.ok) {
    let code = "http_error";
    let message = `${res.status} ${res.statusText}`;
    let details: unknown;
    try {
      const body = await res.json();
      code = body?.error?.code ?? code;
      message = body?.error?.message ?? message;
      details = body?.error?.details;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, code, message, details);
  }
  return res.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

function qs(params: Record<string, string | number | undefined | null>): string {
  const p = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") p.set(k, String(v));
  });
  const s = p.toString();
  return s ? `?${s}` : "";
}

export const api = {
  health: () => request<Health>("/api/health"),
  metadata: () => request<DatasetMetadata>("/api/dataset/metadata"),
  datasets: () => request<DatasetRecord[]>("/api/datasets"),
  dataset: (id: number) => request<DatasetRecord>(`/api/datasets/${id}`),
  sources: () => request<{ sources: { path: string; type: string; protocols: string[] }[] }>("/api/dataset/sources"),
  ingest: (path: string, protocols?: string[]) => request<DatasetRecord>("/api/dataset/ingest", json({ path, protocols })),
  validateFile: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return request<Record<string, unknown>>("/api/dataset/validate", { method: "POST", body: fd });
  },
  uploadFile: (file: File, name: string) => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("name", name);
    return request<DatasetRecord>("/api/dataset/upload", { method: "POST", body: fd });
  },
  quality: () => request<QualityResponse>("/api/dataset/quality"),
  sessions: (params: { participant?: string; protocol?: string } = {}) => request<SessionInfo[]>(`/api/sessions${qs(params)}`),
  signals: (key: string, params: { start_s?: number; end_s?: number; signals?: string; max_points?: number } = {}) =>
    request<SignalsResponse>(`/api/sessions/${encodeURIComponent(key)}/signals${qs(params)}`),
  participants: () => request<ParticipantsResponse>("/api/participants"),
  baseline: (pid: string, modelId?: number) =>
    request<BaselineResponse>(`/api/participants/${encodeURIComponent(pid)}/baseline${qs({ model_id: modelId })}`),
  methods: () => request<{ methods: MethodInfo[] }>("/api/models/methods"),
  train: (body: { method: string; modalities?: string[]; threshold_quantile?: number; n_folds?: number; seed?: number; params?: Record<string, number | string> }) =>
    request<TrainingRun>("/api/models/train", json(body)),
  trainingRuns: (page = 1) => request<Page<TrainingRun>>(`/api/models/training-runs${qs({ page, page_size: 50 })}`),
  models: () => request<ModelVersion[]>("/api/models"),
  modelMetrics: (id: number) => request<ModelMetrics>(`/api/models/${id}/metrics`),
  detect: (model_id: number, session_key: string, threshold?: number) =>
    request<DetectionResult>("/api/anomalies/detect", json({ model_id, session_key, threshold })),
  anomalies: (params: { model_id?: number; participant?: string; session_key?: string; page?: number; page_size?: number } = {}) =>
    request<Page<AnomalyEvent>>(`/api/anomalies${qs(params)}`),
  anomaly: (id: number) => request<{ id: number; session_key: string; participant: string; explanation: Explanation }>(`/api/anomalies/${id}`),
  detections: () => request<Page<{ id: number; session_key: string; model_id: number; method: string; version: number; threshold: number; n_windows: number; n_scored: number; n_flagged: number; created_at: string }>>("/api/anomalies/detections"),
  detection: (id: number) => request<DetectionResult>(`/api/anomalies/detections/${id}`),
  explain: (model_id: number, session_key: string, window_index: number) =>
    request<Explanation>("/api/explain", json({ model_id, session_key, window_index })),
  experiments: () => request<{ runs: ExperimentListItem[] }>("/api/experiments"),
  experiment: (id: string) => request<Experiment>(`/api/experiments/${encodeURIComponent(id)}`),
  figureUrl: (runId: string, name: string) => `${base()}/api/experiments/${encodeURIComponent(runId)}/figures/${encodeURIComponent(name)}`,
  runExperiments: (quick: boolean) => request<{ run_id: string; status: string }>("/api/experiments/run", json({ quick })),
  reports: () => request<Page<Report>>("/api/reports"),
  createReport: (title: string, experiment_run_id?: string) => request<Report>("/api/reports", json({ title, experiment_run_id, include_models: true })),
  reportUrl: (id: number) => `${base()}/api/reports/${id}/download`,
};
