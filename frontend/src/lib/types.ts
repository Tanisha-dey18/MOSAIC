// Response types mirroring backend/app/schemas and service payloads.

export interface Page<T> { items: T[]; total: number; page: number; page_size: number }

export interface Health {
  status: "ok" | "degraded";
  version: string;
  environment: string;
  database: { ok: boolean; error: string | null };
  storage: { ok: boolean };
  dataset: { configured: boolean; id?: number; name?: string; version?: string };
  models: { count: number };
}

export interface DatasetRecord {
  id: number; name: string; slug: string; version: string; doi: string | null; license: string | null;
  source_type: string; status: "pending" | "processing" | "ready" | "failed"; progress: string | null; error: string | null;
  n_sessions: number; n_participants: number; n_windows: number; feature_config: Record<string, unknown> | null;
  created_at: string; updated_at: string;
}

export interface FeatureInfo { name: string; modality: string; unit: string; description: string }

export interface DatasetMetadata {
  configured: boolean;
  dataset?: DatasetRecord;
  latest_attempt?: DatasetRecord | null;
  provenance?: Record<string, unknown> & { citation?: string; url?: string; device?: string; modalities?: Record<string, string> };
  processed_at?: string;
  modalities?: string[];
  sampling_rates?: Record<string, number[]>;
  protocols?: string[];
  features?: FeatureInfo[];
  ingestion_errors?: string[];
  how_to_import: { steps: string[]; dataset_url: string };
}

export interface ModalityQuality { status: string; missing_fraction: number; out_of_range_fraction: number; flatline_fraction: number; issues: string[] }

export interface QualityResponse {
  dataset_id: number; n_sessions: number; status_counts: Record<string, number>;
  excluded_sessions: { session_key: string; reason: string }[];
  per_modality: Record<string, { mean_valid_fraction: number; windows_below_threshold: number }>;
  feature_missing_fraction: Record<string, number>;
  sessions: { session_key: string; participant: string; protocol: string; quality_status: string; excluded: boolean; modalities: Record<string, ModalityQuality> }[];
}

export interface Phase { name: string; start_s: number; end_s: number; label: number | null }

export interface SessionInfo {
  id: number; session_key: string; protocol: string; duration_s: number; origin: string | null; quality_status: string;
  excluded: boolean; exclusion_reason: string | null; n_windows: number; n_calibration_windows: number; n_eval_windows: number;
  calibration_end_s: number | null; modalities: string[] | null; sampling_rates: Record<string, number> | null;
  quality: Record<string, ModalityQuality & { fs: number; duration_s: number; n_samples: number }> | null;
  phases: Phase[] | null; notes: string[] | null; participant: string | null;
}

export interface SignalsResponse {
  session_key: string; protocol: string; origin: string | null; resolution_s: number; signals: string[];
  units: Record<string, string>; sampling_rates: Record<string, number> | null; missing_fraction: Record<string, number>;
  phases: Phase[]; calibration_end_s: number | null; points: Record<string, number | null>[];
}

export interface ParticipantSummary {
  participant: string; protocol_version: string | null; total_minutes: number; complete_calibration_windows: number;
  baseline_status: "available" | "insufficient" | "no_calibration_period"; min_calibration_windows: number;
  sessions: { session_key: string; protocol: string; duration_s: number; quality_status: string; excluded: boolean }[];
}
export interface ParticipantsResponse { dataset_id: number; participants: ParticipantSummary[] }

export interface BaselineResponse {
  participant: string; protocol_version: string | null; reference_source: string; model_id: number | null; stored_profile: boolean;
  status: "ok" | "insufficient"; reason?: string; warning?: string; construction_rule: string;
  coverage: { session_key: string; protocol: string; duration_min: number; calibration_end_s: number | null; n_calibration_windows: number; n_eval_windows: number; excluded: boolean }[];
  n_windows?: number; baseline_session?: string;
  period?: { start_s: number; end_s: number; start_time: string; end_time: string };
  features?: { feature: string; unit: string; baseline_median: number | null; baseline_scale: number | null; population_median: number | null; after_calibration_median: number | null; after_calibration_z: number | null }[];
  deviation_trend?: { session_key: string; start_s: number; phase: string | null; is_calibration: boolean; deviation: number | null }[];
  deviation_definition?: string;
}

export interface MethodInfo { method: string; scorer: string; normalization: string; description: string; limitations: string; analysis: string }

export interface TrainingRun {
  id: number; dataset_id: number; method: string; status: "queued" | "running" | "completed" | "failed"; progress: string | null;
  error: string | null; features: string[]; params: Record<string, unknown> | null; config: Record<string, unknown>;
  created_at: string; started_at: string | null; finished_at: string | null; model_version_id: number | null;
}

export interface Metrics {
  roc_auc: number | null; pr_auc: number | null; participant_auc_median: number | null; participant_auc_iqr?: number[] | null;
  precision: number | null; recall: number | null; f1: number | null; fpr: number | null; false_alarms_per_hour: number | null;
  event_sensitivity: number | null; prevalence: number | null; n_windows: number; n_participants: number;
  tp: number; fp: number; tn: number; fn: number; ci95?: Record<string, number[]>;
}

export interface ModelVersion {
  id: number; training_run_id: number; dataset_id: number; method: string; scorer: string; normalization: string; version: number;
  feature_hash: string; features: string[]; threshold: number | null; threshold_quantile: number | null; seed: number;
  created_at: string; metrics: Metrics | null; artifact_ok: boolean | null;
}

export interface Curves { roc?: { fpr: number[]; tpr: number[] }; pr?: { precision: number[]; recall: number[] } }

export interface ModelMetrics {
  model_id: number; method: string; version: number; description: string; limitations: string;
  evaluations: { id: number; kind: string; metrics: Metrics | null; curves: Curves | null; per_participant: Record<string, number> | null; details: Record<string, unknown> | null; created_at: string }[];
}

export interface TimelinePoint {
  window_index: number; start_s: number; end_s: number; start_time: string | null; phase: string | null; label: number | null;
  is_calibration: boolean; score: number | null; flagged: boolean; [quality: string]: unknown;
}

export interface DetectionResult {
  detection_id: number; mode: string; mode_note: string;
  model: { id: number; method: string; version: number; threshold: number | null };
  session: { session_key: string; protocol: string; quality_status: string; excluded: boolean; exclusion_reason: string | null; phases: Phase[] | null };
  threshold: number; n_windows: number; n_scored: number; n_flagged: number;
  score_summary: { median: number | null; p95: number | null; max: number | null };
  reference: Record<string, unknown> & { type: string; status: string; reason?: string; warning?: string };
  timeline: TimelinePoint[]; event_ids: number[]; created_at: string;
}

export interface AnomalyEvent {
  id: number; detection_run_id: number; model_version_id: number; session_key: string | null; participant: string | null;
  window_index: number; start_s: number; end_s: number; start_time: string | null; score: number; threshold: number;
  phase: string | null; quality: Record<string, number | null> | null; created_at: string;
}

export interface ExplanationFeature {
  feature: string; modality: string; unit: string; description: string; value: number | null; reference_center: number | null;
  reference_scale: number | null; z: number | null; contribution: number | null; contribution_share: number | null;
}

export interface Explanation {
  model: { id: number; method: string; version: number; scorer: string; normalization: string; limitations: string };
  session_key: string; window_index: number; start_s: number; end_s: number; start_time: string | null; phase: string | null;
  label: number | null; score: number | null; threshold: number | null; flagged: boolean; quality: Record<string, number | null>;
  reference: Record<string, unknown> & { type: string; status: string };
  explanation_method: string; contributions_supported: boolean; features: ExplanationFeature[]; caveats: string[];
}

export interface ExperimentListItem {
  run_id: string; status: string; generated_at?: string; dataset?: string; n_participants?: number;
  n_eval_windows?: number; runtime_seconds?: number; started_at?: string; log_tail?: string;
}

export interface PairedComparison {
  a: string; b: string; n_participants: number; pooled_auc_diff: number | null; pooled_auc_diff_ci95: (number | null)[];
  participant_auc_diff_median: number | null; n_participants_b_better: number; wilcoxon: { statistic: number; p_value: number } | null;
}

export interface Experiment {
  run_id: string; generated_at: string; dataset: string; dataset_doi: string; config: Record<string, unknown>;
  eligibility: { eligible: string[]; ineligible: Record<string, string> }; n_eval_windows: number;
  label_counts: { stressor: number; rest: number };
  primary: { metrics: Record<string, Metrics> }; secondary: { metrics: Record<string, Metrics> };
  curves: Record<string, Curves>; per_participant_auc: Record<string, Record<string, number | null>>;
  paired: PairedComparison[]; timing: Record<string, { fit_seconds_mean: number; infer_ms_per_window: number }>;
  threshold_sensitivity: { method: string; quantile: number; precision: number; recall: number; f1: number; fpr: number; false_alarms_per_hour: number; event_sensitivity: number }[];
  modality_ablation: { feature_set: string; method: string; n_features: number; roc_auc: number; pr_auc: number; participant_auc_median: number; f1: number }[];
  calibration_length: { calibration_s: number; method: string; roc_auc: number; participant_auc_median: number; f1: number; false_alarms_per_hour: number }[];
  dropout: { scenario: string; method: string; roc_auc: number; f1: number }[];
  self_report: { available: boolean; note?: string; methods?: Record<string, { n_pairs: number; pooled_spearman_rho: number; pooled_p_value: number; within_participant_rho_median: number | null }> };
  error_analysis: Record<string, { flag_rate_by_phase: Record<string, number>; median_acc_std_false_positives: number | null; median_acc_std_true_negatives: number | null }>;
  figures: Record<string, string>; environment: string; runtime_seconds: number;
}

export interface Report { id: number; title: string; experiment_run: string | null; bytes: number; sha256: string; created_at: string }
