import { api } from "../lib/api";
import { useApi } from "../lib/hooks";
import { Card, KeyValue, Notice, PageHeader } from "../components/ui";

export default function About() {
  const meta = useApi(() => api.metadata(), []);
  const prov = meta.data?.provenance;
  return (
    <div className="space-y-6">
      <PageHeader title="Project information" subtitle="MOSAIC - Multimodal Observation System for Anomaly Identification and Clinical Trends. Final-year B.Tech CSE (Data Science) major project." />
      <Notice tone="warn">
        <b>Safety statement.</b> MOSAIC is academic research software. It is not a medical device, has not been clinically validated, and must not be used to diagnose
        disease, predict medical emergencies or replace professional medical assessment. No compliance with any healthcare regulation is claimed.
      </Notice>
      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Research question">
          <p className="text-sm">Does personalised baseline modelling improve physiological anomaly detection compared with global statistical thresholds and non-personalised machine-learning methods?</p>
          <p className="mt-2 text-xs text-ink-soft">Treated as a hypothesis. "Anomaly" is operationalised as deviation from a resting state, evaluated against protocol phases (stressor tasks vs rest) of an induced-stress experiment.</p>
          <h3 className="mt-4 text-sm font-semibold">Objectives</h3>
          <ol className="mt-1 list-decimal space-y-1 pl-5 text-sm">
            <li>Reliable ingestion of multimodal wearable signals with quality control.</li>
            <li>Compare interpretable statistical baselines with machine learning (Isolation Forest, One-Class SVM, autoencoder).</li>
            <li>Investigate participant-specific (personal) baselines without temporal leakage.</li>
            <li>Evaluate with participant-grouped cross-validation.</li>
            <li>Measure the contribution of each sensor modality.</li>
            <li>Provide interpretable, honest explanations.</li>
            <li>Deliver an integrated web application and reproducible experiments.</li>
          </ol>
        </Card>
        <Card title="Dataset provenance">
          {prov ? (
            <KeyValue items={[
              ["Dataset", String(prov.name ?? "")],
              ["Version", String(prov.version ?? "")],
              ["DOI", String(prov.doi ?? "")],
              ["License", String(prov.license ?? "")],
              ["Device", String(prov.device ?? "")],
              ["Citation", String(prov.citation ?? "")],
            ]} />
          ) : (
            <p className="text-sm text-ink-soft">No dataset imported. MOSAIC is designed for the PhysioNet <i>Wearable Device Dataset from Induced Stress and Structured Exercise Sessions</i> v1.0.1 (DOI 10.13026/he0v-tf17, ODC-By 1.0).</p>
          )}
        </Card>
        <Card title="Methodology">
          <ul className="list-disc space-y-1 pl-5 text-sm">
            <li>Signal-specific filtering (EDA low-pass, BVP band-pass, TEMP/HR median, ACC magnitude) and plausibility/flat-line checks.</li>
            <li>30 s windows (10 s hop) → 14 interpretable features across EDA, HR, TEMP, BVP and ACC.</li>
            <li>Personal baseline = last 5 min before the first stressor, scale shrunk towards the typical within-person scale.</li>
            <li>Global vs personal vs rolling normalisation × robust z / Isolation Forest / One-Class SVM / autoencoder.</li>
            <li>5-fold participant-grouped CV; thresholds from training participants' rest windows; bootstrap CIs by participant.</li>
            <li>Ablations: modalities, calibration length, threshold, sensor drop-out; computational cost.</li>
          </ul>
        </Card>
        <Card title="System architecture">
          <pre className="overflow-x-auto rounded-lg bg-slate-50 p-3 text-xs leading-relaxed">{`React + Vite + Tailwind (this UI)
        │  HTTPS / JSON  (VITE_API_BASE_URL)
FastAPI backend ── background job thread (ingest, train)
   │          │
   │          └─ ml/ package: ingestion → QC → filters → windows
   │             → features → normalisation → detectors → evaluation
   │
PostgreSQL (metadata, models, baselines, anomalies, reports)
File storage (Parquet features, 1 Hz signals, model artifacts, reports)`}</pre>
          <p className="mt-2 text-xs text-ink-soft">Full documentation: <code>docs/</code> in the repository (report, API reference, deployment, viva preparation). Interactive API docs at <code>/docs</code> on the backend.</p>
        </Card>
      </div>
    </div>
  );
}
