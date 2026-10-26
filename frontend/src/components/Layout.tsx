import { useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { api } from "../lib/api";
import BackendConnect from "./BackendConnect";
import { useApi } from "../lib/hooks";

const NAV = [
  { to: "/", label: "Dashboard", icon: "▦" },
  { to: "/data", label: "Data Explorer", icon: "≋" },
  { to: "/monitoring", label: "Anomaly Monitoring", icon: "⌁" },
  { to: "/baselines", label: "Personal Baselines", icon: "◎" },
  { to: "/models", label: "Model Laboratory", icon: "⚗" },
  { to: "/explain", label: "Explainability", icon: "◫" },
  { to: "/reports", label: "Research Reports", icon: "▤" },
  { to: "/about", label: "Project Information", icon: "ⓘ" },
];

export default function Layout() {
  const [open, setOpen] = useState(false);
  const [backendOpen, setBackendOpen] = useState(false);
  const health = useApi(() => api.health(), []);

  const nav = (
    <nav aria-label="Main navigation" className="flex flex-col gap-1">
      {NAV.map((n) => (
        <NavLink
          key={n.to}
          to={n.to}
          end={n.to === "/"}
          onClick={() => setOpen(false)}
          className={({ isActive }) =>
            `flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition ${
              isActive ? "bg-brand-700 text-white" : "text-slate-200 hover:bg-brand-800 hover:text-white"
            }`
          }
        >
          <span aria-hidden className="w-4 text-center">{n.icon}</span>
          {n.label}
        </NavLink>
      ))}
    </nav>
  );

  const status = health.error
    ? health.error.code === "backend_not_configured"
      ? { text: "Not connected", cls: "bg-slate-400" }
      : { text: "API unreachable", cls: "bg-rose-500" }
    : health.data
      ? { text: health.data.status === "ok" ? "API online" : "API degraded", cls: health.data.status === "ok" ? "bg-emerald-400" : "bg-amber-400" }
      : { text: "Connecting…", cls: "bg-slate-400" };

  return (
    <div className="min-h-screen lg:flex">
      <aside className="hidden w-64 shrink-0 flex-col bg-brand-900 px-4 py-5 lg:flex">
        <Brand />
        <div className="mt-8 flex-1">{nav}</div>
        <SafetyNote />
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-20 flex items-center justify-between gap-3 border-b border-slate-200 bg-white/90 px-4 py-3 backdrop-blur sm:px-6">
          <div className="flex items-center gap-3">
            <button className="btn-secondary px-2.5 lg:hidden" aria-label="Open navigation" aria-expanded={open} onClick={() => setOpen(true)}>
              ☰
            </button>
            <span className="text-sm font-semibold text-brand-800 lg:hidden">MOSAIC</span>
            <span className="hidden rounded-full bg-amber-50 px-2.5 py-1 text-xs font-medium text-amber-800 ring-1 ring-amber-200 sm:inline">
              Research use only · not a diagnostic tool
            </span>
          </div>
          <div className="flex items-center gap-4 text-xs text-ink-soft">
            {health.data?.dataset.configured && <span className="hidden md:inline">Dataset: {health.data.dataset.name?.slice(0, 40)}… v{health.data.dataset.version}</span>}
            <div className="relative">
              <button
                className="flex items-center gap-2 rounded-full px-2 py-1 hover:bg-slate-100"
                role="status"
                aria-expanded={backendOpen}
                title="Backend connection"
                onClick={() => setBackendOpen((v) => !v)}
              >
                <span className={`h-2.5 w-2.5 rounded-full ${status.cls}`} aria-hidden />
                {status.text}
                <span aria-hidden>▾</span>
              </button>
              {backendOpen && (
                <div className="card absolute right-0 top-9 z-40 w-[min(26rem,calc(100vw-2rem))] p-4 shadow-lg">
                  <BackendConnect compact />
                </div>
              )}
            </div>
          </div>
        </header>

        {open && (
          <div className="fixed inset-0 z-30 lg:hidden" role="dialog" aria-modal="true" aria-label="Navigation">
            <div className="absolute inset-0 bg-slate-900/50" onClick={() => setOpen(false)} />
            <div className="absolute inset-y-0 left-0 flex w-72 flex-col bg-brand-900 px-4 py-5">
              <div className="flex items-center justify-between">
                <Brand />
                <button className="text-slate-200" aria-label="Close navigation" onClick={() => setOpen(false)}>✕</button>
              </div>
              <div className="mt-8 flex-1">{nav}</div>
              <SafetyNote />
            </div>
          </div>
        )}

        <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 sm:px-6 lg:px-8">
          <Outlet />
        </main>
        <footer className="border-t border-slate-200 bg-white px-6 py-3 text-xs text-ink-faint">
          MOSAIC v{health.data?.version ?? "1.0.0"} · Multimodal Observation System for Anomaly Identification and Clinical Trends · academic research prototype
        </footer>
      </div>
    </div>
  );
}

function Brand() {
  return (
    <div className="flex items-center gap-3">
      <img src="/favicon.svg" alt="" className="h-9 w-9" />
      <div>
        <div className="text-lg font-semibold tracking-wide text-white">MOSAIC</div>
        <div className="text-[11px] leading-tight text-brand-200">Wearable anomaly research</div>
      </div>
    </div>
  );
}

function SafetyNote() {
  return (
    <p className="mt-6 rounded-lg bg-brand-800/60 p-3 text-[11px] leading-relaxed text-brand-100">
      Offline research analysis of recorded data. Scores are not probabilities and do not indicate any medical condition.
    </p>
  );
}
