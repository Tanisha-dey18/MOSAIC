// Which MOSAIC backend the frontend talks to, decided at runtime.
//
// Order: ?api=<url> in the page link (remembered) > remembered choice > VITE_API_BASE_URL (build time)
// > http://localhost:8000 during local development. A static deployment (Netlify) can therefore be
// pointed at a backend whose address changes (e.g. a temporary tunnel) without rebuilding the site.

const KEY = "mosaic.apiBase";

/** Normalise and validate a backend URL: https anywhere, http only for localhost. Returns its origin or null. */
export function normaliseBackendUrl(raw: string): string | null {
  let url: URL;
  try {
    url = new URL(raw.trim());
  } catch {
    return null;
  }
  const local = ["localhost", "127.0.0.1"].includes(url.hostname);
  if (url.protocol !== "https:" && !(url.protocol === "http:" && local)) return null;
  return url.origin;
}

function readStored(): string | null {
  try {
    const v = window.localStorage.getItem(KEY);
    return v ? normaliseBackendUrl(v) : null;
  } catch {
    return null; // storage blocked (private mode etc.)
  }
}

function store(value: string | null) {
  try {
    if (value) window.localStorage.setItem(KEY, value);
    else window.localStorage.removeItem(KEY);
  } catch {
    /* ignore: the choice then lasts only for this page load */
  }
}

let current: string | null = null;
let initialised = false;

function init() {
  initialised = true;
  const params = new URLSearchParams(window.location.search);
  const fromLink = params.get("api");
  if (fromLink !== null) {
    const ok = normaliseBackendUrl(fromLink);
    if (ok) store(ok);
    params.delete("api");
    const q = params.toString();
    window.history.replaceState(null, "", `${window.location.pathname}${q ? `?${q}` : ""}${window.location.hash}`);
    if (ok) {
      current = ok;
      return;
    }
  }
  const built = import.meta.env.VITE_API_BASE_URL as string | undefined;
  current = readStored() ?? (built ? normaliseBackendUrl(built) : null) ?? (import.meta.env.DEV ? "http://localhost:8000" : null);
}

/** The backend origin in use, or null when none is configured (deployed site opened without one). */
export function getApiBase(): string | null {
  if (!initialised) init();
  return current;
}

/** Remember a new backend (or forget it with null) and reload so every page refetches. */
export function setApiBase(value: string | null) {
  store(value);
  window.location.reload();
}

/** A shareable link that opens this site already connected to the current backend. */
export function shareLink(): string | null {
  const base = getApiBase();
  return base ? `${window.location.origin}/?api=${encodeURIComponent(base)}` : null;
}
