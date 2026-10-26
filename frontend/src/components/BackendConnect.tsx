import { useState } from "react";
import { getApiBase, normaliseBackendUrl, setApiBase, shareLink } from "../lib/backend";

/** Form to point this site at a MOSAIC backend (remembered in this browser). */
export default function BackendConnect({ compact = false }: { compact?: boolean }) {
  const current = getApiBase();
  const [value, setValue] = useState(current ?? "");
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const url = normaliseBackendUrl(value);
    if (!url) {
      setError("Enter a full https:// address (http:// is allowed only for localhost).");
      return;
    }
    setApiBase(url);
  };

  const copy = async () => {
    const link = shareLink();
    if (!link) return;
    try {
      await navigator.clipboard.writeText(link);
      setCopied(true);
    } catch {
      window.prompt("Copy this link:", link);
    }
  };

  return (
    <form onSubmit={submit} className={compact ? "space-y-2" : "space-y-3"}>
      {!compact && (
        <p className="text-sm text-ink-soft">
          MOSAIC's website and its Python backend run separately. Enter the address of a running MOSAIC backend, for
          example <code>https://name.trycloudflare.com</code> or <code>https://user-mosaic-api.hf.space</code>. It is
          remembered in this browser.
        </p>
      )}
      <label className="label" htmlFor={compact ? "backend-url-compact" : "backend-url"}>Backend address</label>
      <div className="flex flex-col gap-2 sm:flex-row">
        <input
          id={compact ? "backend-url-compact" : "backend-url"}
          className="input"
          type="url"
          inputMode="url"
          placeholder="https://…"
          value={value}
          onChange={(e) => { setValue(e.target.value); setError(null); }}
          required
        />
        <button className="btn-primary shrink-0">Connect</button>
      </div>
      {error && <p className="text-xs text-rose-700" role="alert">{error}</p>}
      {current && (
        <div className="flex flex-wrap items-center gap-3 text-xs text-ink-soft">
          <span>Current: <code>{current}</code></span>
          <button type="button" className="font-medium text-brand-700 hover:underline" onClick={copy}>
            {copied ? "Link copied" : "Copy shareable link"}
          </button>
          <button type="button" className="font-medium text-ink-soft hover:underline" onClick={() => setApiBase(null)}>
            Forget
          </button>
        </div>
      )}
    </form>
  );
}
