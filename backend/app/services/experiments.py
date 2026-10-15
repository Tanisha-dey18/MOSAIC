"""Access to experiment runs produced by `python -m ml.experiments.run_experiments`.

Runs live in REPORTS_DIR/experiments/<run_id>/ (results.json, summary.md, figures/).
The API can launch a new run as a separate OS process so the server is never blocked.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import PROJECT_ROOT, get_settings
from app.core.errors import APIError, not_found
from app.core.paths import safe_name

FIGURE_RE = re.compile(r"^[a-z0-9_]+\.png$")


def _runs_dir() -> Path:
    return get_settings().experiments_dir


def list_runs() -> list[dict]:
    root = _runs_dir()
    if not root.exists():
        return []
    out = []
    for d in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True):
        res = d / "results.json"
        status_file = d / "status.json"
        if res.exists():
            try:
                data = json.loads(res.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                out.append({"run_id": d.name, "status": "corrupted"})
                continue
            out.append({"run_id": d.name, "status": "completed", "generated_at": data.get("generated_at"),
                        "dataset": data.get("dataset"), "n_participants": len(data.get("eligibility", {}).get("eligible", [])),
                        "n_eval_windows": data.get("n_eval_windows"), "runtime_seconds": data.get("runtime_seconds")})
        elif status_file.exists():
            st = json.loads(status_file.read_text(encoding="utf-8"))
            log_tail = ""
            log_file = d / "run.log"
            if log_file.exists():
                log_tail = "\n".join(log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-5:])
            out.append({"run_id": d.name, "status": st.get("status", "running"), "started_at": st.get("started_at"),
                        "log_tail": log_tail})
    return out


def get_run(run_id: str) -> dict:
    d = _runs_dir() / safe_name(run_id, "run id")
    res = d / "results.json"
    if not res.exists():
        raise not_found("experiment run", run_id)
    data = json.loads(res.read_text(encoding="utf-8"))
    data["run_id"] = run_id
    return data


def figure_path(run_id: str, name: str) -> Path:
    if not FIGURE_RE.match(name):
        raise APIError(400, "invalid_name", "invalid figure name")
    path = _runs_dir() / safe_name(run_id, "run id") / "figures" / name
    if not path.exists():
        raise not_found("figure", name)
    return path


def launch_run(features_dir: Path, quick: bool) -> dict:
    """Start the experiment suite in a child process; progress goes to run.log."""
    running = [r for r in list_runs() if r.get("status") == "running"]
    if running:
        raise APIError(409, "experiment_running", f"experiment {running[0]['run_id']} is still running")
    run_id = datetime.now(timezone.utc).strftime("exp_%Y%m%dT%H%M%SZ")
    d = _runs_dir() / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "status.json").write_text(json.dumps({"status": "running",
                                               "started_at": datetime.now(timezone.utc).isoformat()}), encoding="utf-8")
    cmd = [sys.executable, "-W", "ignore", "-m", "ml.experiments.run_experiments", "--features-dir", str(features_dir),
           "--out", str(d)]
    if quick:
        cmd.append("--quick")
    log = (d / "run.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=log, stderr=subprocess.STDOUT)  # fixed argv, no shell

    def _finish():
        code = proc.wait()
        log.close()
        status = "completed" if code == 0 and (d / "results.json").exists() else "failed"
        (d / "status.json").write_text(json.dumps({"status": status, "exit_code": code}), encoding="utf-8")

    import threading
    threading.Thread(target=_finish, daemon=True).start()
    return {"run_id": run_id, "status": "running"}
