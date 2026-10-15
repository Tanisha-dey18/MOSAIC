"""Train and register models from the command line (same code path as POST /api/models/train).

    python scripts/train.py                       # the default set of methods
    python scripts/train.py --methods iforest_personal,arousal_z_rolling
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

from app.database.models import ModelVersion, TrainingRun  # noqa: E402
from app.database.session import session_factory  # noqa: E402
from app.services import datasets as dsvc  # noqa: E402
from app.services import training as tsvc  # noqa: E402

DEFAULT = ["hr_rule", "robust_z_global", "robust_z_personal", "robust_z_rolling", "iforest_global", "iforest_personal",
           "ocsvm_global", "ocsvm_personal", "autoencoder_global", "autoencoder_personal", "arousal_z_rolling"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", default=",".join(DEFAULT))
    ap.add_argument("--threshold-quantile", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    failed = 0
    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        db = session_factory()()
        ds = dsvc.active_dataset(db)
        run = tsvc.create_run(db, ds, method, None, None, args.threshold_quantile, 5, args.seed)
        run_id = run.id
        db.close()
        tsvc.execute_run(run_id)
        db = session_factory()()
        run = db.get(TrainingRun, run_id)
        mv = run.model_version
        if run.status == "completed" and mv:
            m = (mv.evaluations[0].metrics or {}) if mv.evaluations else {}
            print(f"{method:22s} model #{mv.id} v{mv.version}  ROC-AUC={m.get('roc_auc')}  "
                  f"median participant AUC={m.get('participant_auc_median')}  threshold={mv.threshold:.4g}")
        else:
            failed += 1
            print(f"{method:22s} FAILED: {run.error}")
        db.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
