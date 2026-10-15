"""Ingest a dataset into MOSAIC from the command line (same code path as the API).

    python scripts/ingest.py --path datasets/raw/wearable-device-dataset-1.0.1
    python scripts/ingest.py --csv my_recordings.csv --name "Pilot upload"

Requires the database schema: run `alembic -c backend/alembic.ini upgrade head` first.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

from app.core.config import get_settings  # noqa: E402
from app.database.models import Dataset  # noqa: E402
from app.database.session import session_factory  # noqa: E402
from app.services import datasets as svc  # noqa: E402
from ml.config import FeatureConfig  # noqa: E402
from ml.data_ingestion import physionet_wearable  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--path", help="PhysioNet release folder (must be inside DATASETS_DIR)")
    src.add_argument("--csv", help="generic long-format CSV file")
    ap.add_argument("--name", default="Uploaded dataset")
    ap.add_argument("--protocols", default=None, help="comma-separated subset, e.g. STRESS")
    args = ap.parse_args()

    settings = get_settings()
    settings.ensure_dirs()
    db = session_factory()()
    try:
        if args.path:
            abs_path = Path(args.path).resolve()
            try:
                rel = abs_path.relative_to(settings.datasets_dir)
            except ValueError:
                print(f"error: {abs_path} is not inside DATASETS_DIR ({settings.datasets_dir})", file=sys.stderr)
                return 2
            report = svc.validate_physionet_dir(abs_path)
            if not report["valid"]:
                print("validation failed:", report["errors"], file=sys.stderr)
                return 2
            ds = svc.create_dataset(db, "physionet_wearable", rel.as_posix(), physionet_wearable.DATASET_NAME, FeatureConfig())
            protocols = args.protocols.split(",") if args.protocols else None
            source = str(abs_path)
        else:
            report = svc.validate_csv_file(Path(args.csv))
            if not report["valid"]:
                print("validation failed:", report["errors"], file=sys.stderr)
                return 2
            ds = svc.create_dataset(db, "generic_csv", Path(args.csv).name, args.name, FeatureConfig())
            protocols, source = None, str(Path(args.csv).resolve())
        print(f"dataset #{ds.id}: ingesting (this takes about a minute for the PhysioNet release)...")
    finally:
        db.close()
    svc.run_ingestion(ds.id, source, protocols)
    db = session_factory()()
    ds = db.get(Dataset, ds.id)
    print(f"status={ds.status} sessions={ds.n_sessions} participants={ds.n_participants} windows={ds.n_windows}"
          + (f" error={ds.error}" if ds.error else ""))
    if ds.storage_path:
        print(f"processed features: {settings.storage_dir / ds.storage_path}")
    db.close()
    return 0 if ds.status == "ready" else 1


if __name__ == "__main__":
    sys.exit(main())
