"""Download and verify the PhysioNet wearable dataset (see ml/data_ingestion/download.py).

    python scripts/download_dataset.py --out datasets/raw/wearable-device-dataset-1.0.1
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.data_ingestion.download import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
