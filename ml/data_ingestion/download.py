"""Download the PhysioNet Wearable Device Dataset (v1.0.1) and verify checksums.

Source: Hongn A., Bosch F., Prado L., Bonomini P. (2025). "Wearable Device Dataset
from Induced Stress and Structured Exercise Sessions" (version 1.0.1). PhysioNet.
https://doi.org/10.13026/he0v-tf17  -- License: Open Data Commons Attribution v1.0.

Files are fetched from PhysioNet's official open-data S3 bucket (physionet-open),
which mirrors https://physionet.org/content/wearable-device-dataset/1.0.1/.
Every file is verified against the dataset's own SHA256SUMS.txt.

Usage:
    python scripts/download_dataset.py --out datasets/raw/wearable-device-dataset-1.0.1
    python scripts/download_dataset.py --out ... --skip-notebook   # skip the 34 MB notebook
(The backend's start-up bootstrap calls `download()` directly.)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_URL = "https://physionet-open.s3.amazonaws.com/wearable-device-dataset/1.0.1/"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, dest: Path, retries: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as fh:
                while chunk := resp.read(1 << 20):
                    fh.write(chunk)
            tmp.replace(dest)
            return
        except Exception as exc:  # network errors are retried
            last_error = exc
    raise RuntimeError(f"failed to download {url}: {last_error}")


def parse_sums(text: str) -> list[tuple[str, str]]:
    entries = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*").strip()
        # Reject anything that could escape the output directory.
        if name.startswith(("/", "\\")) or ".." in Path(name).parts:
            raise ValueError(f"unsafe path in SHA256SUMS: {name}")
        entries.append((digest.lower(), name))
    return entries


def download(out: str | Path, skip_notebook: bool = False, workers: int = 8, log=print) -> Path:
    """Download (or resume) the release into `out`, verifying every SHA-256. Raises on any failure."""
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    sums_path = out / "SHA256SUMS.txt"
    fetch(BASE_URL + "SHA256SUMS.txt", sums_path)
    entries = parse_sums(sums_path.read_text(encoding="utf-8"))
    if skip_notebook:
        entries = [e for e in entries if not e[1].endswith(".ipynb")]
    todo = [(d, n) for d, n in entries if not ((out / n).exists() and sha256_of(out / n) == d)]
    log(f"{len(entries)} files listed, {len(todo)} to download")

    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, BASE_URL + name.replace("\\", "/"), out / name): (d, name) for d, name in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            digest, name = futures[fut]
            try:
                fut.result()
                if sha256_of(out / name) != digest:
                    failures.append(f"checksum mismatch: {name}")
            except Exception as exc:
                failures.append(str(exc))
            if i % 100 == 0 or i == len(todo):
                log(f"  {i}/{len(todo)} downloaded")
    if failures:
        raise RuntimeError("dataset download failed:\n" + "\n".join(failures[:20]))
    log(f"All files verified against SHA256SUMS.txt in {out}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="datasets/raw/wearable-device-dataset-1.0.1")
    parser.add_argument("--skip-notebook", action="store_true", help="do not download Wearable_Dataset.ipynb")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    try:
        download(args.out, args.skip_notebook, args.workers)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
