"""Adapter for the PhysioNet "Wearable Device Dataset from Induced Stress and
Structured Exercise Sessions", version 1.0.1 (Hongn et al., 2025).

DOI: 10.13026/he0v-tf17   License: Open Data Commons Attribution v1.0

Directory layout (verified against the downloaded release):

    <root>/Wearable_Dataset/{STRESS,AEROBIC,ANAEROBIC}/<subject>/{ACC,BVP,EDA,HR,IBI,TEMP,tags}.csv
    <root>/Stress_Level_v1.csv, Stress_Level_v2.csv, subject-info.csv, data_constraints.txt

Phase labelling (STRESS protocol only)
--------------------------------------
The dataset notebook (Wearable_Dataset.ipynb) shades the stressor tasks using
the button tags. With T0..Tn being the tags in tags.csv:

  * version 1 (S01-S18, 13 tags): Stroop [T2,T3], TMCT [T4,T5],
    Real Opinion [T6,T7], Opposite Opinion [T8,T9], Subtract [T10,T11]
  * version 2 (f01-f18, 9 tags):  TMCT [T1,T2], Real Opinion [T3,T4],
    Opposite Opinion [T5,T6], Subtract [T7,T8]

The rest phases are the intervals between stressors that the
Stress_Level_v*.csv files name "First Rest" / "Second Rest" (V1: [T3,T4],
[T5,T6]; V2: [T2,T3], [T6,T7]). The "Baseline" phase [T0,T1] is *inferred*
from the protocol order (it is the first segment in the Stress_Level files and
lasts ~3 min in V1, as the protocol describes); the notebook does not shade it.

These are protocol-phase labels (stressor task vs rest), used as a *proxy*
for "physiological deviation from a resting state". They are not clinical
anomaly labels. Sessions whose tag count does not match the protocol are kept
for exploration but receive no labels.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

from ml.data_ingestion.empatica import SIGNAL_FILES, IngestionError, read_signal, read_tags
from ml.data_ingestion.session import LABEL_REST, LABEL_STRESSOR, Phase, Session

log = logging.getLogger(__name__)

DATASET_NAME = "PhysioNet Wearable Device Dataset from Induced Stress and Structured Exercise Sessions"
DATASET_SLUG = "wearable-device-dataset"
DATASET_VERSION = "1.0.1"
DATASET_DOI = "10.13026/he0v-tf17"
DATASET_LICENSE = "Open Data Commons Attribution License v1.0"
DATASET_URL = "https://physionet.org/content/wearable-device-dataset/1.0.1/"
PROTOCOLS = ("STRESS", "AEROBIC", "ANAEROBIC")

# Known issues from data_constraints.txt. Excluded sessions are still ingested
# (and shown in the Data Explorer) but are not used for model training/evaluation.
EXCLUDED_SESSIONS: dict[tuple[str, str], str] = {
    ("STRESS", "S02"): "data_constraints.txt: duplicated signal segments from E4 Connect downloads",
    ("STRESS", "f14_a"): "data_constraints.txt: recording split (f14_a = baseline only, no tags)",
    ("STRESS", "f14_b"): "data_constraints.txt: recording split (baseline is in f14_a; personal baseline not available causally)",
    ("AEROBIC", "S11_a"): "data_constraints.txt: recording split across S11_a/S11_b",
    ("AEROBIC", "S11_b"): "data_constraints.txt: recording split across S11_a/S11_b",
    ("ANAEROBIC", "S16_a"): "data_constraints.txt: recording split across S16_a/S16_b",
    ("ANAEROBIC", "S16_b"): "data_constraints.txt: recording split across S16_a/S16_b",
}
# Modalities documented as invalid for a session.
INVALID_MODALITIES: dict[tuple[str, str], dict[str, str]] = {
    ("STRESS", "f07"): {
        m: "data_constraints.txt: protective dock not removed; PPG/temperature sensors blocked"
        for m in ("BVP", "HR", "TEMP")
    },
}

STRESS_V1_PHASES = [
    # (name, start tag index, end tag index, label)
    ("Baseline", 0, 1, LABEL_REST),
    ("Transition", 1, 2, None),
    ("Stroop", 2, 3, LABEL_STRESSOR),
    ("First Rest", 3, 4, LABEL_REST),
    ("TMCT", 4, 5, LABEL_STRESSOR),
    ("Second Rest", 5, 6, LABEL_REST),
    ("Real Opinion", 6, 7, LABEL_STRESSOR),
    ("Pause", 7, 8, None),
    ("Opposite Opinion", 8, 9, LABEL_STRESSOR),
    ("Pause", 9, 10, None),
    ("Subtract", 10, 11, LABEL_STRESSOR),
    ("Post", 11, 12, None),
]
STRESS_V2_PHASES = [
    ("Baseline", 0, 1, LABEL_REST),
    ("TMCT", 1, 2, LABEL_STRESSOR),
    ("First Rest", 2, 3, LABEL_REST),
    ("Real Opinion", 3, 4, LABEL_STRESSOR),
    ("Pause", 4, 5, None),
    ("Opposite Opinion", 5, 6, LABEL_STRESSOR),
    ("Second Rest", 6, 7, LABEL_REST),
    ("Subtract", 7, 8, LABEL_STRESSOR),
]
EXPECTED_TAGS = {"V1": 13, "V2": 9}


def resolve_root(path: str | Path) -> Path:
    """Accept either the release root or its Wearable_Dataset sub-folder."""
    path = Path(path)
    if (path / "Wearable_Dataset").is_dir():
        return path
    if path.name == "Wearable_Dataset" and path.is_dir():
        return path.parent
    raise IngestionError(f"{path} does not contain a 'Wearable_Dataset' folder")


def participant_from_folder(folder: str) -> str:
    """'f14_a' -> 'f14'. Identifiers are the dataset's own pseudonyms."""
    return folder.split("_")[0]


def protocol_version(participant_id: str) -> str:
    return "V1" if participant_id.startswith("S") else "V2"


def stress_phases(tags_s: list[float], version: str, session_end_s: float) -> tuple[list[Phase], float | None, str | None]:
    """Build STRESS-protocol phases. Returns (phases, calibration_end_s, problem)."""
    expected = EXPECTED_TAGS[version]
    if len(tags_s) != expected:
        return [], None, f"expected {expected} tags for STRESS {version}, found {len(tags_s)}; phases not labelled"
    spec = STRESS_V1_PHASES if version == "V1" else STRESS_V2_PHASES
    phases = [Phase("Pre-protocol", 0.0, tags_s[0], None)]
    for name, a, b, label in spec:
        phases.append(Phase(name, tags_s[a], tags_s[b], label))
    last = spec[-1][2]
    if last < len(tags_s) - 1 or tags_s[-1] < session_end_s:
        phases.append(Phase("Post", tags_s[-1], session_end_s, None))
    first_stressor = next(p for p in phases if p.label == LABEL_STRESSOR)
    return phases, first_stressor.start_s, None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_session(folder: Path, protocol: str, hash_files: bool = True) -> Session:
    subject = folder.name
    pid = participant_from_folder(subject)
    signals = {}
    notes: list[str] = []
    source_files = []
    for name in SIGNAL_FILES:
        f = folder / f"{name}.csv"
        if not f.exists():
            notes.append(f"{name}.csv missing")
            continue
        try:
            signals[name] = read_signal(f, name)
        except IngestionError as exc:
            notes.append(f"{name}.csv unreadable: {exc}")
            continue
        source_files.append(
            {"path": f"{protocol}/{subject}/{f.name}", "bytes": f.stat().st_size,
             "sha256": sha256_file(f) if hash_files else None}
        )
    if not signals:
        raise IngestionError(f"{protocol}/{subject}: no readable signal files")

    session = Session(
        session_id=f"{protocol}_{subject}",
        participant_id=pid,
        protocol=protocol,
        protocol_version=protocol_version(pid),
        signals=signals,
        notes=notes,
        source_files=source_files,
    )
    tags = read_tags(folder / "tags.csv")
    session.tags_s = [float((t - session.origin).total_seconds()) for t in tags]

    key = (protocol, subject)
    if key in EXCLUDED_SESSIONS:
        session.excluded = True
        session.exclusion_reason = EXCLUDED_SESSIONS[key]
    session.invalid_modalities = dict(INVALID_MODALITIES.get(key, {}))

    if protocol == "STRESS" and not session.excluded:
        phases, cal_end, problem = stress_phases(session.tags_s, session.protocol_version, session.duration_s)
        session.phases, session.calibration_end_s = phases, cal_end
        if problem:
            notes.append(problem)
    elif protocol != "STRESS":
        notes.append("exercise protocol: no phase labels are defined in code by the dataset; unlabelled")
    return session


def discover(root: str | Path, protocols: tuple[str, ...] = PROTOCOLS) -> list[tuple[str, Path]]:
    base = resolve_root(root) / "Wearable_Dataset"
    found = []
    for protocol in protocols:
        pdir = base / protocol
        if not pdir.is_dir():
            continue
        for sub in sorted(p for p in pdir.iterdir() if p.is_dir()):
            found.append((protocol, sub))
    return found


def load_dataset(root: str | Path, protocols: tuple[str, ...] = PROTOCOLS, hash_files: bool = True) -> tuple[list[Session], list[str]]:
    """Load all sessions. Returns (sessions, errors)."""
    sessions, errors = [], []
    for protocol, folder in discover(root, protocols):
        try:
            sessions.append(load_session(folder, protocol, hash_files=hash_files))
        except IngestionError as exc:
            errors.append(str(exc))
            log.warning("skipping session", extra={"session": f"{protocol}/{folder.name}", "error": str(exc)})
    return sessions, errors


def load_stress_levels(root: str | Path) -> pd.DataFrame:
    """Self-reported stress levels (1-10) in long format: participant_id, phase, level."""
    root = resolve_root(root)
    frames = []
    for fname in ("Stress_Level_v1.csv", "Stress_Level_v2.csv"):
        f = root / fname
        if not f.exists():
            continue
        df = pd.read_csv(f, index_col=0)
        df.index = df.index.astype(str).str.strip()
        long = df.reset_index(names="participant_id").melt(id_vars="participant_id", var_name="phase", value_name="level")
        long["phase"] = long["phase"].str.strip()
        frames.append(long.dropna(subset=["level"]))
    if not frames:
        return pd.DataFrame(columns=["participant_id", "phase", "level"])
    return pd.concat(frames, ignore_index=True)


def metadata() -> dict:
    return {
        "name": DATASET_NAME,
        "slug": DATASET_SLUG,
        "version": DATASET_VERSION,
        "doi": DATASET_DOI,
        "license": DATASET_LICENSE,
        "url": DATASET_URL,
        "citation": (
            "Hongn, A., Bosch, F., Prado, L., & Bonomini, P. (2025). Wearable Device Dataset from Induced Stress "
            "and Structured Exercise Sessions (version 1.0.1). PhysioNet. https://doi.org/10.13026/he0v-tf17"
        ),
        "device": "Empatica E4 wristband (non-dominant wrist)",
        "modalities": {"EDA": "4 Hz, uS", "BVP": "64 Hz, a.u.", "HR": "1 Hz, bpm (derived from BVP)",
                       "TEMP": "4 Hz, deg C", "ACC": "32 Hz, 1/64 g, 3 axes", "IBI": "event-based, s"},
    }
