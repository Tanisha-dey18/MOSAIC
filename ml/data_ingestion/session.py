"""Dataset-independent representation of one recording session."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ml.data_ingestion.empatica import Signal

# Window labels used in evaluation.
LABEL_STRESSOR = 1  # window lies fully inside a protocol stressor task
LABEL_REST = 0  # window lies fully inside a protocol rest phase


@dataclass
class Phase:
    """A protocol segment, in seconds relative to the session origin."""

    name: str
    start_s: float
    end_s: float
    label: int | None  # 1 = stressor, 0 = rest, None = not used for evaluation

    def to_dict(self) -> dict:
        return {"name": self.name, "start_s": self.start_s, "end_s": self.end_s, "label": self.label}


@dataclass
class Session:
    """One continuous recording of one participant under one protocol."""

    session_id: str
    participant_id: str  # pseudonymous identifier as published by the dataset
    protocol: str  # e.g. STRESS / AEROBIC / ANAEROBIC / UPLOAD
    signals: dict[str, Signal]
    protocol_version: str | None = None
    origin: pd.Timestamp | None = None  # UTC time of t = 0
    tags_s: list[float] = field(default_factory=list)
    phases: list[Phase] = field(default_factory=list)
    # Time (s) at which the first stressor starts; data before it may be used as personal baseline.
    calibration_end_s: float | None = None
    excluded: bool = False
    exclusion_reason: str | None = None
    invalid_modalities: dict[str, str] = field(default_factory=dict)  # modality -> reason
    notes: list[str] = field(default_factory=list)
    source_files: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.origin is None and self.signals:
            self.origin = min(s.start for s in self.signals.values())

    @property
    def modalities(self) -> list[str]:
        return sorted(self.signals.keys())

    def offset_s(self, name: str) -> float:
        """Seconds between the session origin and the first sample of a signal."""
        return float((self.signals[name].start - self.origin).total_seconds())

    @property
    def duration_s(self) -> float:
        if not self.signals:
            return 0.0
        return max(self.offset_s(n) + s.duration_s for n, s in self.signals.items())

    def segment(self, name: str, start_s: float, end_s: float) -> np.ndarray:
        """Raw samples of a signal between two session times (s)."""
        sig = self.signals[name]
        off = self.offset_s(name)
        i0 = int(np.floor((start_s - off) * sig.fs))
        i1 = int(np.floor((end_s - off) * sig.fs))
        i0, i1 = max(i0, 0), min(i1, sig.n_samples)
        if i1 <= i0:
            return np.empty((0,) + sig.values.shape[1:])
        return sig.values[i0:i1]

    def phase_at(self, start_s: float, end_s: float) -> Phase | None:
        """Return the phase that fully contains [start_s, end_s), if any."""
        for ph in self.phases:
            if ph.start_s <= start_s and end_s <= ph.end_s:
                return ph
        return None
