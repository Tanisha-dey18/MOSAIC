"""Central, serialisable configuration for the MOSAIC pipeline."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

MODALITIES = ("EDA", "HR", "TEMP", "BVP", "ACC")


@dataclass
class FeatureConfig:
    window_s: float = 30.0  # window length; 30 s lets the 30-s stressor tasks contain whole windows
    step_s: float = 10.0  # window hop (overlapping windows; dependence handled by participant-level bootstrap)
    min_valid_fraction: float = 0.8  # a modality's features are NaN when <80% of its samples are valid
    rest_guard_s: float = 60.0  # ignore the first 60 s of each rest phase (physiological recovery)
    calibration_s: float = 300.0  # personal baseline = the last 5 min before the first stressor
    min_calibration_windows: int = 10  # below this a participant is "insufficient history"
    modalities: tuple[str, ...] = MODALITIES

    def to_dict(self) -> dict:
        d = asdict(self)
        d["modalities"] = list(self.modalities)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureConfig":
        d = dict(d)
        if "modalities" in d:
            d["modalities"] = tuple(d["modalities"])
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class ExperimentConfig:
    seed: int = 42
    n_folds: int = 5  # participant-grouped cross-validation folds
    threshold_quantile: float = 0.95  # threshold = q-quantile of scores on training participants' rest windows
    bootstrap_iterations: int = 1000
    ocsvm_max_train: int = 3000  # OCSVM training is O(n^2)-O(n^3); subsample for tractability
    autoencoder_epochs: int = 200
    features: FeatureConfig = field(default_factory=FeatureConfig)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["features"] = self.features.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        d = dict(d)
        feats = FeatureConfig.from_dict(d.pop("features", {}))
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(features=feats, **known)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
