"""Anomaly detectors: a scorer applied to normalised features.

Method name = "<scorer>_<normalisation>", e.g. ``iforest_personal``.

Scorers (all produce "higher = more unusual"; none is a probability):
  * robust_z    RMS of robust z-scores: s(z) = sqrt(mean_j z_j^2)  (two-sided: any deviation)
  * arousal_z   one-sided RMS over features with a physiological direction of sympathetic
                arousal: s(z) = sqrt(mean_j max(d_j z_j, 0)^2), d_j in {+1,-1} (see AROUSAL_DIRECTION).
                Added as a *secondary, post-hoc* analysis after the primary two-sided results
                showed that relaxation and drift below/above baseline are also "deviations".
  * iforest     Isolation Forest, s = -score_samples (mean path-length based)
  * ocsvm       One-Class SVM (RBF), s = -decision_function
  * autoencoder reconstruction MSE of a PyTorch autoencoder
  * hr_rule     conventional fixed rule: mean heart rate, threshold 100 bpm
                (the usual adult resting tachycardia cut-off) - a reference only

Training uses the *calibration windows of training participants* (pre-stressor,
assumed "normal"). The alert threshold is the q-quantile of scores on the
training participants' rest windows (target false-positive rate 1 - q).
"""
from __future__ import annotations

import hashlib
import io
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from ml import __version__ as ML_VERSION
from ml.config import FeatureConfig
from ml.models.autoencoder import AutoencoderScorer
from ml.models.normalization import (
    ParticipantBaseline, ReferenceStats, build_baselines, complete_calibration_rows, fit_reference, normalize,
)

SCORERS = ("robust_z", "arousal_z", "iforest", "ocsvm", "autoencoder", "hr_rule")
ROLLING_SCORERS = ("robust_z", "arousal_z")
# Direction in which each feature is expected to move under sympathetic arousal
# (EDA level and SCR frequency rise; heart rate rises; peripheral vasoconstriction lowers
# pulse amplitude and skin temperature). Features without a clear direction are omitted.
AROUSAL_DIRECTION = {"eda_mean": 1, "eda_std": 1, "eda_slope": 1, "eda_scr_rate": 1, "hr_mean": 1, "bvp_rate": 1,
                     "bvp_amp": -1, "temp_mean": -1, "temp_slope": -1}
NORMALIZATIONS = ("global", "personal", "rolling", "none")
Z_CLIP = 20.0  # |z| is clipped to limit the influence of single extreme artefacts
HR_RULE_BPM = 100.0
ARTIFACT_FORMAT = 1

DESCRIPTIONS = {
    "robust_z": "Robust statistical baseline: RMS of median/MAD z-scores across features.",
    "arousal_z": "One-sided robust z-score: only deviations in the direction of sympathetic arousal count.",
    "iforest": "Isolation Forest: anomalies are isolated by fewer random axis-aligned splits.",
    "ocsvm": "One-Class SVM (RBF kernel): learns a boundary around normal windows.",
    "autoencoder": "PyTorch autoencoder: anomaly score is the feature reconstruction error.",
    "hr_rule": "Conventional fixed threshold: mean heart rate > 100 bpm.",
}
LIMITATIONS = {
    "robust_z": "Assumes features are roughly unimodal; treats features independently (no interactions).",
    "arousal_z": "Encodes prior physiological assumptions; post-hoc secondary analysis; ignores decreases in arousal.",
    "iforest": "Axis-aligned splits; scores are relative, not calibrated probabilities; contributions are occlusion-based.",
    "ocsvm": "Scales poorly (O(n^2)); trained on a random subsample; sensitive to nu/gamma.",
    "autoencoder": "Small training set (~30 windows per participant); may under- or over-fit; scores are not probabilities.",
    "hr_rule": "Ignores all other modalities and individual differences; 100 bpm is a resting cut-off, not a stress marker.",
}


class ModelNotReadyError(RuntimeError):
    pass


class ArtifactError(RuntimeError):
    pass


def method_name(scorer: str, normalization: str) -> str:
    return "hr_rule" if scorer == "hr_rule" else f"{scorer}_{normalization}"


def parse_method(name: str) -> tuple[str, str]:
    if name == "hr_rule":
        return "hr_rule", "none"
    for scorer in SCORERS:
        if name.startswith(scorer + "_"):
            norm = name[len(scorer) + 1:]
            if norm in NORMALIZATIONS:
                return scorer, norm
    raise ValueError(f"unknown method '{name}'")


class Detector:
    def __init__(self, scorer: str, normalization: str, features: list[str], feature_cfg: FeatureConfig,
                 params: dict | None = None, seed: int = 42):
        if scorer not in SCORERS:
            raise ValueError(f"unknown scorer {scorer}")
        if normalization not in NORMALIZATIONS:
            raise ValueError(f"unknown normalisation {normalization}")
        if scorer == "hr_rule":
            normalization, features = "none", ["hr_mean"]
        if normalization == "rolling" and scorer not in ROLLING_SCORERS:
            raise ValueError("rolling normalisation is only implemented for the robust_z/arousal_z scorers")
        if scorer == "arousal_z" and not any(f in AROUSAL_DIRECTION for f in features):
            raise ValueError("arousal_z needs at least one feature with a defined arousal direction")
        self.scorer, self.normalization = scorer, normalization
        self.features = list(features)
        self.feature_cfg = feature_cfg
        self.params = dict(params or {})
        self.seed = seed
        self.ref: ReferenceStats | None = None
        self.model = None
        self.threshold: float | None = HR_RULE_BPM if scorer == "hr_rule" else None
        self.threshold_quantile: float | None = None
        self.train_info: dict = {}

    @property
    def name(self) -> str:
        return method_name(self.scorer, self.normalization)

    @property
    def is_fitted(self) -> bool:
        return self.scorer == "hr_rule" or self.ref is not None

    # ------------------------------------------------------------------
    def _z(self, df: pd.DataFrame, baselines: dict[str, ParticipantBaseline] | None) -> np.ndarray:
        if self.normalization == "personal" and baselines is None:
            baselines = build_baselines(df, self.ref, self.feature_cfg)
        Z = normalize(df, self.normalization, self.ref, self.feature_cfg, baselines)
        return np.clip(Z, -Z_CLIP, Z_CLIP)

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> "Detector":
        """Fit on training participants. `train` must contain their calibration windows."""
        t0 = time.perf_counter()
        if self.scorer == "hr_rule":
            self.train_info = {"n_train_windows": 0, "fit_seconds": 0.0}
            return self
        self.ref = fit_reference(train, self.features, self.feature_cfg)
        cal = complete_calibration_rows(train, self.features)
        Z = self._z(cal, None) if self.normalization != "rolling" else np.zeros((0, len(self.features)))
        Z = Z[~np.isnan(Z).all(axis=1)] if len(Z) else Z
        Z = np.nan_to_num(Z)
        rng = np.random.default_rng(self.seed)
        if self.scorer == "iforest":
            self.model = IsolationForest(
                n_estimators=self.params.get("n_estimators", 200),
                max_samples=self.params.get("max_samples", "auto"),
                contamination=self.params.get("contamination", "auto"),
                random_state=self.seed,
            ).fit(Z)
        elif self.scorer == "ocsvm":
            max_n = int(self.params.get("max_train", 3000))
            if len(Z) > max_n:
                Z = Z[rng.choice(len(Z), max_n, replace=False)]
            self.model = OneClassSVM(kernel="rbf", nu=self.params.get("nu", 0.05),
                                     gamma=self.params.get("gamma", "scale")).fit(Z)
        elif self.scorer == "autoencoder":
            Zv = None
            if val is not None and len(val):
                vcal = complete_calibration_rows(val, self.features)
                if len(vcal):
                    Zv = np.nan_to_num(self._z(vcal, None))
                    Zv = Zv[~np.isnan(Zv).all(axis=1)]
            ae_params = {k: v for k, v in self.params.items()
                         if k in ("hidden", "bottleneck", "dropout", "lr", "weight_decay", "epochs", "batch_size", "patience")}
            self.model = AutoencoderScorer(seed=self.seed, **ae_params).fit(Z, Zv)
        self.train_info = {"n_train_windows": int(len(Z)), "n_train_participants": int(train["participant_id"].nunique()),
                           "fit_seconds": round(time.perf_counter() - t0, 4)}
        if self.scorer == "autoencoder" and self.model.history:
            self.train_info["history"] = self.model.history.to_dict()
        return self

    def score(self, df: pd.DataFrame, baselines: dict[str, ParticipantBaseline] | None = None) -> np.ndarray:
        """Anomaly score per row; NaN where the row cannot be scored (e.g. no baseline, no features)."""
        if not self.is_fitted:
            raise ModelNotReadyError(f"{self.name} is not trained")
        if self.scorer == "hr_rule":
            return df["hr_mean"].to_numpy(float)
        Z = self._z(df, baselines)
        unscorable = np.isnan(Z).all(axis=1)
        Z0 = np.nan_to_num(Z)
        if self.scorer in ("robust_z", "arousal_z"):
            Zs = self._directed(Z) if self.scorer == "arousal_z" else Z
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                s = np.sqrt(np.nanmean(Zs**2, axis=1))
            unscorable |= np.isnan(Zs).all(axis=1)
        elif self.scorer == "iforest":
            s = -self.model.score_samples(Z0)
        elif self.scorer == "ocsvm":
            s = -self.model.decision_function(Z0)
        else:
            s = self.model.score(Z0)
        s = np.asarray(s, dtype=float)
        s[unscorable] = np.nan
        return s

    def _directed(self, Z: np.ndarray) -> np.ndarray:
        """Keep only arousal-direction deviations; features without a direction become NaN."""
        d = np.array([AROUSAL_DIRECTION.get(f, np.nan) for f in self.features])
        out = np.maximum(Z * d, 0.0)
        out[:, np.isnan(d)] = np.nan
        out[np.isnan(Z)] = np.nan
        return out

    def contributions(self, df: pd.DataFrame, baselines: dict[str, ParticipantBaseline] | None = None) -> np.ndarray | None:
        """Per-feature contribution to each row's score (n, d), or None if unsupported.

        robust_z    -> z_j^2 / d (sums to the squared score)
        autoencoder -> squared reconstruction error of feature j / d (sums to the score)
        iforest/ocsvm -> occlusion: score(z) - score(z with z_j set to the baseline value 0)
        Contributions describe the model's computation, not physiological causes.
        """
        if self.scorer == "hr_rule":
            return None
        Z = self._z(df, baselines)
        Z0 = np.nan_to_num(Z)
        if self.scorer == "robust_z":
            out = Z0**2 / Z.shape[1]
        elif self.scorer == "arousal_z":
            Zd = self._directed(Z)
            n_dir = max(int(np.isfinite(Zd).any(axis=0).sum()), 1)
            out = np.nan_to_num(Zd) ** 2 / n_dir
        elif self.scorer == "autoencoder":
            out = self.model.contributions(Z0)
        else:
            base = self.score(df, baselines)
            out = np.zeros_like(Z0)
            for j in range(Z0.shape[1]):
                Zj = Z0.copy()
                Zj[:, j] = 0.0
                sj = -self.model.score_samples(Zj) if self.scorer == "iforest" else -self.model.decision_function(Zj)
                out[:, j] = base - sj
        out[np.isnan(Z).all(axis=1)] = np.nan
        return out

    def set_threshold_from(self, normal_scores: np.ndarray, quantile: float) -> float:
        if self.scorer == "hr_rule":
            return self.threshold
        finite = normal_scores[np.isfinite(normal_scores)]
        if finite.size == 0:
            raise ValueError("no finite scores to calibrate the threshold")
        self.threshold = float(np.quantile(finite, quantile))
        self.threshold_quantile = quantile
        return self.threshold

    # ------------------------------------------------------------------
    def metadata(self) -> dict:
        return {
            "artifact_format": ARTIFACT_FORMAT, "ml_version": ML_VERSION,
            "method": self.name, "scorer": self.scorer, "normalization": self.normalization,
            "features": self.features, "feature_hash": feature_hash(self.features),
            "feature_config": self.feature_cfg.to_dict(), "params": self.params, "seed": self.seed,
            "threshold": self.threshold, "threshold_quantile": self.threshold_quantile,
            "reference": self.ref.to_dict() if self.ref else None, "train_info": self.train_info,
            "description": DESCRIPTIONS[self.scorer], "limitations": LIMITATIONS[self.scorer],
        }

    def save(self, directory: str | Path, extra_metadata: dict | None = None) -> Path:
        """Persist the detector. Writes model.bin (if any) and metadata.json with checksums."""
        if not self.is_fitted:
            raise ModelNotReadyError("cannot save an untrained model")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        meta = self.metadata()
        meta["saved_at"] = datetime.now(timezone.utc).isoformat()
        meta.update(extra_metadata or {})
        blob = None
        if self.scorer in ("iforest", "ocsvm"):
            buf = io.BytesIO()
            joblib.dump(self.model, buf)
            blob = buf.getvalue()
        elif self.scorer == "autoencoder":
            blob = self.model.state_bytes()
        if blob is not None:
            (directory / "model.bin").write_bytes(blob)
            meta["model_sha256"] = hashlib.sha256(blob).hexdigest()
        (directory / "metadata.json").write_text(json.dumps(meta, indent=2, default=_json_default), encoding="utf-8")
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> "Detector":
        """Load and validate an artifact. Never substitutes an untrained model."""
        directory = Path(directory)
        meta_path = directory / "metadata.json"
        if not meta_path.exists():
            raise ArtifactError(f"model artifact missing: {meta_path}")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ArtifactError(f"corrupted metadata.json: {exc}") from exc
        for key in ("artifact_format", "scorer", "normalization", "features", "feature_hash", "feature_config"):
            if key not in meta:
                raise ArtifactError(f"metadata.json missing key '{key}'")
        if meta["artifact_format"] != ARTIFACT_FORMAT:
            raise ArtifactError(f"unsupported artifact format {meta['artifact_format']}")
        if feature_hash(meta["features"]) != meta["feature_hash"]:
            raise ArtifactError("feature schema hash mismatch")
        det = cls(meta["scorer"], meta["normalization"], meta["features"],
                  FeatureConfig.from_dict(meta["feature_config"]), meta.get("params"), meta.get("seed", 42))
        det.threshold = meta.get("threshold")
        det.threshold_quantile = meta.get("threshold_quantile")
        det.train_info = meta.get("train_info", {})
        if meta["scorer"] != "hr_rule":
            if not meta.get("reference"):
                raise ArtifactError("metadata.json has no reference statistics")
            det.ref = ReferenceStats.from_dict(meta["reference"])
        if meta["scorer"] in ("iforest", "ocsvm", "autoencoder"):
            bin_path = directory / "model.bin"
            if not bin_path.exists():
                raise ArtifactError(f"model weights missing: {bin_path}")
            blob = bin_path.read_bytes()
            if hashlib.sha256(blob).hexdigest() != meta.get("model_sha256"):
                raise ArtifactError("model.bin checksum mismatch (corrupted or modified artifact)")
            if meta["scorer"] == "autoencoder":
                det.model = AutoencoderScorer.from_bytes(blob)
            else:
                det.model = joblib.load(io.BytesIO(blob))
        return det


def feature_hash(features: list[str]) -> str:
    return hashlib.sha256("|".join(features).encode()).hexdigest()[:16]


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
