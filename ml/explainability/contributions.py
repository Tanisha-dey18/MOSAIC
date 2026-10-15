"""Feature-level explanation of one window's anomaly score.

The model-specific contribution values come from ``Detector.contributions``:

* robust_z / arousal_z: squared (directed) z-score / number of features (exact decomposition of s^2)
* autoencoder: squared reconstruction error per feature / d (exact decomposition of the score)
* iforest / ocsvm: occlusion, i.e. the score drop when the feature is replaced by its reference value

This module turns them into a table that compares each observed feature value
with its reference (personal baseline, rolling window or population). It
describes the model's computation (association), never a physiological cause.
"""
from __future__ import annotations

import math

from ml.features.extract import FEATURE_MODALITY, FEATURE_SPECS

UNITS = {f[0]: f[2] for f in FEATURE_SPECS}
DESCRIPTIONS = {f[0]: f[3] for f in FEATURE_SPECS}

METHOD_NOTES = {
    "robust_z": "Contribution = squared robust z-score of the feature divided by the number of features.",
    "arousal_z": "Contribution = squared one-sided (arousal-direction) z-score divided by the number of directed features.",
    "autoencoder": "Contribution = squared reconstruction error of the feature divided by the number of features.",
    "iforest": "Contribution = score drop when the feature is replaced by its reference value (occlusion).",
    "ocsvm": "Contribution = score drop when the feature is replaced by its reference value (occlusion).",
    "hr_rule": "The fixed rule uses only mean heart rate; no multi-feature explanation is available.",
}

CAVEATS = [
    "Scores are relative measures of deviation from a reference, not probabilities of any condition.",
    "Feature contributions describe how the model computed the score (association), not physiological causes.",
    "Protocol labels (stressor/rest) are experimental phases, not clinical ground truth.",
    "This is research software and must not be used for diagnosis or medical decisions.",
]


def _num(x) -> float | None:
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) or math.isinf(x) else x


def feature_table(features: list[str], values, contributions=None, center=None, scale=None) -> list[dict]:
    """Rows sorted by |contribution| (largest first), with value vs reference and z-score."""
    rows = []
    for j, f in enumerate(features):
        value = _num(values[j])
        c = _num(center[j]) if center is not None else None
        s = _num(scale[j]) if scale is not None else None
        z = (value - c) / s if value is not None and c is not None and s else None
        rows.append({"feature": f, "modality": FEATURE_MODALITY[f], "unit": UNITS[f], "description": DESCRIPTIONS[f],
                     "value": value, "reference_center": c, "reference_scale": s, "z": _num(z),
                     "contribution": _num(contributions[j]) if contributions is not None else None})
    total = sum(abs(r["contribution"]) for r in rows if r["contribution"] is not None) or None
    for r in rows:
        r["contribution_share"] = abs(r["contribution"]) / total if total and r["contribution"] is not None else None
    rows.sort(key=lambda r: -(abs(r["contribution"]) if r["contribution"] is not None else -1))
    return rows


__all__ = ["feature_table", "METHOD_NOTES", "CAVEATS"]
