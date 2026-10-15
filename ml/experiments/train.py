"""Train one method: cross-validated evaluation + a final model on all eligible participants.

Used by the API's training jobs and by `python -m ml.experiments.train`.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml.config import ExperimentConfig
from ml.evaluation.cross_validation import eligibility, run_cv
from ml.evaluation.metrics import curves, per_participant_auc, summarize
from ml.features.extract import FEATURE_NAMES, features_for_modalities
from ml.models.detectors import Detector, parse_method
from ml.pipeline import load_features


def select_training_frame(df: pd.DataFrame, protocol: str = "STRESS") -> pd.DataFrame:
    """Use the labelled STRESS protocol when present (PhysioNet), otherwise all non-excluded sessions."""
    df = df[~df["excluded"]]
    if (df["protocol"] == protocol).any():
        return df[df["protocol"] == protocol]
    return df


def train_and_evaluate(df: pd.DataFrame, method: str, cfg: ExperimentConfig, features: list[str] | None = None,
                       params: dict | None = None, artifact_dir: str | Path | None = None,
                       extra_metadata: dict | None = None, progress=None) -> dict:
    scorer, norm = parse_method(method)
    features = features or FEATURE_NAMES
    labelled = df[df["is_eval"]]
    has_labels = (labelled["label"] == 1).any() and (labelled["label"] == 0).any()
    result: dict = {"method": method, "features": features, "config": cfg.to_dict(),
                    "started_at": datetime.now(timezone.utc).isoformat()}
    eligible, ineligible = eligibility(df, features, cfg.features.min_calibration_windows, require_labels=has_labels)
    result["eligible_participants"], result["ineligible_participants"] = eligible, ineligible
    if len(eligible) < 2:
        raise ValueError(f"need >= 2 eligible participants, found {len(eligible)}: {ineligible}")

    if has_labels and len(eligible) >= cfg.n_folds:
        cv = run_cv(df, [method], features, cfg, params={scorer: params or {}}, eligible=eligible,
                    progress=(lambda f, m: progress(f"fold {f + 1}/{cfg.n_folds}")) if progress else None)
        pred = cv.predictions
        summary = summarize(pred, f"score__{method}", f"thr__{method}", cfg.features.step_s)
        result["evaluation"] = {
            "kind": "participant-grouped cross-validation (out-of-fold, protocol-phase labels)",
            "metrics": summary, "curves": curves(pred, f"score__{method}"),
            "per_participant_auc": per_participant_auc(pred, f"score__{method}").round(4).dropna().to_dict(),
            "timing": cv.timing[method], "folds": cv.folds,
        }
    else:
        result["evaluation"] = {"kind": "none", "reason": "no stressor/rest labels or too few participants; "
                                "only unsupervised score statistics are available"}

    # Final model on all eligible participants.
    final = df[df["participant_id"].isin(eligible)]
    det = Detector(scorer, norm, features, cfg.features, params, seed=cfg.seed)
    if scorer == "autoencoder":
        det.params.setdefault("epochs", cfg.autoencoder_epochs)
        rng = np.random.default_rng(cfg.seed)
        val = set(rng.choice(eligible, max(1, len(eligible) // 5), replace=False))
        det.fit(final[~final["participant_id"].isin(val)], final[final["participant_id"].isin(val)])
    else:
        if scorer == "ocsvm":
            det.params.setdefault("max_train", cfg.ocsvm_max_train)
        det.fit(final)
    scores = det.score(final)
    rest = (final["is_eval"] & (final["label"] == 0)).to_numpy()
    normal_ref = scores[rest] if rest.any() else scores[final["is_calibration"].to_numpy()]
    if scorer != "hr_rule":
        det.set_threshold_from(normal_ref, cfg.threshold_quantile)
    finite = scores[np.isfinite(scores)]
    result["score_distribution"] = {
        "quantiles": {str(q): float(np.quantile(finite, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)} if finite.size else {},
        "fraction_above_threshold": float(np.mean(finite > det.threshold)) if finite.size else None,
    }
    result["threshold"] = det.threshold
    result["train_info"] = det.train_info
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    if artifact_dir is not None:
        det.save(artifact_dir, extra_metadata={**(extra_metadata or {}), "evaluation_summary":
                                               result["evaluation"].get("metrics")})
    result["detector"] = det
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Train and evaluate one MOSAIC method")
    ap.add_argument("--features-dir", default="storage/processed/physionet_wearable_1.0.1")
    ap.add_argument("--method", default="robust_z_personal")
    ap.add_argument("--modalities", default=None, help="comma-separated subset, e.g. EDA,HR")
    ap.add_argument("--out", default="artifacts/manual")
    ap.add_argument("--config", default=None, help="JSON ExperimentConfig file")
    args = ap.parse_args()
    cfg = ExperimentConfig.load(args.config) if args.config else ExperimentConfig()
    df = select_training_frame(load_features(args.features_dir))
    feats = features_for_modalities(args.modalities.split(",")) if args.modalities else FEATURE_NAMES
    out = Path(args.out) / args.method
    res = train_and_evaluate(df, args.method, cfg, feats, artifact_dir=out, progress=print)
    res.pop("detector")
    (out / "training_result.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    m = res["evaluation"].get("metrics", {})
    print(json.dumps({k: m.get(k) for k in ("roc_auc", "pr_auc", "participant_auc_median", "f1", "false_alarms_per_hour")}, indent=1))
    print(f"artifact written to {out}")


if __name__ == "__main__":
    main()
