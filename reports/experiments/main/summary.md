# MOSAIC experiment results

Run: `exp_20261015T064235Z`  |  Generated: 2026-10-15T06:42:36.152796+00:00  |  Dataset: PhysioNet Wearable Device Dataset from Induced Stress and Structured Exercise Sessions v1.0.1

Labels are protocol phases (stressor task = 1, rest = 0) used as a proxy for deviation from a resting state; they are not clinical anomaly annotations. Scores are not probabilities.

Eligible participants: 33. Ineligible: f07 (insufficient complete calibration windows (0 < 10))


## E1 Primary comparison (pre-declared)

| Method | ROC-AUC [95% CI] | PR-AUC [95% CI] | Median participant AUC | Precision | Recall | F1 | False alarms / h | Event sensitivity |
|---|---|---|---|---|---|---|---|---|
| hr_rule | 0.680 [0.613, 0.765] | 0.341 [0.264, 0.561] | 0.930 | 0.341 | 0.076 | 0.124 | 13.410 | 0.163 |
| robust_z_global | 0.587 [0.535, 0.639] | 0.287 [0.242, 0.341] | 0.654 | 0.345 | 0.115 | 0.173 | 20.000 | 0.194 |
| robust_z_personal | 0.420 [0.329, 0.513] | 0.186 [0.145, 0.263] | 0.367 | 0.216 | 0.087 | 0.124 | 29.017 | 0.173 |
| robust_z_rolling | 0.648 [0.569, 0.720] | 0.308 [0.238, 0.396] | 0.635 | 0.333 | 0.100 | 0.154 | 18.266 | 0.306 |
| iforest_global | 0.598 [0.548, 0.654] | 0.279 [0.231, 0.355] | 0.672 | 0.324 | 0.119 | 0.174 | 22.659 | 0.265 |
| iforest_personal | 0.513 [0.417, 0.604] | 0.257 [0.195, 0.351] | 0.439 | 0.388 | 0.141 | 0.206 | 20.231 | 0.265 |
| ocsvm_global | 0.424 [0.368, 0.478] | 0.185 [0.154, 0.237] | 0.363 | 0.230 | 0.090 | 0.129 | 27.515 | 0.184 |
| ocsvm_personal | 0.370 [0.288, 0.457] | 0.159 [0.129, 0.211] | 0.313 | 0.080 | 0.023 | 0.035 | 23.815 | 0.071 |
| autoencoder_global | 0.576 [0.509, 0.642] | 0.228 [0.191, 0.290] | 0.556 | 0.192 | 0.077 | 0.110 | 29.711 | 0.163 |
| autoencoder_personal | 0.357 [0.278, 0.445] | 0.152 [0.126, 0.201] | 0.288 | 0.054 | 0.015 | 0.024 | 24.046 | 0.061 |

Prevalence of stressor windows (PR-AUC chance level): 0.202


## E2 Secondary, post-hoc (arousal-directed)

| Method | ROC-AUC [95% CI] | PR-AUC [95% CI] | Median participant AUC | Precision | Recall | F1 | False alarms / h | Event sensitivity |
|---|---|---|---|---|---|---|---|---|
| arousal_z_global | 0.656 [0.593, 0.724] | 0.343 [0.285, 0.412] | 0.810 | 0.407 | 0.150 | 0.219 | 19.884 | 0.225 |
| arousal_z_personal | 0.583 [0.501, 0.681] | 0.272 [0.215, 0.367] | 0.737 | 0.375 | 0.126 | 0.188 | 19.075 | 0.184 |
| arousal_z_rolling | 0.796 [0.745, 0.846] | 0.457 [0.366, 0.574] | 0.875 | 0.482 | 0.207 | 0.289 | 20.231 | 0.398 |

Prevalence of stressor windows (PR-AUC chance level): 0.202


## Paired comparisons (same windows)

| A | B | Pooled AUC diff (B-A) [95% CI] | Median participant AUC diff | B better in | Wilcoxon p |
|---|---|---|---|---|---|
| robust_z_global | robust_z_personal | -0.167 [-0.243, -0.090] | -0.121 | 9/33 | 0.0005 |
| iforest_global | iforest_personal | -0.085 [-0.167, -0.005] | -0.056 | 16/33 | 0.1045 |
| ocsvm_global | ocsvm_personal | -0.054 [-0.156, +0.058] | -0.028 | 15/33 | 0.4368 |
| autoencoder_global | autoencoder_personal | -0.219 [-0.274, -0.150] | -0.220 | 10/33 | 0.0002 |
| robust_z_global | robust_z_rolling | +0.061 [-0.009, +0.121] | +0.024 | 21/33 | 0.2865 |
| arousal_z_global | arousal_z_personal | -0.073 [-0.163, +0.028] | -0.029 | 14/33 | 0.0798 |
| arousal_z_personal | arousal_z_rolling | +0.213 [+0.110, +0.302] | +0.159 | 22/33 | 0.0031 |
| hr_rule | arousal_z_personal | -0.096 [-0.207, +0.005] | -0.083 | 4/33 | 0.0011 |

No correction for multiple comparisons is applied; with 8 comparisons, a Bonferroni threshold would be p < 0.00625.


## E4 Modality ablation (pooled ROC-AUC / median participant AUC)

| feature_set | arousal_z_global | arousal_z_personal | iforest_global | iforest_personal | robust_z_global | robust_z_personal |
|---|---|---|---|---|---|---|
| ACC only | n/a | n/a | 0.506 | 0.445 | 0.532 | 0.453 |
| BVP only | 0.678 | 0.691 | 0.545 | 0.583 | 0.53 | 0.569 |
| EDA only | 0.644 | 0.66 | 0.578 | 0.532 | 0.584 | 0.532 |
| HR only | 0.669 | 0.73 | 0.622 | 0.601 | 0.621 | 0.611 |
| TEMP only | 0.508 | 0.435 | 0.458 | 0.281 | 0.447 | 0.244 |
| all | 0.656 | 0.583 | 0.598 | 0.513 | 0.586 | 0.42 |
| without ACC | 0.656 | 0.583 | 0.589 | 0.529 | 0.598 | 0.425 |
| without BVP | 0.648 | 0.573 | 0.595 | 0.503 | 0.589 | 0.414 |
| without EDA | 0.652 | 0.591 | 0.567 | 0.479 | 0.523 | 0.357 |
| without HR | 0.652 | 0.558 | 0.587 | 0.464 | 0.58 | 0.395 |
| without TEMP | 0.668 | 0.694 | 0.609 | 0.568 | 0.596 | 0.527 |

Median per-participant AUC:

| feature_set | arousal_z_global | arousal_z_personal | iforest_global | iforest_personal | robust_z_global | robust_z_personal |
|---|---|---|---|---|---|---|
| ACC only | n/a | n/a | 0.474 | 0.516 | 0.522 | 0.494 |
| BVP only | 0.777 | 0.743 | 0.588 | 0.616 | 0.573 | 0.568 |
| EDA only | 0.776 | 0.795 | 0.621 | 0.423 | 0.662 | 0.418 |
| HR only | 0.783 | 0.854 | 0.638 | 0.618 | 0.681 | 0.681 |
| TEMP only | 0.561 | 0.416 | 0.418 | 0.301 | 0.389 | 0.263 |
| all | 0.81 | 0.737 | 0.672 | 0.439 | 0.654 | 0.367 |
| without ACC | 0.81 | 0.737 | 0.657 | 0.516 | 0.657 | 0.383 |
| without BVP | 0.786 | 0.646 | 0.616 | 0.489 | 0.64 | 0.354 |
| without EDA | 0.807 | 0.77 | 0.643 | 0.62 | 0.544 | 0.474 |
| without HR | 0.81 | 0.615 | 0.671 | 0.43 | 0.661 | 0.347 |
| without TEMP | 0.828 | 0.895 | 0.668 | 0.574 | 0.657 | 0.561 |


## E5 Calibration length (personal baselines)

| calibration_s | arousal_z_personal | iforest_personal | robust_z_personal |
|---|---|---|---|
| 60 | 0.605 | 0.475 | 0.4 |
| 120 | 0.49 | 0.419 | 0.309 |
| 180 | 0.54 | 0.501 | 0.343 |
| 240 | 0.579 | 0.537 | 0.41 |
| 300 | 0.583 | 0.513 | 0.42 |


## E6 Sensor drop-out after calibration (pooled ROC-AUC)

| scenario | arousal_z_personal | autoencoder_personal | iforest_global | iforest_personal | ocsvm_personal | robust_z_global | robust_z_personal |
|---|---|---|---|---|---|---|---|
| ACC lost | 0.583 | 0.376 | 0.598 | 0.524 | 0.388 | 0.598 | 0.425 |
| EDA lost | 0.591 | 0.321 | 0.547 | 0.477 | 0.325 | 0.523 | 0.357 |
| PPG lost (HR+BVP) | 0.531 | 0.321 | 0.594 | 0.456 | 0.327 | 0.578 | 0.388 |
| TEMP lost | 0.694 | 0.484 | 0.607 | 0.575 | 0.485 | 0.596 | 0.527 |
| none | 0.583 | 0.358 | 0.598 | 0.513 | 0.37 | 0.586 | 0.42 |


## E7 Self-reported stress agreement (exploratory)

| Method | n | pooled Spearman rho | p | median within-participant rho |
|---|---|---|---|---|
| hr_rule | 164 | 0.309 | 0.0001 | 0.632 |
| robust_z_global | 164 | -0.002 | 0.9849 | 0.316 |
| robust_z_personal | 164 | -0.236 | 0.0023 | 0.000 |
| robust_z_rolling | 164 | 0.002 | 0.9834 | 0.316 |
| iforest_global | 164 | 0.041 | 0.6049 | 0.369 |
| iforest_personal | 164 | -0.176 | 0.0239 | 0.112 |
| ocsvm_global | 164 | -0.032 | 0.6841 | -0.169 |
| ocsvm_personal | 164 | -0.278 | 0.0003 | -0.112 |
| autoencoder_global | 164 | 0.008 | 0.9184 | 0.131 |
| autoencoder_personal | 164 | -0.279 | 0.0003 | -0.211 |
| arousal_z_global | 164 | 0.013 | 0.8672 | 0.447 |
| arousal_z_personal | 164 | -0.117 | 0.1372 | 0.338 |
| arousal_z_rolling | 164 | 0.113 | 0.1481 | 0.618 |


## E9 Computational cost

| Method | mean fit time per fold (s) | inference (ms / window) |
|---|---|---|
| hr_rule | 0.000 | 0.0001 |
| robust_z_global | 0.105 | 0.0012 |
| robust_z_personal | 0.299 | 0.0406 |
| robust_z_rolling | 0.086 | 1.2171 |
| iforest_global | 0.600 | 0.0369 |
| iforest_personal | 0.785 | 0.0756 |
| ocsvm_global | 0.107 | 0.0088 |
| ocsvm_personal | 0.417 | 0.0550 |
| autoencoder_global | 6.446 | 0.0018 |
| autoencoder_personal | 5.429 | 0.0466 |
| arousal_z_global | 0.105 | 0.0035 |
| arousal_z_personal | 0.415 | 0.0402 |
| arousal_z_rolling | 0.113 | 1.2648 |


Environment: Python 3.13.15, scikit-learn 1.9.0, torch 2.13.0+cpu, Windows-11-10.0.26300-SP0
