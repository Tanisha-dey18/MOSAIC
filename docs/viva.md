# Viva preparation: questions and model answers

## Problem and motivation

**1. What problem does MOSAIC address?**
It detects windows of wearable physiological data that deviate from a person's resting state, and it tests whether personal references detect such deviations better than population references and non-personalised ML.

**2. Why not just use thresholds like HR > 100 bpm?**
Resting physiology varies greatly between people. Our results show the HR rule ranks windows very well *within* a person (median AUC 0.93) but is poorly calibrated *across* people (pooled AUC 0.68, recall 7.6%). That is precisely the inter-individual problem.

**3. Is MOSAIC a diagnostic tool?**
No. It is offline research software, and scores are not probabilities. The data comes from healthy volunteers in a laboratory, and the labels are experimental phases, not diagnoses.

## Data

**4. Which dataset, and why?**
The PhysioNet *Wearable Device Dataset from Induced Stress and Structured Exercise Sessions* v1.0.1 (DOI 10.13026/he0v-tf17, ODC-By). It is open access, multimodal (Empatica E4: EDA, BVP, HR, TEMP, ACC, IBI) and has a structured stress protocol with button tags, which gives defensible proxy labels.

**5. How did you obtain labels?**
From the button tags, using the tag spans the dataset notebook shades as stressor tasks (Stroop, TMCT, opinion tasks, subtraction) and the rest phases named in the stress-level files. A window is labelled only if it lies fully inside one phase. The first 60 s of rest are excluded because of recovery.

**6. Are these real anomaly labels?**
No. They mark induced perturbation versus rest, so "anomaly" is operationalised as deviation from rest. We say so everywhere.

**7. What data-quality problems did you handle?**
Missing and non-numeric rows, out-of-range values, flat-lines, constant signals, different sampling rates (HR starts 10 s later), split recordings, a duplicated recording (S02), blocked sensors (f07), duplicate timestamps and gaps in uploads.

**8. Why not store all samples in PostgreSQL?**
There are about 10⁸ samples with no relational queries over them. Parquet window features and 1 Hz summaries are smaller and faster, while the database stores metadata, provenance and results.

## Method

**9. Explain your feature set.**
14 features per 30 s window:

* EDA mean, std, slope and SCR rate;
* HR mean, std and slope;
* TEMP mean and slope;
* BVP amplitude and dominant pulse rate;
* ACC magnitude mean and std;
* the EDA–HR correlation.

They are interpretable, so explanations make sense.

**10. Why 30 s windows with a 10 s hop?**
Several stressor tasks last only about 30 s, so longer windows would never fit inside them. A 10 s hop gives enough windows, and the overlap is handled with a participant-level bootstrap.

**11. What is the median absolute deviation and why use it?**
MAD = median|x − median(x)|. Scaled by 1.4826 it estimates the SD for Gaussian data, and its 50% breakdown point makes it robust to artefacts, unlike mean ± SD.

**12. How is the personal baseline constructed?**
From the participant's complete windows in the last 5 minutes before the first stressor, with at least 10 windows. The centre is the median; the scale is the MAD, shrunk towards the typical within-person scale because only about 9 effective windows exist.

**13. Why shrink the personal scale?**
With few windows, a feature can be almost constant, so the MAD approaches 0 and z-scores explode. Shrinkage, (n_eff·s_p² + k·s_w²)/(n_eff + k), is a principled compromise.

**14. Explain Isolation Forest.**
It builds random trees by choosing a random feature and a random split value. Anomalies are isolated in fewer splits, so they have a shorter average path length. The score is 2^(−E[h(x)]/c(n)).

**15. Explain One-Class SVM.**
It maps the data with an RBF kernel and finds a hyperplane separating the data from the origin with maximal margin. ν upper-bounds the fraction of training outliers and lower-bounds the fraction of support vectors. Points on the wrong side are anomalous.

**16. Explain the autoencoder.**
It compresses 14 features to 4 and reconstructs them, trained on normal windows with MSE loss. Windows unlike the training data reconstruct poorly, so the reconstruction error is the anomaly score. Training uses early stopping on a participant-disjoint validation set.

**17. Why no LSTM/GRU?**
Each participant has only about 5 minutes of baseline, and the phases are a few minutes long. There are too few independent sequences to justify a sequence model, so simpler models are more defensible.

**18. How is the threshold chosen?**
It is the 95th percentile of scores on the training participants' rest windows, targeting a 5% false-positive rate. The realised test FPRs were 3.7–8.3%.

**19. What is the "arousal-directed" score?**
A one-sided robust z that counts only deviations in the physiologically expected direction of sympathetic arousal: EDA, HR and pulse rate up; pulse amplitude and skin temperature down.

**20. Why is it called post hoc?**
It was defined after the primary results showed that relaxation below baseline was being flagged. We label it exploratory because choosing an analysis after seeing the data inflates performance estimates.

## Evaluation

**21. How did you prevent leakage?**

* Folds are split by participant, so no person appears in both training and test.
* References, models and thresholds are fitted on training participants only.
* The personal baseline uses only data before the first stressor, and evaluation windows start after it.
* No window crosses a phase boundary.
* All of these rules have dedicated tests.

**22. Why participant-grouped CV and not random splits?**
Windows of one person are highly correlated. Random splits let the model "recognise the person", which inflates results (Saeb et al., 2017).

**23. Pooled AUC vs per-participant AUC?**
Pooled AUC ranks windows across people, so it rewards correct cross-person calibration. Per-participant AUC ranks within a person and ignores offsets. The two disagree for the HR rule, which is informative.

**24. Why report PR-AUC?**
The classes are imbalanced (20% stressor windows). PR-AUC's chance level is the prevalence, 0.202, and it focuses on the positive class.

**25. How did you compute confidence intervals?**
With a participant-level bootstrap: resample participants with replacement, 1,000 times, and take the 2.5th and 97.5th percentiles. This respects within-person dependence.

**26. Which statistical test compares methods?**
The Wilcoxon signed-rank test on paired per-participant AUCs, plus a paired bootstrap for the pooled AUC difference. With 8 comparisons, the Bonferroni threshold is 0.00625.

## Results

**27. Was your hypothesis confirmed?**
Not for fixed personal baselines. Personalisation reduced AUC for all four model families, significantly for robust z (−0.167, p = 0.0005) and the autoencoder (−0.219, p = 0.0002).

**28. Why did personalisation hurt?**

* Drift: skin temperature rises during the session.
* Relaxation: HR falls below baseline in rest.

Two-sided scores treat both as anomalies, and a narrow personal scale amplifies them. Removing TEMP raised the median per-participant AUC of robust z personal from 0.37 to 0.56.

**29. What worked best?**
The rolling personal baseline with arousal-directed scoring: ROC-AUC 0.796 [0.745, 0.846], median per-participant AUC 0.875. It is exploratory and needs confirmation.

**30. Which modality matters most?**
Heart rate (and pulse rate from BVP). EDA helps globally. TEMP hurts personal methods, and ACC alone is at chance.

**31. Is the system practically usable for monitoring?**
Not yet. Even the best method raises about 20 false alarms per hour of rest at 21% window-level recall. We state this openly.

**32. What happens when a sensor drops out?**
Its features become NaN and are treated as no evidence, so scoring continues. Losing PPG hurt most; losing TEMP actually helped personal methods.

**33. Did the scores agree with self-reported stress?**
Partially. Within-participant Spearman ρ was 0.63 for the HR rule and 0.62 for arousal z rolling; the two-sided personal detectors correlated negatively. This analysis is exploratory.

## Explainability

**34. How do you explain a score?**

* For robust z: the squared z of each feature over d, which sums exactly to s².
* For the autoencoder: per-feature reconstruction error.
* For IF and OCSVM: occlusion, the score drop when a feature is set to its reference value.

The UI shows each value next to its reference, the z-scores, data quality and the signals around the window.

**35. Why not SHAP?**
Our main models are additive, so exact decompositions exist. Occlusion is cheap for the rest, and SHAP would add cost and approximation without changing the conclusions.

**36. Do explanations show causes?**
No. They show how the model computed the score. Physiological causation is not established, and the UI says so.

## Engineering

**37. How are long training jobs handled?**
A background worker thread; the status lives in the database and the UI polls it. The full experiment suite runs as a separate process. The upgrade path is a task queue.

**38. What happens if a model file is deleted or corrupted?**
`Detector.load` checks the SHA-256, the feature-schema hash and required metadata. On failure the API returns 409 `model_not_ready`; it never silently uses an untrained model.

**39. What security measures exist?**

* path traversal protection (`safe_join`) and server-generated upload names;
* extension, size, zip-slip and zip-bomb checks;
* Pydantic validation with a parameter whitelist;
* ORM-only database access, which prevents SQL injection;
* a CORS allow-list;
* environment-based secrets and logs without signal values.

**40. How is reproducibility ensured?**
Fixed seeds, recorded configurations, dataset checksums, pinned dependency versions, and artifacts with complete metadata. One command regenerates every number in the report.

**41. How would you deploy it?**
The frontend goes on Netlify with `VITE_API_BASE_URL` set. The backend container goes on a host with a persistent disk, alongside managed PostgreSQL; migrations run on start-up and CORS is set to the Netlify origin. This was documented but **not** deployed in this project.

**42. What would you do next?**
Confirm E2 on WESAD with a pre-registered plan; try drift-aware personal baselines (EWMA, state-space), multi-day baselines, motion-aware suppression and HRV features.
