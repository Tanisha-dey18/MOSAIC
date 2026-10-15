# Datasets

Raw data is **never committed** to the repository (see `.gitignore`). This folder is the root from which the backend may ingest server-side data (`DATASETS_DIR`); paths outside it are rejected.

## Primary dataset (verified)

| Field | Value |
|---|---|
| Name | Wearable Device Dataset from Induced Stress and Structured Exercise Sessions |
| Version | 1.0.1 (published 24 June 2025) |
| Authors | Andrea Hongn, Facundo Bosch, Lara Prado, Paula Bonomini |
| DOI | 10.13026/he0v-tf17 |
| Landing page | https://physionet.org/content/wearable-device-dataset/1.0.1/ |
| License | Open Data Commons Attribution License v1.0 (open access, attribution required) |
| Device | Empatica E4 on the non-dominant wrist |
| Size | ~247 MB uncompressed, 708 files (checksums in `SHA256SUMS.txt`) |

**Structure** (verified after download): `Wearable_Dataset/{STRESS,AEROBIC,ANAEROBIC}/<subject>/{ACC,BVP,EDA,HR,IBI,TEMP,tags}.csv`, plus `Stress_Level_v1.csv`, `Stress_Level_v2.csv`, `subject-info.csv`, `data_constraints.txt`, `Data_Dictionary.csv` and `Wearable_Dataset.ipynb`.

| Signal | Rate | Unit |
|---|---|---|
| EDA | 4 Hz | µS |
| BVP | 64 Hz | a.u. |
| HR | 1 Hz (derived from BVP, starts 10 s after the other signals) | bpm |
| TEMP | 4 Hz | °C |
| ACC | 32 Hz, 3 axes | 1/64 g |
| IBI | event-based | s |

The sessions found locally are 37 STRESS folders (36 participants; f14 is split into f14_a/f14_b), 31 AEROBIC and 32 ANAEROBIC. Participants are pseudonymous (`S01`–`S18`, `f01`–`f18`). Dates are shifted for de-identification.

**Known issues** (from `data_constraints.txt`, encoded in `ml/data_ingestion/physionet_wearable.py`):

* STRESS S02 has duplicated signal segments. It is excluded from modelling.
* STRESS f07 had the dock left on, so its PPG and temperature are invalid. Those modalities are marked invalid; the participant becomes ineligible because complete baseline windows are missing.
* STRESS f14 is split into two recordings. It is excluded because its personal baseline cannot be built causally.
* AEROBIC S11 and ANAEROBIC S16 are split recordings. They are kept for exploration but are unlabelled.

## Download

```bash
python scripts/download_dataset.py --out datasets/raw/wearable-device-dataset-1.0.1
```

The script fetches every file listed in the dataset's `SHA256SUMS.txt` from PhysioNet's official open-data bucket (`physionet-open` on AWS S3, which mirrors physionet.org) and verifies each checksum. The ZIP from physionet.org (~70 MB) is equivalent but was very slow when tested.

## Ingest

```bash
python scripts/ingest.py --path datasets/raw/wearable-device-dataset-1.0.1
```

Alternatively, use the Data Explorer page or `POST /api/dataset/ingest {"path": "raw/wearable-device-dataset-1.0.1"}`.

## Own data (generic CSV)

You can upload a long-format CSV with columns `timestamp` (ISO-8601 or Unix seconds) and `participant_id`, plus any of `eda`, `hr`, `temp`, `bvp`, `acc_x/acc_y/acc_z` (or `acc`). The optional columns are `session_id`, `label` (0/1), `phase` and `calibration` (1 for the leading rows that may serve as a personal baseline). Only upload data you are permitted to process, and use pseudonymous identifiers.

## Synthetic data

`ml/tests/synthetic.py` generates **synthetic** files in the same format, used **only** by the automated tests. Synthetic data is never shown in the application or used for reported results.
