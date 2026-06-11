# EV Charger Anomaly Detection

Unsupervised and session-level anomaly detection for EV charging station event logs. Built as a take-home ML assignment simulating a Network Operations Center (NOC) workflow.

## Architecture

| Layer | Method | Purpose |
|-------|--------|---------|
| **Layer 1** | Deterministic physics rules | High-confidence flags for station-independent impossibilities |
| **Layer 2** | Isolation Forest | Event-level multivariate outlier detection with station-relative features |
| **Layer 3** | LightGBM | Session/station concurrent risk scoring (not temporal prediction) |

An event is flagged `is_anomaly=1` if Layer 1 **or** Layer 2 fires. Each event also receives a `session_risk_score` from Layer 3.

## Project structure

```
ev-charger-anomaly-detection/
├── README.md
├── REPORT.md
├── AI_USAGE.md
├── requirements.txt
├── predict.py
├── data/charging_logs.csv
├── notebooks/01_eda.ipynb
├── src/charger_anomaly/
│   ├── config.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── rules.py
│   ├── model.py
│   └── pipeline.py
├── scripts/train.py
├── models/artifact.joblib      # bundled trained model (run train.py to regenerate)
└── outputs/
    ├── figures/
    └── metrics.json
```

## Data setup

Place the assignment dataset at:

```
data/charging_logs.csv
```

This repo includes `data/charging_logs.csv` for reproducibility. If you cloned without the data file, copy `charging_logs.csv` from the assignment packet into `data/` before training or inference.

## Setup

```bash
cd ev-charger-anomaly-detection
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick start

`models/artifact.joblib` is included in this repo so `predict.py` runs without retraining.

```bash
python predict.py --input data/charging_logs.csv --output outputs/predictions.csv
```

To retrain from scratch (regenerates the artifact and metrics):

```bash
python scripts/train.py --data data/charging_logs.csv
```

## Train

```bash
python scripts/train.py --data data/charging_logs.csv
```

Options:
- `--output-dir models/` — artifact output directory (default)
- `--target-recall 0.90` — IF threshold selection target recall on known faults

Outputs:
- `models/artifact.joblib` — bundled models, baselines, thresholds
- `outputs/metrics.json` — validation and test metrics
- `outputs/figures/` — threshold sweep and feature importance tables

## Predict

```bash
python predict.py --input data/charging_logs.csv --output outputs/predictions.csv
```

Optional:
```bash
python predict.py --input new_logs.csv --output predictions.csv --model models/artifact.joblib
```

Output columns: all original columns plus `layer1_flag`, `layer2_flag`, `is_anomaly`, `if_score`, `session_risk_score`.

## Notebook

```bash
jupyter notebook notebooks/01_eda.ipynb
```

Run from the project root so relative paths resolve correctly.

## Reproducibility

- Temporal split: train Jan–Sep, validate Oct–Nov, test Dec 2024 (session-level, no leakage)
- `features.py` is shared by training and inference to prevent train/serve skew
- Station baselines and imputation medians are saved in the artifact
- Unknown stations at inference fall back to global training baselines
