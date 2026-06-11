#!/usr/bin/env python3
"""CLI inference script for charger anomaly detection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from charger_anomaly.config import DEFAULT_ARTIFACT_PATH  # noqa: E402
from charger_anomaly.model import load_artifact  # noqa: E402
from charger_anomaly.pipeline import run_full_pipeline  # noqa: E402
from charger_anomaly.preprocessing import coerce_types, load_data  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run anomaly detection on new charging log CSV files."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input CSV path")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV path")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_ARTIFACT_PATH,
        help="Path to trained artifact.joblib",
    )
    args = parser.parse_args()

    artifact = load_artifact(args.model)
    df = load_data(args.input)
    df = coerce_types(df)

    predictions = run_full_pipeline(df, artifact)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)

    total = len(predictions)
    anomalies = int(predictions["is_anomaly"].sum())
    layer1_only = int(
        ((predictions["layer1_flag"] == 1) & (predictions["layer2_flag"] == 0)).sum()
    )
    layer2_only = int(
        ((predictions["layer2_flag"] == 1) & (predictions["layer1_flag"] == 0)).sum()
    )
    both = int(
        ((predictions["layer1_flag"] == 1) & (predictions["layer2_flag"] == 1)).sum()
    )
    high_risk_sessions = int((predictions.groupby("session_id")["session_risk_score"].first() > 0.5).sum())
    total_sessions = predictions["session_id"].nunique()

    print(f"Total events processed: {total}")
    print(f"Anomalies flagged (is_anomaly=1): {anomalies} ({100 * anomalies / total:.2f}%)")
    print(f"Layer 1 only: {layer1_only} | Layer 2 only: {layer2_only} | Both: {both}")
    print(f"Sessions with risk_score > 0.5: {high_risk_sessions} / {total_sessions}")
    print(f"Predictions written to {args.output}")


if __name__ == "__main__":
    main()
