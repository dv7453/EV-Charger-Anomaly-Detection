"""Data loading, type coercion, and missing-value imputation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from charger_anomaly.config import NUMERIC_COLS, RAW_COLS


def load_data(path: str | Path) -> pd.DataFrame:
    """
    Load CSV, parse timestamp as UTC, sort by (station_id, session_id, timestamp).

    Raises:
        ValueError: If required columns are missing.
    """
    path = Path(path)
    df = pd.read_csv(path, parse_dates=["timestamp"])
    missing = set(RAW_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[RAW_COLS].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["station_id", "session_id", "timestamp"]).reset_index(drop=True)
    return df


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast each column to expected dtype.

    Unparseable numeric values become NaN rather than raising.
    """
    out = df.copy()
    for col in NUMERIC_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["error_code"] = pd.to_numeric(out["error_code"], errors="coerce")
    out["station_id"] = out["station_id"].astype(str)
    out["session_id"] = out["session_id"].astype(str)
    out["message"] = out["message"].astype(str)
    return out


def compute_input_medians(df: pd.DataFrame) -> dict[str, float]:
    """
    Compute per-column medians from training data for use in impute_missing.

    Called once during training; result stored in the artifact.
    """
    medians: dict[str, float] = {}
    for col in NUMERIC_COLS:
        medians[col] = float(df[col].median())
    return medians


def impute_missing(
    df: pd.DataFrame,
    medians: dict[str, float],
) -> pd.DataFrame:
    """
    Fill NaN in numeric columns using provided medians dict.

    Fills message NaN with 'OK' and error_code NaN with 0.
    """
    out = df.copy()
    for col in NUMERIC_COLS:
        out[col] = out[col].fillna(medians[col])
    out["message"] = out["message"].fillna("OK")
    out["error_code"] = out["error_code"].fillna(0).astype(int)
    return out
