"""Feature engineering shared by training and inference pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd

from charger_anomaly.config import (
    IF_FEATURE_COLUMNS,
    LGBM_FEATURE_COLUMNS,
    RATIO_NAN_FILL,
    SESSION_ROLLING_WINDOW,
    STATION_BASELINE_COLS,
    STATION_HISTORY_WINDOW,
)
from charger_anomaly.rules import apply_all_rules


def compute_station_baselines(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    Compute per-station mean and std for STATION_BASELINE_COLS.

    Called on training data only. Result is stored in the artifact.
    """
    baselines: dict[str, dict[str, float]] = {}
    for station_id, group in df.groupby("station_id"):
        entry: dict[str, float] = {}
        for col in STATION_BASELINE_COLS:
            values = group[col].astype(float)
            entry[f"{col}_mean"] = float(values.mean())
            entry[f"{col}_std"] = float(values.std(ddof=0))
        baselines[str(station_id)] = entry
    return baselines


def compute_global_baseline(df: pd.DataFrame) -> dict[str, float]:
    """
    Compute global mean/std fallback for station_ids not seen during training.
    """
    entry: dict[str, float] = {}
    for col in STATION_BASELINE_COLS:
        values = df[col].astype(float)
        entry[f"{col}_mean"] = float(values.mean())
        entry[f"{col}_std"] = float(values.std(ddof=0))
    return entry


def add_physics_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add power_ratio and energy_ratio columns.

    Safe division: denominator <= 0 yields NaN (not inf).
    """
    out = df.copy()
    denom_power = out["voltage"] * out["current"] / 1000.0
    out["power_ratio"] = np.where(denom_power > 0, out["power_kw"] / denom_power, np.nan)

    denom_energy = out["power_kw"] * out["duration_sec"] / 3600.0
    out["energy_ratio"] = np.where(denom_energy > 0, out["energy_kwh"] / denom_energy, np.nan)
    return out


def _lookup_baseline(
    station_id: str,
    col: str,
    station_baselines: dict[str, dict[str, float]],
    global_baseline: dict[str, float],
) -> tuple[float, float]:
    """Return (mean, std) for a station/column with global fallback."""
    baseline = station_baselines.get(station_id, global_baseline)
    mean = baseline.get(f"{col}_mean", global_baseline[f"{col}_mean"])
    std = baseline.get(f"{col}_std", global_baseline[f"{col}_std"])
    return mean, std


def add_station_zscores(
    df: pd.DataFrame,
    station_baselines: dict[str, dict[str, float]],
    global_baseline: dict[str, float],
) -> pd.DataFrame:
    """
    Add station-relative z-scores for voltage, current, temperature, and power.

    Unknown stations fall back to global_baseline. Std == 0 yields zscore 0.0.
    """
    out = df.copy()
    zscore_map = {
        "voltage_zscore_station": "voltage",
        "current_zscore_station": "current",
        "temperature_zscore_station": "temperature_c",
        "power_zscore_station": "power_kw",
    }
    for feature_name, col in zscore_map.items():
        zscores = np.zeros(len(out), dtype=float)
        for idx, (station_id, value) in enumerate(zip(out["station_id"], out[col])):
            mean, std = _lookup_baseline(
                str(station_id), col, station_baselines, global_baseline
            )
            if std == 0:
                zscores[idx] = 0.0
            else:
                zscores[idx] = (float(value) - mean) / std
        out[feature_name] = zscores
    return out


def add_session_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add within-session rolling and delta features.

    Requires df sorted by (session_id, timestamp).
    """
    out = df.copy()
    grouped = out.groupby("session_id", sort=False)

    out["voltage_rolling_std_5"] = grouped["voltage"].transform(
        lambda s: s.rolling(SESSION_ROLLING_WINDOW, min_periods=1).std(ddof=0)
    )
    out["voltage_delta"] = grouped["voltage"].diff().fillna(0.0)
    out["power_delta"] = grouped["power_kw"].diff().fillna(0.0)
    out["temp_delta"] = grouped["temperature_c"].diff().fillna(0.0)

    gap = grouped["timestamp"].diff().dt.total_seconds()
    out["inter_event_gap_sec"] = gap.fillna(0.0)

    session_sizes = grouped["timestamp"].transform("size")
    event_index = grouped.cumcount()
    denom = (session_sizes - 1).clip(lower=1)
    out["event_position"] = event_index / denom

    # Fill rolling std NaNs at session head with session mean rolling std
    session_mean_std = grouped["voltage_rolling_std_5"].transform("mean")
    out["voltage_rolling_std_5"] = out["voltage_rolling_std_5"].fillna(session_mean_std)
    out["voltage_rolling_std_5"] = out["voltage_rolling_std_5"].fillna(0.0)

    return out


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour_of_day (0-23) and is_weekend (0/1)."""
    out = df.copy()
    ts = out["timestamp"]
    out["hour_of_day"] = ts.dt.hour.astype(int)
    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    return out


def add_error_code_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Add error_code_binary = (error_code != 0)."""
    out = df.copy()
    out["error_code_binary"] = (out["error_code"] != 0).astype(int)
    return out


def _fill_if_feature_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Fill remaining NaN in IF feature columns before scaling."""
    out = df.copy()
    ratio_cols = {"power_ratio", "energy_ratio"}
    for col in IF_FEATURE_COLUMNS:
        if col in ratio_cols:
            out[col] = out[col].fillna(RATIO_NAN_FILL)
        else:
            out[col] = out[col].fillna(0.0)
    return out


def build_event_features(
    df: pd.DataFrame,
    station_baselines: dict[str, dict[str, float]],
    global_baseline: dict[str, float],
) -> pd.DataFrame:
    """
    Master dispatcher for event-level features used by Layer 2.

    Returns df with all IF_FEATURE_COLUMNS present and NaN-free.
    """
    out = df.copy()
    out = add_physics_ratios(out)
    out = add_station_zscores(out, station_baselines, global_baseline)
    out = add_session_rolling_features(out)
    out = add_temporal_features(out)
    out = add_error_code_binary(out)
    out = _fill_if_feature_nans(out)
    return out


def _compute_station_recent_fault_rate(
    station_id: str,
    session_start: pd.Timestamp,
    station_fault_history: dict[str, list[dict]],
) -> float:
    """
    Compute causal backward-looking fault rate for a station at session start.

    Uses only prior sessions in station_fault_history (ordered by time).
    """
    history = station_fault_history.get(str(station_id), [])
    prior = [entry for entry in history if entry["session_start"] < session_start]
    if not prior:
        return 0.0
    recent = prior[-STATION_HISTORY_WINDOW:]
    return float(np.mean([entry["label"] for entry in recent]))


def build_session_features(
    event_df: pd.DataFrame,
    station_fault_history: dict[str, list[dict]],
    session_fault_labels: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Aggregate event_df to one row per session with Layer 3 features.

    If session_fault_labels is provided, adds a 'label' column for training.
    """
    if "layer1_flag" not in event_df.columns:
        violations = apply_all_rules(event_df)
    else:
        violations = event_df["layer1_flag"].astype(bool)

    work = event_df.copy()
    work["_physics_violation"] = violations.astype(int)

    session_groups = work.groupby("session_id", sort=False)
    rows: list[dict] = []

    for session_id, group in session_groups:
        group = group.sort_values("timestamp")
        session_start = group["timestamp"].iloc[0]
        station_id = str(group["station_id"].iloc[0])
        duration_sec = float(
            (group["timestamp"].iloc[-1] - group["timestamp"].iloc[0]).total_seconds()
        )

        row: dict[str, float | int | str] = {
            "session_id": str(session_id),
            "station_id": station_id,
            "session_mean_voltage": float(group["voltage"].mean()),
            "session_std_voltage": float(group["voltage"].std(ddof=0)),
            "session_min_voltage": float(group["voltage"].min()),
            "session_max_voltage": float(group["voltage"].max()),
            "session_mean_temperature": float(group["temperature_c"].mean()),
            "session_std_temperature": float(group["temperature_c"].std(ddof=0)),
            "session_max_temperature": float(group["temperature_c"].max()),
            "session_mean_power": float(group["power_kw"].mean()),
            "session_std_power": float(group["power_kw"].std(ddof=0)),
            "session_total_energy_kwh": float(group["energy_kwh"].sum()),
            "session_n_events": int(len(group)),
            "session_duration_sec": duration_sec,
            "session_concurrent_violation_fraction": float(group["_physics_violation"].mean()),
            "session_start_hour": int(session_start.hour),
            "session_is_weekend": int(session_start.dayofweek >= 5),
            "station_recent_fault_rate": _compute_station_recent_fault_rate(
                station_id, session_start, station_fault_history
            ),
        }
        if session_fault_labels is not None:
            row["label"] = int(session_fault_labels.get(str(session_id), 0))
        rows.append(row)

    session_df = pd.DataFrame(rows)
    if not session_df.empty:
        session_df = session_df.sort_values("session_id").reset_index(drop=True)
    return session_df[LGBM_FEATURE_COLUMNS + (["label"] if session_fault_labels is not None else []) + ["session_id", "station_id"]]


def build_station_fault_history(
    session_meta: pd.DataFrame,
) -> dict[str, list[dict]]:
    """
    Build ordered per-station session history for causal fault-rate features.

    session_meta columns: session_id, station_id, session_start, label
    """
    history: dict[str, list[dict]] = {}
    ordered = session_meta.sort_values("session_start")
    for _, row in ordered.iterrows():
        station_id = str(row["station_id"])
        entry = {
            "session_id": str(row["session_id"]),
            "label": int(row["label"]),
            "session_start": row["session_start"],
        }
        history.setdefault(station_id, []).append(entry)
    return history
