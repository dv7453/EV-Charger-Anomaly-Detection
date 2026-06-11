"""Configuration constants for the charger anomaly detection pipeline."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
DEFAULT_DATA_PATH = DATA_DIR / "charging_logs.csv"
DEFAULT_ARTIFACT_PATH = MODELS_DIR / "artifact.joblib"
DEFAULT_METRICS_PATH = OUTPUTS_DIR / "metrics.json"

# ---------------------------------------------------------------------------
# Raw column names
# ---------------------------------------------------------------------------
RAW_COLS: list[str] = [
    "station_id",
    "timestamp",
    "session_id",
    "voltage",
    "current",
    "power_kw",
    "temperature_c",
    "duration_sec",
    "energy_kwh",
    "error_code",
    "message",
]

NUMERIC_COLS: list[str] = [
    "voltage",
    "current",
    "power_kw",
    "temperature_c",
    "duration_sec",
    "energy_kwh",
]

STATION_BASELINE_COLS: list[str] = [
    "voltage",
    "current",
    "power_kw",
    "temperature_c",
]

# ---------------------------------------------------------------------------
# Layer 1 thresholds (station-independent physics impossibilities)
# ---------------------------------------------------------------------------
POWER_RATIO_LOW = 0.8
POWER_RATIO_HIGH = 1.2
ENERGY_RATIO_LOW = 0.5
ENERGY_RATIO_HIGH = 2.0

# ---------------------------------------------------------------------------
# Layer 2 — Isolation Forest
# ---------------------------------------------------------------------------
IF_N_ESTIMATORS = 200
IF_CONTAMINATION = 0.05  # prior only; real threshold selected empirically
IF_RANDOM_STATE = 42
IF_TARGET_RECALL = 0.90

IF_FEATURE_COLUMNS: list[str] = [
    "voltage_zscore_station",
    "current_zscore_station",
    "temperature_zscore_station",
    "power_zscore_station",
    "power_ratio",
    "energy_ratio",
    "voltage_rolling_std_5",
    "voltage_delta",
    "power_delta",
    "temp_delta",
    "inter_event_gap_sec",
    "event_position",
    "hour_of_day",
    "is_weekend",
    "error_code_binary",
]

# ---------------------------------------------------------------------------
# Layer 3 — LightGBM session risk classifier
# ---------------------------------------------------------------------------
LGBM_NUM_LEAVES = 31
LGBM_LR = 0.05
LGBM_N_ESTIMATORS = 500
LGBM_EARLY_STOP = 30
LGBM_RANDOM_STATE = 42

LGBM_FEATURE_COLUMNS: list[str] = [
    "session_mean_voltage",
    "session_std_voltage",
    "session_min_voltage",
    "session_max_voltage",
    "session_mean_temperature",
    "session_std_temperature",
    "session_max_temperature",
    "session_mean_power",
    "session_std_power",
    "session_total_energy_kwh",
    "session_n_events",
    "session_duration_sec",
    "session_concurrent_violation_fraction",
    "session_start_hour",
    "session_is_weekend",
    "station_recent_fault_rate",
]

# ---------------------------------------------------------------------------
# Rolling / history windows
# ---------------------------------------------------------------------------
SESSION_ROLLING_WINDOW = 5
STATION_HISTORY_WINDOW = 10

# ---------------------------------------------------------------------------
# Temporal split boundaries (session start time, inclusive)
# ---------------------------------------------------------------------------
TRAIN_END = "2024-09-30"
VAL_END = "2024-11-30"

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
RATIO_NAN_FILL = 1.0

# ---------------------------------------------------------------------------
# Output columns added by predict.py
# ---------------------------------------------------------------------------
OUT_LAYER1_FLAG = "layer1_flag"
OUT_LAYER2_FLAG = "layer2_flag"
OUT_IS_ANOMALY = "is_anomaly"
OUT_IF_SCORE = "if_score"
OUT_SESSION_RISK = "session_risk_score"

# ---------------------------------------------------------------------------
# Label / evaluation helpers
# ---------------------------------------------------------------------------
FAULT_MESSAGES: set[str] = {
    "Inconsistent metering data observed",
    "Unexpected reboot during active session",
    "Repeated handshake failures across multiple modules",
    "Unknown hardware fault code: 0x7F3A",
    "Severe voltage instability detected",
}

OK_REF_PREFIX = "OK (ref="
