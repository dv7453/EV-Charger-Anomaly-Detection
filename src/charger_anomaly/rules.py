"""Layer 1 deterministic physics rules (station-independent impossibilities)."""

from __future__ import annotations

import pandas as pd

from charger_anomaly.config import (
    ENERGY_RATIO_HIGH,
    ENERGY_RATIO_LOW,
    POWER_RATIO_HIGH,
    POWER_RATIO_LOW,
)


def flag_zero_current_positive_power(df: pd.DataFrame) -> pd.Series:
    """Return bool Series: (current == 0) & (power_kw > 0)."""
    return (df["current"] == 0) & (df["power_kw"] > 0)


def flag_negative_power(df: pd.DataFrame) -> pd.Series:
    """Return bool Series: power_kw < 0."""
    return df["power_kw"] < 0


def flag_power_ratio_violation(
    df: pd.DataFrame,
    low: float = POWER_RATIO_LOW,
    high: float = POWER_RATIO_HIGH,
) -> pd.Series:
    """
    Return bool Series: power_ratio outside [low, high].

    Requires power_ratio column. Rows where power_ratio is NaN return False.
    """
    ratio = df["power_ratio"]
    valid = ratio.notna()
    return valid & ((ratio < low) | (ratio > high))


def flag_energy_ratio_violation(
    df: pd.DataFrame,
    low: float = ENERGY_RATIO_LOW,
    high: float = ENERGY_RATIO_HIGH,
) -> pd.Series:
    """
    Return bool Series: energy_ratio outside [low, high].

    Requires energy_ratio column. Rows where energy_ratio is NaN return False.
    """
    ratio = df["energy_ratio"]
    valid = ratio.notna()
    return valid & ((ratio < low) | (ratio > high))


def apply_all_rules(df: pd.DataFrame) -> pd.Series:
    """
    Return bool Series: True if ANY of the four Layer 1 rules fire.

    Requires power_ratio and energy_ratio columns to be present.
    """
    return (
        flag_zero_current_positive_power(df)
        | flag_negative_power(df)
        | flag_power_ratio_violation(df)
        | flag_energy_ratio_violation(df)
    )


def get_rule_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return DataFrame with four bool columns for diagnostics/reporting.

    Columns: rule_zero_current, rule_negative_power,
             rule_power_ratio, rule_energy_ratio.
    """
    return pd.DataFrame(
        {
            "rule_zero_current": flag_zero_current_positive_power(df),
            "rule_negative_power": flag_negative_power(df),
            "rule_power_ratio": flag_power_ratio_violation(df),
            "rule_energy_ratio": flag_energy_ratio_violation(df),
        },
        index=df.index,
    )
