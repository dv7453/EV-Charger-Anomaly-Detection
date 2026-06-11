"""End-to-end orchestration for event and session scoring."""

from __future__ import annotations

import pandas as pd

from charger_anomaly.config import (
    LGBM_FEATURE_COLUMNS,
    OUT_IF_SCORE,
    OUT_IS_ANOMALY,
    OUT_LAYER1_FLAG,
    OUT_LAYER2_FLAG,
    OUT_SESSION_RISK,
)
from charger_anomaly.features import build_event_features, build_session_features
from charger_anomaly.model import AnomalyArtifact
from charger_anomaly.preprocessing import impute_missing
from charger_anomaly.rules import apply_all_rules


def run_event_scoring(
    df: pd.DataFrame,
    artifact: AnomalyArtifact,
) -> pd.DataFrame:
    """
    Score events with Layer 1 rules and Layer 2 Isolation Forest.

    Returns df with layer1_flag, if_score, and layer2_flag columns added.
    """
    out = impute_missing(df.copy(), artifact.input_medians)
    out = build_event_features(
        out,
        artifact.station_baselines,
        artifact.global_baseline,
    )

    out[OUT_LAYER1_FLAG] = apply_all_rules(out).astype(int)

    X = out[artifact.if_feature_columns]
    X_scaled = artifact.scaler.transform(X)
    scores = artifact.if_model.decision_function(X_scaled)
    out[OUT_IF_SCORE] = scores
    out[OUT_LAYER2_FLAG] = (scores < artifact.if_threshold).astype(int)
    return out


def run_session_scoring(
    event_scored_df: pd.DataFrame,
    artifact: AnomalyArtifact,
) -> pd.DataFrame:
    """
    Score sessions with Layer 3 LightGBM risk classifier.

    Returns session-level DataFrame with session_id and session_risk_score.
    """
    session_features = build_session_features(
        event_scored_df,
        artifact.station_fault_history,
        session_fault_labels=None,
    )
    session_probs = artifact.lgbm_model.predict(session_features[LGBM_FEATURE_COLUMNS])
    return pd.DataFrame(
        {
            "session_id": session_features["session_id"],
            OUT_SESSION_RISK: session_probs,
        }
    )


def run_full_pipeline(
    df: pd.DataFrame,
    artifact: AnomalyArtifact,
) -> pd.DataFrame:
    """
    Run Layer 1 + Layer 2 event scoring and attach Layer 3 session risk scores.

    Returns original columns plus layer1_flag, layer2_flag, is_anomaly,
    if_score, and session_risk_score.
    """
    original_cols = df.columns.tolist()
    event_scored = run_event_scoring(df, artifact)
    session_scores = run_session_scoring(event_scored, artifact)

    out = event_scored.merge(session_scores, on="session_id", how="left")
    out[OUT_IS_ANOMALY] = (
        (out[OUT_LAYER1_FLAG] == 1) | (out[OUT_LAYER2_FLAG] == 1)
    ).astype(int)

    new_cols = [
        OUT_LAYER1_FLAG,
        OUT_LAYER2_FLAG,
        OUT_IS_ANOMALY,
        OUT_IF_SCORE,
        OUT_SESSION_RISK,
    ]
    return out[original_cols + new_cols]
