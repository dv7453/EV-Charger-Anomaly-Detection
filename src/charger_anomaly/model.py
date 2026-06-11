"""Model training, threshold selection, and artifact persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from charger_anomaly.config import (
    IF_CONTAMINATION,
    IF_N_ESTIMATORS,
    IF_RANDOM_STATE,
    IF_TARGET_RECALL,
    LGBM_EARLY_STOP,
    LGBM_FEATURE_COLUMNS,
    LGBM_LR,
    LGBM_N_ESTIMATORS,
    LGBM_NUM_LEAVES,
    LGBM_RANDOM_STATE,
)


@dataclass
class AnomalyArtifact:
    """Serializable bundle of models, scalers, baselines, and thresholds."""

    if_model: IsolationForest
    scaler: StandardScaler
    if_feature_columns: list[str]
    if_threshold: float

    lgbm_model: lgb.Booster
    lgbm_feature_columns: list[str]

    station_baselines: dict[str, dict[str, float]]
    global_baseline: dict[str, float]
    station_fault_history: dict[str, list[dict]]

    input_medians: dict[str, float]

    metadata: dict[str, Any] = field(default_factory=dict)


def train_isolation_forest(
    X: pd.DataFrame,
    contamination: float = IF_CONTAMINATION,
    n_estimators: int = IF_N_ESTIMATORS,
    random_state: int = IF_RANDOM_STATE,
) -> tuple[StandardScaler, IsolationForest]:
    """Fit scaler then Isolation Forest. Returns (scaler, if_model)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if_model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    if_model.fit(X_scaled)
    return scaler, if_model


def select_if_threshold(
    if_model: IsolationForest,
    scaler: StandardScaler,
    X_val: pd.DataFrame,
    positive_mask: pd.Series,
    target_recall: float = IF_TARGET_RECALL,
) -> tuple[float, pd.DataFrame]:
    """
    Sweep threshold candidates on decision_function scores.

    Selection rule: lowest flag rate achieving >= target_recall on positives.
    """
    X_scaled = scaler.transform(X_val)
    scores = if_model.decision_function(X_scaled)
    positives = positive_mask.to_numpy(dtype=bool)

    if positives.sum() == 0:
        raise ValueError("No positive samples in validation set for threshold selection.")

    candidates = np.linspace(scores.min(), scores.max(), num=100)
    rows: list[dict[str, float]] = []

    for threshold in candidates:
        flagged = scores < threshold
        tp = int(np.sum(flagged & positives))
        fp = int(np.sum(flagged & ~positives))
        fn = int(np.sum(~flagged & positives))
        recall = tp / positives.sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        flag_rate = flagged.mean()
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        rows.append(
            {
                "threshold": float(threshold),
                "recall": float(recall),
                "precision": float(precision),
                "flag_rate": float(flag_rate),
                "f1": float(f1),
                "tp": float(tp),
                "fp": float(fp),
                "fn": float(fn),
            }
        )

    sweep_df = pd.DataFrame(rows)
    eligible = sweep_df[sweep_df["recall"] >= target_recall]
    if eligible.empty:
        raise ValueError(
            f"No threshold achieves target recall >= {target_recall:.2f} on validation positives."
        )

    best = eligible.sort_values(["flag_rate", "threshold"], ascending=[True, False]).iloc[0]
    return float(best["threshold"]), sweep_df


def train_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> lgb.Booster:
    """Train LightGBM binary classifier with early stopping on validation AUC."""
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    train_set = lgb.Dataset(X_train[LGBM_FEATURE_COLUMNS], label=y_train)
    val_set = lgb.Dataset(X_val[LGBM_FEATURE_COLUMNS], label=y_val, reference=train_set)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": LGBM_LR,
        "num_leaves": LGBM_NUM_LEAVES,
        "verbosity": -1,
        "seed": LGBM_RANDOM_STATE,
        "scale_pos_weight": scale_pos_weight,
    }

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=LGBM_N_ESTIMATORS,
        valid_sets=[val_set],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=LGBM_EARLY_STOP, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return booster


def save_artifact(artifact: AnomalyArtifact, path: str | Path) -> None:
    """Persist artifact to disk via joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path, compress=3)


def load_artifact(path: str | Path) -> AnomalyArtifact:
    """Load artifact from disk. Raises FileNotFoundError with a helpful message."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Run scripts/train.py first."
        )
    return joblib.load(path)
