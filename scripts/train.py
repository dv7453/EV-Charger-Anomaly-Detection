#!/usr/bin/env python3
"""Train Layer 2 Isolation Forest and Layer 3 LightGBM session risk classifier."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from charger_anomaly.config import (  # noqa: E402
    DEFAULT_ARTIFACT_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_METRICS_PATH,
    FAULT_MESSAGES,
    FIGURES_DIR,
    IF_FEATURE_COLUMNS,
    IF_TARGET_RECALL,
    LGBM_FEATURE_COLUMNS,
    OK_REF_PREFIX,
    OUTPUTS_DIR,
    TRAIN_END,
    VAL_END,
)
from charger_anomaly.features import (  # noqa: E402
    build_event_features,
    build_session_features,
    build_station_fault_history,
    compute_global_baseline,
    compute_station_baselines,
)
from charger_anomaly.model import (  # noqa: E402
    AnomalyArtifact,
    save_artifact,
    select_if_threshold,
    train_isolation_forest,
    train_lgbm,
)
from charger_anomaly.pipeline import run_full_pipeline  # noqa: E402
from charger_anomaly.preprocessing import (  # noqa: E402
    coerce_types,
    compute_input_medians,
    impute_missing,
    load_data,
)
from charger_anomaly.rules import apply_all_rules  # noqa: E402


def is_known_fault_row(df: pd.DataFrame) -> pd.Series:
    """Set A: explicit labeled faults."""
    return (df["error_code"] != 0) | df["message"].isin(FAULT_MESSAGES)


def is_ok_ref_row(df: pd.DataFrame) -> pd.Series:
    """Set B: subtle planted anomalies."""
    return df["message"].str.startswith(OK_REF_PREFIX, na=False)


def build_session_labels(df: pd.DataFrame) -> dict[str, int]:
    """Session label = 1 if any known fault row appears in the session."""
    fault_rows = is_known_fault_row(df)
    labels = (
        df.assign(_fault=fault_rows.astype(int))
        .groupby("session_id")["_fault"]
        .max()
        .astype(int)
        .to_dict()
    )
    return {str(k): int(v) for k, v in labels.items()}


def build_session_meta(df: pd.DataFrame, session_labels: dict[str, int]) -> pd.DataFrame:
    """Build session-level metadata for temporal splitting and history."""
    meta = (
        df.groupby("session_id", as_index=False)
        .agg(
            station_id=("station_id", "first"),
            session_start=("timestamp", "min"),
        )
        .assign(label=lambda x: x["session_id"].map(session_labels).fillna(0).astype(int))
    )
    return meta


def assign_split(session_meta: pd.DataFrame) -> pd.Series:
    """Assign train / val / test split based on session start time."""
    train_end = pd.Timestamp(TRAIN_END, tz="UTC")
    val_end = pd.Timestamp(VAL_END, tz="UTC")

    def _split(ts: pd.Timestamp) -> str:
        if ts <= train_end:
            return "train"
        if ts <= val_end:
            return "val"
        return "test"

    return session_meta["session_start"].map(_split)


def evaluate_event_level(
    df: pd.DataFrame,
    positive_mask: pd.Series,
    prefix: str,
) -> dict[str, float]:
    """Compute recall, apparent precision, and flag rate for event predictions."""
    flagged = df["is_anomaly"] == 1
    positives = positive_mask.to_numpy(dtype=bool)
    tp = int(np.sum(flagged & positives))
    fp = int(np.sum(flagged & ~positives))
    fn = int(np.sum(~flagged & positives))
    recall = tp / positives.sum() if positives.sum() > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return {
        f"{prefix}_recall": float(recall),
        f"{prefix}_apparent_precision": float(precision),
        f"{prefix}_flag_rate": float(flagged.mean()),
        f"{prefix}_tp": float(tp),
        f"{prefix}_fp": float(fp),
        f"{prefix}_fn": float(fn),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train charger anomaly detection models.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--artifact-name", type=str, default="artifact.joblib")
    parser.add_argument("--target-recall", type=float, default=IF_TARGET_RECALL)
    args = parser.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_data(args.data)
    df = coerce_types(df)

    session_labels = build_session_labels(df)
    session_meta = build_session_meta(df, session_labels)
    session_meta["split"] = assign_split(session_meta)

    train_sessions = set(session_meta.loc[session_meta["split"] == "train", "session_id"])
    val_sessions = set(session_meta.loc[session_meta["split"] == "val", "session_id"])
    test_sessions = set(session_meta.loc[session_meta["split"] == "test", "session_id"])

    train_df = df[df["session_id"].isin(train_sessions)].copy()
    val_df = df[df["session_id"].isin(val_sessions)].copy()
    test_df = df[df["session_id"].isin(test_sessions)].copy()

    print(
        f"Sessions — train: {len(train_sessions)}, val: {len(val_sessions)}, "
        f"test: {len(test_sessions)}"
    )
    print(
        f"Events — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}"
    )

    input_medians = compute_input_medians(train_df)
    train_df = impute_missing(train_df, input_medians)
    val_df = impute_missing(val_df, input_medians)
    test_df = impute_missing(test_df, input_medians)

    station_baselines = compute_station_baselines(train_df)
    global_baseline = compute_global_baseline(train_df)

    train_events = build_event_features(train_df, station_baselines, global_baseline)
    val_events = build_event_features(val_df, station_baselines, global_baseline)

    train_events["layer1_flag"] = apply_all_rules(train_events).astype(int)
    val_events["layer1_flag"] = apply_all_rules(val_events).astype(int)

    print("Training Isolation Forest...")
    scaler, if_model = train_isolation_forest(train_events[IF_FEATURE_COLUMNS])

    val_positive_mask = is_known_fault_row(val_df)
    threshold, sweep_df = select_if_threshold(
        if_model,
        scaler,
        val_events[IF_FEATURE_COLUMNS],
        val_positive_mask,
        target_recall=args.target_recall,
    )
    sweep_path = FIGURES_DIR / "if_threshold_sweep.csv"
    sweep_df.to_csv(sweep_path, index=False)
    print(f"Selected IF threshold: {threshold:.4f}")
    print(f"Threshold sweep saved to {sweep_path}")

    val_scores = if_model.decision_function(scaler.transform(val_events[IF_FEATURE_COLUMNS]))
    val_flagged = val_scores < threshold
    val_recall = float(
        np.mean(val_flagged[val_positive_mask.to_numpy()])
        if val_positive_mask.any()
        else 0.0
    )
    val_flag_rate = float(val_flagged.mean())

    history_meta = session_meta[session_meta["split"].isin(["train", "val"])].copy()
    station_fault_history = build_station_fault_history(history_meta)

    print("Training LightGBM session risk classifier...")
    train_session_features = build_session_features(
        train_events,
        station_fault_history,
        session_fault_labels=session_labels,
    )
    val_session_features = build_session_features(
        val_events,
        station_fault_history,
        session_fault_labels=session_labels,
    )

    lgbm_model = train_lgbm(
        train_session_features,
        train_session_features["label"],
        val_session_features,
        val_session_features["label"],
    )

    val_probs = lgbm_model.predict(val_session_features[LGBM_FEATURE_COLUMNS])
    val_auc = float(
        roc_auc_score(val_session_features["label"], val_probs)
        if val_session_features["label"].nunique() > 1
        else 0.0
    )
    val_ap = float(
        average_precision_score(val_session_features["label"], val_probs)
        if val_session_features["label"].nunique() > 1
        else 0.0
    )
    print(f"Validation session AUC: {val_auc:.4f}, AP: {val_ap:.4f}")

    artifact = AnomalyArtifact(
        if_model=if_model,
        scaler=scaler,
        if_feature_columns=IF_FEATURE_COLUMNS.copy(),
        if_threshold=threshold,
        lgbm_model=lgbm_model,
        lgbm_feature_columns=LGBM_FEATURE_COLUMNS.copy(),
        station_baselines=station_baselines,
        global_baseline=global_baseline,
        station_fault_history=station_fault_history,
        input_medians=input_medians,
        metadata={
            "version": "1.0",
            "train_date": datetime.now(UTC).date().isoformat(),
            "train_range": ["2024-01-01", TRAIN_END],
            "n_train_rows": int(len(train_df)),
            "n_train_sessions": int(len(train_sessions)),
            "n_positive_sessions": int(sum(session_labels[s] for s in train_sessions)),
            "if_val_recall": val_recall,
            "if_val_flag_rate": val_flag_rate,
            "lgbm_val_auc": val_auc,
            "lgbm_val_ap": val_ap,
        },
    )

    artifact_path = args.output_dir / args.artifact_name
    save_artifact(artifact, artifact_path)
    print(f"Artifact saved to {artifact_path}")

    print("Evaluating on test set...")
    test_predictions = run_full_pipeline(test_df, artifact)

    metrics: dict[str, float | dict[str, float]] = {
        "validation": {
            "if_threshold": float(threshold),
            "if_recall_known_faults": val_recall,
            "if_flag_rate": val_flag_rate,
            "lgbm_auc": val_auc,
            "lgbm_ap": val_ap,
        },
        "test_event_level": {},
        "test_layer_contributions": {},
    }

    known_fault_mask = is_known_fault_row(test_df)
    ok_ref_mask = is_ok_ref_row(test_df)
    physics_mask = apply_all_rules(
        build_event_features(test_df, station_baselines, global_baseline)
    )

    metrics["test_event_level"].update(
        evaluate_event_level(test_predictions, known_fault_mask, "known_faults")
    )
    metrics["test_event_level"].update(
        evaluate_event_level(test_predictions, ok_ref_mask, "ok_ref")
    )
    metrics["test_event_level"].update(
        evaluate_event_level(test_predictions, physics_mask, "physics_violations")
    )

    layer1 = test_predictions["layer1_flag"] == 1
    layer2 = test_predictions["layer2_flag"] == 1
    both = layer1 & layer2
    metrics["test_layer_contributions"] = {
        "layer1_only": int(np.sum(layer1 & ~layer2)),
        "layer2_only": int(np.sum(layer2 & ~layer1)),
        "both": int(np.sum(both)),
        "total_anomalies": int(test_predictions["is_anomaly"].sum()),
        "total_events": int(len(test_predictions)),
    }

    test_session_features = build_session_features(
        test_predictions,
        station_fault_history,
        session_fault_labels=session_labels,
    )
    test_session_probs = lgbm_model.predict(test_session_features[LGBM_FEATURE_COLUMNS])
    test_labels = test_session_features["label"]
    metrics["test_session_level"] = {
        "auc": float(roc_auc_score(test_labels, test_session_probs))
        if test_labels.nunique() > 1
        else 0.0,
        "ap": float(average_precision_score(test_labels, test_session_probs))
        if test_labels.nunique() > 1
        else 0.0,
        "sessions_high_risk_gt_0_5": int((test_session_probs > 0.5).sum()),
        "total_sessions": int(len(test_session_features)),
    }

    # IsolationForest has no native importances; use permutation on a train sample.
    sample = train_events[IF_FEATURE_COLUMNS].sample(
        n=min(5000, len(train_events)), random_state=42
    )
    base_scores = if_model.decision_function(scaler.transform(sample))
    importances = []
    for col in IF_FEATURE_COLUMNS:
        perturbed = sample.copy()
        perturbed[col] = np.random.permutation(perturbed[col].values)
        pert_scores = if_model.decision_function(scaler.transform(perturbed))
        importances.append(float(np.mean(np.abs(pert_scores - base_scores))))
    if_importance = pd.DataFrame({"feature": IF_FEATURE_COLUMNS, "importance": importances})
    if_importance = if_importance.sort_values("importance", ascending=False)
    if_importance.to_csv(FIGURES_DIR / "if_feature_importance.csv", index=False)

    lgbm_importance = pd.DataFrame(
        {
            "feature": LGBM_FEATURE_COLUMNS,
            "importance": lgbm_model.feature_importance(importance_type="gain"),
        }
    ).sort_values("importance", ascending=False)
    lgbm_importance.to_csv(FIGURES_DIR / "lgbm_feature_importance.csv", index=False)

    metrics["if_feature_importance"] = if_importance.set_index("feature")["importance"].to_dict()
    metrics["lgbm_feature_importance"] = lgbm_importance.set_index("feature")[
        "importance"
    ].to_dict()

    with open(DEFAULT_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {DEFAULT_METRICS_PATH}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
