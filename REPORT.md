# Technical Report: EV Charger Anomaly Detection

## 1. Problem Understanding

Network Operations Center (NOC) teams monitoring EV charging infrastructure need to surface anomalous events from high-volume telemetry streams without overwhelming operators with false alerts. This assignment provides ~200K synthetic charging **events** (not full sessions) spanning 20 stations and 4,000 sessions across 2024.

The core challenge is **mostly unsupervised anomaly detection** with partial labels:

| Anomaly type | Approx. size | Label availability |
|---|---|---|
| Labeled faults (`error_code ≠ 0`, explicit fault messages) | ~1,109 events | Partial positive labels |
| Physical outliers (voltage >260V or <200V, temp >55°C in OK rows) | ~2,300 events | Unlabeled |
| Physics violations (power/energy inconsistency, zero current with power, negative power) | ~3,400 events | Unlabeled; zero overlap with labeled faults |
| Subtle planted (`OK (ref=...)`) | ~1,172 events | Evaluation-only third set |

953 of 4,000 sessions contain at least one labeled fault, providing implicit session-level labels. Fault events are uniformly distributed across session position — there is **no within-session early-warning temporal signal**.

**Design implication:** We detect concurrent anomalies at the event level and score session/station risk concurrently — we do not claim to predict faults before they occur within a session.

---

## 2. Key EDA Insights

### Station heterogeneity is severe
Stations operate at materially different baselines (e.g., STATION_6 mean voltage ~246V vs STATION_8 ~226V). Global thresholds on raw telemetry produce false positives. **Station-relative z-scores** (computed from training data, with global fallback for unseen stations) are mandatory.

### Three anomaly populations are complementary
- **Labeled faults** cluster around explicit error codes and messages.
- **Physics violations** are entirely unlabeled and invisible to error-code-based rules.
- **OK (ref=...)** rows have normal-looking telemetry but anomalous messages — useful for evaluating whether multivariate models catch subtle cases.

### Session labels are usable but not predictive within-session
~24% of sessions are fault-positive. Fault position within sessions is uniform, so session aggregates reflect **concurrent risk indicators**, not leading indicators of future faults in the same session.

---

## 3. Modeling Approach

### Layer 1 — Deterministic Physics Rules (no ML)

Flags station-independent physical impossibilities:

| Rule | Condition |
|---|---|
| Zero current, positive power | `current == 0` and `power_kw > 0` |
| Negative power | `power_kw < 0` |
| Power ratio violation | `power_kw / (V×I/1000)` outside [0.8, 1.2] |
| Energy ratio violation | `energy_kwh / (power×duration/3600)` outside [0.5, 2.0] |

**Rationale:** High precision, zero latency, fully explainable. Catches the unlabeled physics-violation class that labeled-fault signals miss entirely.

Voltage and temperature outlier detection is **not** handled here — those are station-relative and delegated to Layer 2.

### Layer 2 — Isolation Forest (event-level)

**Features (15):** `voltage_zscore_station`, `current_zscore_station`, `temperature_zscore_station`, `power_zscore_station`, `power_ratio`, `energy_ratio`, `voltage_rolling_std_5`, `voltage_delta`, `power_delta`, `temp_delta`, `inter_event_gap_sec`, `event_position`, `hour_of_day`, `is_weekend`, `error_code_binary`.

**Rationale:** Isolation Forest handles mixed feature types, scales to ~200K rows, and finds multivariate outliers without requiring complete labels. `contamination=0.05` is used only as a fitting prior; the operating threshold is selected empirically.

**Feature importance note:** Permutation-based importance shows `error_code_binary` has minimal influence (ranked last), so Layer 2 is not dominated by the labeled-fault flag. Top contributors are temporal and volatility features (`is_weekend`, `voltage_rolling_std_5`, `inter_event_gap_sec`).

### Layer 3 — LightGBM Session Risk Scoring (not prediction)

**Label:** Session = 1 if it contains any labeled fault row. `OK (ref=...)` rows are excluded from labels.

**Features (16):** Session aggregates (mean/std/min/max of voltage, temperature, power), `session_total_energy_kwh`, `session_n_events`, `session_duration_sec`, `session_concurrent_violation_fraction`, `session_start_hour`, `session_is_weekend`, `station_recent_fault_rate`.

**Honest framing:**
- `session_concurrent_violation_fraction` is a **concurrent risk indicator** — it rises because anomalies are already present in the session.
- `station_recent_fault_rate` is the primary **genuinely backward-looking/predictive** signal: fraction of the station's last 10 sessions that contained a fault, computed causally.
- At inference, `station_recent_fault_rate` is **static** (seeded from training-period history). Streaming updates would be a production improvement.

**Rationale:** Helps NOC operators prioritize sessions and identify degrading stations, even when individual events look borderline.

### Final output logic

```
is_anomaly = layer1_flag OR layer2_flag
session_risk_score = LightGBM probability (attached to all events in session)
```

---

## 4. Evaluation Methodology

### Data split (temporal, session-grouped)
| Split | Period | Sessions | Events |
|---|---|---|---|
| Train | Jan–Sep 2024 | 2,970 | 148,059 |
| Validation | Oct–Nov 2024 | 699 | 35,172 |
| Test | Dec 2024 | 331 | 16,335 |

No session straddles a split boundary.

### Evaluation sets
| Set | Used for |
|---|---|
| **A — Known faults** | IF threshold selection; primary recall metric |
| **B — OK (ref=...)** | Held-out third evaluation; not used for tuning |
| **C — Physics violations** | Layer 1 recall verification |

### Threshold selection (PU-aware)
Because most rows are unlabeled, precision against Set A alone is a **lower bound** — many apparent false positives are true anomalies in Sets B or C.

Selection rule: lowest flag rate achieving ≥90% recall on known faults in the validation set.

---

## 5. Results

### Validation (Oct–Nov 2024)
| Metric | Value |
|---|---|
| IF threshold | 0.037 |
| IF recall (known faults) | 92.0% |
| IF flag rate | 11.0% |
| LightGBM session AUC | 0.631 |
| LightGBM session AP | 0.345 |

### Test (Dec 2024) — Event level

| Set | Recall | Apparent precision* | Flag rate |
|---|---|---|---|
| Known faults (A) | 90.7% | 4.2% | 11.5% |
| OK (ref=...) (B) | 8.2% | 0.4% | 11.5% |
| Physics violations (C) | 100% | 24.7% | 11.5% |

\*Apparent precision is a lower bound due to unlabeled anomalies in the negative pool.

### Layer contributions (test set)
| Source | Count |
|---|---|
| Layer 1 only | 19 |
| Layer 2 only | 1,414 |
| Both layers | 446 |
| **Total flagged** | **1,879 / 16,335 (11.5%)** |

Layer 1 catches physics impossibilities with high confidence. Layer 2 adds broader multivariate coverage. Overlap (446) indicates events violating both physics and multivariate norms.

### Session level (test)
| Metric | Value |
|---|---|
| AUC | 0.593 |
| Average Precision | 0.294 |
| Sessions with risk > 0.5 | 0 / 331 |

Session-level performance is modest. Top LightGBM features are concurrent aggregates (`session_total_energy_kwh`, `session_duration_sec`, `session_std_power`) rather than `station_recent_fault_rate` (ranked 15th). This confirms the EDA finding: within-session aggregates reflect concurrent state, and the causal station history signal is weaker at this granularity.

### Interpretation of false positives and false negatives

**False positives (apparent):** Many flagged events are likely true unlabeled anomalies (physics violations, subtle outliers). The 4.2% apparent precision on known faults is not representative of true precision.

**False negatives on known faults (8 missed on test):** These are likely events with telemetry close to station norms and no physics violation — exactly the subtle fault class that rules cannot catch and IF may miss when multivariate features look normal.

**OK (ref=...) recall (8.2%):** The model largely fails to catch message-only subtle anomalies without NLP features — expected given we deliberately exclude message text from the feature pipeline.

---

## 6. Production Considerations

| Concern | Handling |
|---|---|
| Train/serve skew | Single `features.py` shared by train and predict |
| New stations | Global baseline fallback for z-scores |
| Missing values at inference | Training medians stored in artifact |
| Alert volume | 11.5% flag rate may be high for production; tune threshold upward or add session-level gating |
| Interpretability | Layer 1 rules are fully explainable; IF features are inspectable; LightGBM provides gain-based importances |
| Drift | Retrain station baselines periodically; monitor flag rate per station |
| `station_recent_fault_rate` | Currently static at inference; production would stream-update per station |

---

## 7. What I Would Improve With More Time

1. **Per-station IF models** or quantile-based thresholds to reduce cross-station false positives and lower flag rate.
2. **Message-text features** (TF-IDF on `message`) to catch `OK (ref=...)` subtle anomalies.
3. **Raise IF threshold** or add session-level alert gating to reduce the 11.5% flag rate while maintaining recall on known faults.
4. **Streaming `station_recent_fault_rate`** updated per incoming session batch.
5. **Calibrated probabilities** for Layer 3 to make risk scores more interpretable (currently no sessions exceed 0.5 on test).
6. **Operator feedback loop** to convert dispositions into true labels and refine thresholds over time.
7. **Sequence models** (e.g., LSTM autoencoder on session event sequences) if richer temporal structure emerges in real data.

---

## 8. Conclusion

This system combines three complementary layers: deterministic physics rules for high-confidence unlabeled violations, station-relative Isolation Forest for multivariate outliers, and LightGBM session risk scoring for NOC prioritization. The architecture honestly reflects what the data supports — concurrent anomaly detection and station degradation trends, not within-session fault prediction. Partial labels are used for threshold calibration and evaluation with appropriate PU-aware caveats, while the majority of anomaly classes remain genuinely unsupervised.
