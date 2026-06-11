# AI Tool Usage Documentation

## Tools Used

| Tool | Role |
|---|---|
| **Cursor (Claude)** | Primary coding assistant for architecture design, implementation, debugging, and report drafting |
| **Cursor Agent** | Multi-file project scaffolding, running training/inference, iterative spec review |

## How AI Was Used

### Architecture and specification (high value)
- Reviewed the assignment requirements and EDA findings to propose a three-layer architecture.
- Identified design issues: Layer 3 temporal prediction overclaim, global voltage bands conflicting with station normalization, `contamination=0.05` as an implicit threshold, and positive-unlabeled evaluation pitfalls.
- Produced a complete file/function/feature specification before any code was written.

### Implementation (high value)
- Generated the full project structure: `config.py`, `preprocessing.py`, `rules.py`, `features.py`, `model.py`, `pipeline.py`, `train.py`, `predict.py`.
- Ensured `features.py` is shared between training and inference to prevent train/serve skew.
- Created the EDA notebook skeleton with analysis cells aligned to the report narrative.

### Documentation (high value)
- Drafted `README.md`, `REPORT.md`, and this file based on actual training metrics.
- Structured the report around honest findings (e.g., no within-session prediction signal, PU-aware precision caveats).

### Debugging and validation (moderate value)
- Diagnosed environment issues (externally-managed Python → venv setup).
- Ran end-to-end training and inference to verify the pipeline produces artifacts and predictions.

## Where AI Struggled

1. **Threshold tuning tradeoffs:** AI suggested reasonable defaults but could not know the optimal flag rate without running the full pipeline. The 11.5% flag rate on test required empirical evaluation and honest reporting rather than architectural fixes.
2. **Layer 3 performance expectations:** Initial framing implied predictive power that EDA contradicted. AI correctly flagged this after seeing the uniform fault-position distribution, but the first-pass architecture needed human review to reframe.
3. **Feature importance for Isolation Forest:** sklearn's IF does not expose native importances. AI implemented permutation-based approximation, which is directionally useful but not as rigorous as SHAP or dedicated explainability tooling.
4. **Subtle anomaly class (`OK (ref=...)`):** Without NLP on message text, AI could predict (correctly) that recall on this set would be low — confirmed at 8.2% on test.

## How AI-Generated Code Was Validated

| Check | Method |
|---|---|
| **Spec compliance** | Each file reviewed against the agreed specification before proceeding to the next |
| **End-to-end training** | `python scripts/train.py` completed successfully; artifact saved to `models/artifact.joblib` |
| **Inference** | `python predict.py --input data/charging_logs.csv --output outputs/predictions.csv` produced expected output columns and summary stats |
| **Metric sanity** | Layer 1 catches 100% of physics violations on test; IF recall ~91% on known faults; `error_code_binary` ranked last in IF importance (not dominating) |
| **No train/serve skew** | Verified `features.py` is the single entry point called by both `train.py` and `pipeline.py`/`predict.py` |
| **Split integrity** | Confirmed session-level temporal split produces disjoint train/val/test session sets |
| **Label construction** | Verified `OK (ref=...)` rows excluded from session labels; known faults only |

## Engineering Judgment Applied

- Accepted AI's pushback on Layer 3 "prediction" framing and reframed as concurrent risk scoring.
- Removed global voltage/temperature hard bands from Layer 1 per AI recommendation.
- Kept `error_code_binary` in IF features per original spec but monitored importance (confirmed low — no change needed for v1).
- Renamed `session_physics_violation_fraction` → `session_concurrent_violation_fraction` for report honesty.
- Reported low apparent precision with explicit PU caveat rather than presenting misleading headline metrics.

## Ownership Statement

All architectural decisions, spec resolutions, and evaluation interpretations were reviewed and confirmed before implementation. AI accelerated scaffolding and boilerplate; design tradeoffs, honest limitation framing, and final metric interpretation reflect deliberate engineering judgment.
