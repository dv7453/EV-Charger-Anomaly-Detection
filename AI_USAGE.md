# AI Tool Usage Documentation

## Tools Used

| Tool | Role |
|---|---|
| **Cursor (Claude)** | Coding assistant for design discussion, implementation scaffolding, debugging, and documentation drafts |

## How AI Was Used

After reviewing the assignment and completing EDA, I concluded the problem needed a layered approach — physics rules for high-confidence violations, an unsupervised model for multivariate outliers, and session-level scoring for NOC prioritization. Working with AI through iterative review helped me think through tradeoffs between model choices (e.g., Isolation Forest vs. one-class SVM, global vs. station-relative thresholds) and nail down a concrete spec before writing code.

Several architectural decisions were judgment calls I made after options came up in that back-and-forth: removing global voltage/temperature bands from Layer 1 in favor of station z-scores in Layer 2, reframing Layer 3 as concurrent session risk scoring rather than within-session prediction, and keeping `error_code_binary` in IF features while monitoring whether it dominated. I also renamed `session_concurrent_violation_fraction` and chose to report apparent precision as a lower bound given the positive-unlabeled setting.

AI assisted with implementation scaffolding and boilerplate across the project modules (`preprocessing.py`, `features.py`, `rules.py`, `model.py`, `pipeline.py`, `train.py`, `predict.py`). I reviewed each file against the spec before moving on, and kept `features.py` as the single shared entry point for training and inference to avoid train/serve skew.

For documentation, AI helped with first drafts of `README.md` and `REPORT.md`. I ran the pipeline myself, read the metrics, and decided what needed honest framing in the report — including the 11.5% flag rate, modest Layer 3 AUC, and low recall on `OK (ref=...)` rows.

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
