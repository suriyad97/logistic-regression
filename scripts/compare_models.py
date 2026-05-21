"""
compare_models.py
─────────────────
Champion vs Challenger evaluation used in the QA pipeline.

Logic:
  1. Load challenger model (from DEV)
  2. Load champion model (from QA/PROD) — or skip if first-ever model
  3. Evaluate BOTH on a held-out QA test set
  4. Write comparison_results.json
  5. Exit 0 if challenger wins (or first model), else exit 1

Called by: qa-champion-challenger.yml  →  Compare Model Performance step.
The manual approval gate that follows reads comparison_results.json.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_processing import DataProcessor  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# How much better challenger must be to dethrone the champion
IMPROVEMENT_THRESHOLD = 0.02   # 2 percentage points (passed as --threshold)


# ── helpers ───────────────────────────────────────────────────────
def load_artifacts(model_dir: str):
    """Return (model, DataProcessor) from a model directory."""
    p = Path(model_dir)
    model = joblib.load(p / "model.pkl")
    proc = DataProcessor()
    proc.load_preprocessor(str(p / "preprocessor.pkl"))
    return model, proc


def evaluate(model, processor: DataProcessor, data_path: str) -> dict:
    """Compute classification metrics for a model on a dataset."""
    df = pd.read_csv(data_path)
    df_clean = processor.handle_missing_values(df)
    df_clean = processor.feature_engineering(df_clean, is_training=False)
    df_clean = processor.encode_categorical_features(df_clean, is_training=False)

    y_true = df_clean["Survived"]
    X = df_clean[processor.feature_names]
    X_scaled = processor.scale_features(X, is_training=False)

    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score":  float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_true, y_proba)),
        "support":   int(len(y_true)),
    }


# ── main ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Champion vs Challenger comparison")
    parser.add_argument("--challenger-model", required=True, help="Path to challenger model dir")
    parser.add_argument("--champion-model", default="none",  help="Path to champion model dir (or 'none')")
    parser.add_argument("--test-data",       required=True, help="Path to QA test CSV")
    parser.add_argument("--threshold",       type=float, default=IMPROVEMENT_THRESHOLD,
                        help="Min improvement required for challenger to win")
    parser.add_argument("--output",          default="comparison_results.json")
    args = parser.parse_args()

    # ── Challenger ───────────────────────────────────────────────
    logger.info("Evaluating CHALLENGER from %s", args.challenger_model)
    challenger_model, challenger_proc = load_artifacts(args.challenger_model)
    challenger_metrics = evaluate(challenger_model, challenger_proc, args.test_data)
    logger.info("Challenger metrics: %s", challenger_metrics)

    # ── Champion (may not exist yet) ─────────────────────────────
    is_first_model = args.champion_model.lower() == "none"
    champion_metrics = None

    if not is_first_model:
        logger.info("Evaluating CHAMPION from %s", args.champion_model)
        champion_model, champion_proc = load_artifacts(args.champion_model)
        champion_metrics = evaluate(champion_model, champion_proc, args.test_data)
        logger.info("Champion metrics: %s", champion_metrics)

    # ── Decision ─────────────────────────────────────────────────
    if is_first_model:
        # No existing champion → challenger automatically becomes champion
        challenger_wins = True
        promotion_reason = "First model deployment — no champion exists"
        improvement = None
        logger.info("✅ No champion exists. Challenger will be promoted.")
    else:
        improvement = challenger_metrics["roc_auc"] - champion_metrics["roc_auc"]
        challenger_wins = improvement >= args.threshold
        promotion_reason = (
            f"Challenger ROC-AUC improved by {improvement:.4f} (threshold: {args.threshold})"
            if challenger_wins
            else f"Challenger did NOT improve by threshold ({improvement:.4f} < {args.threshold})"
        )
        logger.info(
            "%s Challenger wins: %s | %s",
            "✅" if challenger_wins else "❌",
            challenger_wins,
            promotion_reason,
        )

    # ── Write results ─────────────────────────────────────────────
    result = {
        "challenger_wins":    challenger_wins,
        "is_first_model":     is_first_model,
        "promotion_reason":   promotion_reason,
        "improvement_threshold": args.threshold,
        "roc_auc_improvement": improvement,
        "challenger_metrics": challenger_metrics,
        "champion_metrics":   champion_metrics,
        "recommendation":     "PROMOTE challenger to champion" if challenger_wins else "KEEP existing champion",
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Comparison report written to %s", args.output)

    # Pipeline step outcome: exit 1 if challenger does not win
    # (the ManualValidation gate will then stop the Promote_to_Prod stage)
    if not challenger_wins:
        logger.warning("Challenger did not win. Blocking promotion.")
        sys.exit(1)

    logger.info("Challenger wins! Ready for manual approval gate.")


if __name__ == "__main__":
    main()
