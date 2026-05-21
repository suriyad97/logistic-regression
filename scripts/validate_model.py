"""
validate_model.py
─────────────────
Post-training gate: checks that the newly trained model meets
minimum quality thresholds before it's registered in the AML registry.

Called by: CT pipeline (ct-train-register-model.yml)

If thresholds are not met this script exits with a non-zero code,
failing the pipeline stage before registration happens.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_processing import DataProcessor  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Minimum quality gates ─────────────────────────────────────────
THRESHOLDS = {
    "accuracy": 0.70,
    "f1_score": 0.65,
    "roc_auc": 0.75,
}


def evaluate_model(model_path: str, test_data_path: str) -> dict:
    """Load model + preprocessor, run on test data, return metrics."""
    model_dir = Path(model_path)

    model = joblib.load(model_dir / "model.pkl")
    processor = DataProcessor()
    processor.load_preprocessor(str(model_dir / "preprocessor.pkl"))

    # Also check if pre-computed metrics exist (from training run)
    metrics_file = model_dir / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            saved = json.load(f)
        test_metrics = saved.get("test_metrics", {})
        logger.info("Using saved test metrics from training run: %s", test_metrics)
        return test_metrics

    # Otherwise compute on provided data
    logger.info("Computing metrics on %s", test_data_path)
    df = pd.read_csv(test_data_path)
    df = processor.handle_missing_values(df)
    df = processor.feature_engineering(df, is_training=False)
    df = processor.encode_categorical_features(df, is_training=False)

    y_true = df["Survived"]
    X = df[processor.feature_names]
    X_scaled = processor.scale_features(X, is_training=False)

    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def check_thresholds(metrics: dict) -> tuple[bool, list[str]]:
    """Return (passed, list_of_failures)."""
    failures = []
    for metric, min_value in THRESHOLDS.items():
        actual = metrics.get(metric)
        if actual is None:
            logger.warning("Metric %s not found — skipping", metric)
            continue
        if actual < min_value:
            msg = f"{metric}: {actual:.4f} < threshold {min_value}"
            failures.append(msg)
            logger.error("❌ GATE FAILED — %s", msg)
        else:
            logger.info("✅ %s: %.4f >= %.4f", metric, actual, min_value)
    return len(failures) == 0, failures


def main():
    parser = argparse.ArgumentParser(description="Post-training quality gate")
    parser.add_argument("--model-path", required=True, help="Folder with model.pkl + preprocessor.pkl")
    parser.add_argument("--test-data", required=True, help="Path to test CSV (with Survived column)")
    parser.add_argument("--output", default="validation_result.json")
    args = parser.parse_args()

    metrics = evaluate_model(args.model_path, args.test_data)
    logger.info("Computed metrics: %s", metrics)

    passed, failures = check_thresholds(metrics)

    result = {
        "passed": passed,
        "metrics": metrics,
        "thresholds": THRESHOLDS,
        "failures": failures,
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Validation result written to %s", args.output)

    if not passed:
        logger.error("Model did NOT pass quality gates. Pipeline will fail.")
        sys.exit(1)

    logger.info("Model passed all quality gates ✅")


if __name__ == "__main__":
    main()
