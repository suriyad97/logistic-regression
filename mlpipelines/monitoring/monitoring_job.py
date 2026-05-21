"""
monitoring_job.py
──────────────────
Orchestrates data drift + concept drift detection as a single
Azure ML job. Results are logged to MLflow so the team can
track drift trends across pipeline runs in AML Studio.

Called by: mlpipelines/monitoring/monitoring_job.yml
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ── Optional MLflow ───────────────────────────────────────────────
try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# Import the existing drift detectors
sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_data_drift    import main as run_data_drift    # noqa: E402
from detect_concept_drift import main as run_concept_drift  # noqa: E402


def compute_prediction_drift(predictions_df: pd.DataFrame, baseline_path: str) -> dict:
    """
    Prediction drift = change in predicted class distribution over time.
    If the model was predicting 38% survival at training time but now
    predicts 55%, that is a signal worth flagging (even without ground truth).
    """
    with open(baseline_path) as f:
        baseline = json.load(f)

    baseline_pos_rate = baseline.get("target_positive_rate", 0.38)  # Titanic baseline ~38%
    current_pos_rate  = float((predictions_df["prediction"] == 1).mean())
    shift = abs(current_pos_rate - baseline_pos_rate)

    return {
        "baseline_survival_rate": baseline_pos_rate,
        "current_survival_rate":  current_pos_rate,
        "absolute_shift":         round(shift, 4),
        "prediction_drift_detected": shift > 0.05,   # 5pp shift threshold
    }


def log_metrics_to_mlflow(
    data_drift_report: dict,
    concept_drift_report: dict,
    pred_drift_report: dict,
) -> None:
    """Log all drift metrics to the active MLflow run."""
    if not HAS_MLFLOW:
        return

    # Data drift — number of drifted features
    features = data_drift_report.get("features", {})
    n_drifted = sum(
        1 for v in features.values()
        if v.get("ks_test", {}).get("drift_detected", False)
    )
    mlflow.log_metric("data_drift_features_count", n_drifted)
    mlflow.log_metric("data_drift_detected", int(data_drift_report.get("overall_drift_detected", False)))

    # Concept drift
    mlflow.log_metric("concept_drift_detected", int(concept_drift_report.get("concept_drift_detected", False)))
    if "current_accuracy" in concept_drift_report:
        mlflow.log_metric("production_accuracy", concept_drift_report["current_accuracy"])
    if "performance_drop" in concept_drift_report and concept_drift_report["performance_drop"] is not None:
        mlflow.log_metric("performance_drop", concept_drift_report["performance_drop"])

    # Prediction drift
    mlflow.log_metric("prediction_drift_detected", int(pred_drift_report.get("prediction_drift_detected", False)))
    mlflow.log_metric("prediction_rate_shift", pred_drift_report.get("absolute_shift", 0))


def main():
    parser = argparse.ArgumentParser(description="AML Monitoring Job — Drift Detection")
    parser.add_argument("--predictions-path",    required=True)
    parser.add_argument("--ground-truth-path",   default=None)
    parser.add_argument("--baseline-model-path", default=None)
    parser.add_argument("--output-path",         required=True)
    args = parser.parse_args()

    out = Path(args.output_path)
    out.mkdir(parents=True, exist_ok=True)

    # ── Start MLflow run ──────────────────────────────────────────
    if HAS_MLFLOW:
        mlflow.set_experiment("titanic-monitoring")
        mlflow.start_run(run_name=f"monitoring-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}")

    try:
        # ── 1. Generate baseline stats from model if needed ───────
        baseline_stats_file = str(out / "baseline_stats.json")
        if args.baseline_model_path and Path(args.baseline_model_path).exists():
            # Extract baseline from model artifacts
            from get_baseline_data_inline import pull_from_model_dir
            stats = pull_from_model_dir(args.baseline_model_path)
            with open(baseline_stats_file, "w") as f:
                json.dump(stats, f, indent=2)
        else:
            # Look for a pre-generated baseline file in the predictions folder
            candidate = Path(args.predictions_path).parent / "baseline_stats.json"
            if candidate.exists():
                baseline_stats_file = str(candidate)
            else:
                logger.warning("No baseline stats found — drift metrics may be incomplete")
                with open(baseline_stats_file, "w") as f:
                    json.dump({}, f)

        # ── 2. Data drift ─────────────────────────────────────────
        data_drift_out = str(out / "data_drift_report.json")
        logger.info("Running Evidently data drift detection…")
        data_drift_report = run_data_drift(
            baseline_stats_file=baseline_stats_file,
            production_data_path=args.predictions_path,
            threshold=float(os.environ.get("DATA_DRIFT_THRESHOLD", "0.5")),  # 50% of features
            output_file=data_drift_out,
        )

        # ── 3. Concept drift ──────────────────────────────────────
        concept_drift_out = str(out / "concept_drift_report.json")
        gt_path = args.ground_truth_path  # None is valid — Evidently handles it
        logger.info("Running Evidently concept drift detection…")
        concept_drift_report = run_concept_drift(
            baseline_stats_file=baseline_stats_file,
            predictions_path=args.predictions_path,
            ground_truth_path=gt_path,
            threshold=float(os.environ.get("CONCEPT_DRIFT_THRESHOLD", "0.03")),
            output_file=concept_drift_out,
        )

        # ── 4. Prediction drift ───────────────────────────────────
        preds_csv = Path(args.predictions_path) / "predictions.csv"
        pred_drift_report = {}
        if preds_csv.exists() and Path(baseline_stats_file).stat().st_size > 10:
            preds_df = pd.read_csv(preds_csv)
            pred_drift_report = compute_prediction_drift(preds_df, baseline_stats_file)
            with open(out / "prediction_drift_report.json", "w") as f:
                json.dump(pred_drift_report, f, indent=2)
            logger.info("Prediction drift: %s", pred_drift_report)

        # ── 5. Log to MLflow ──────────────────────────────────────
        log_metrics_to_mlflow(data_drift_report, concept_drift_report, pred_drift_report)

        # ── 6. Summary report ─────────────────────────────────────
        any_drift = (
            data_drift_report.get("overall_drift_detected", False)
            or concept_drift_report.get("concept_drift_detected", False)
            or pred_drift_report.get("prediction_drift_detected", False)
        )
        summary = {
            "monitoring_timestamp":    datetime.now(timezone.utc).isoformat(),
            "any_drift_detected":      any_drift,
            "data_drift_detected":     data_drift_report.get("overall_drift_detected", False),
            "concept_drift_detected":  concept_drift_report.get("concept_drift_detected", False),
            "prediction_drift_detected": pred_drift_report.get("prediction_drift_detected", False),
            "reports": {
                "data_drift":      data_drift_out,
                "concept_drift":   concept_drift_out,
                "prediction_drift": str(out / "prediction_drift_report.json"),
            },
        }
        with open(out / "monitoring_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("Monitoring complete — any_drift=%s", any_drift)

        if HAS_MLFLOW:
            mlflow.log_artifact(str(out))
            mlflow.end_run()

        # Exit non-zero if drift detected (triggers CM alert stage)
        if any_drift:
            logger.warning("Drift detected — exiting with code 1 to trigger alerts")
            sys.exit(1)

    except Exception as exc:
        logger.error("Monitoring job failed: %s", exc, exc_info=True)
        if HAS_MLFLOW:
            mlflow.end_run(status="FAILED")
        raise


if __name__ == "__main__":
    main()
