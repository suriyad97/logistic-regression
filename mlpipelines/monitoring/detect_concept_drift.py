"""
detect_concept_drift.py  [Evidently AI edition]
────────────────────────────────────────────────
Detects concept drift and prediction drift using Evidently AI.

Three complementary analyses are run:

1. ClassificationPreset  (requires ground truth)
   Measures actual model performance on labelled production data.
   If accuracy drops below the training baseline by > threshold,
   concept drift is flagged.

2. TargetDriftPreset  (ground truth optional)
   Measures whether the distribution of the target/prediction column
   has shifted — useful even without ground truth labels.

3. DataQualityPreset  (always runs)
   Checks for missing values, schema violations, and data anomalies
   in the production feature set.

Outputs:
  concept_drift_report.json  — machine-readable summary
  concept_drift.html         — full Evidently HTML report

Called by:
  cm-monitoring-drift.yml  →  Detect Concept Drift step
  mlpipelines/monitoring/monitoring_job.py
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Evidently ─────────────────────────────────────────────────────
from evidently.report import Report
from evidently.metric_preset import (
    ClassificationPreset,
    TargetDriftPreset,
    DataQualityPreset,
)
from evidently.metrics import (
    ClassificationQualityMetric,
    ColumnDriftMetric,
)
from evidently import ColumnMapping

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Column definitions (must match DataProcessor output) ──────────
NUMERICAL_FEATURES   = ["Age", "Fare", "SibSp", "Parch", "FamilySize", "IsAlone"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Title", "AgeBand", "FareBand", "Embarked"]
TARGET_COLUMN        = "Survived"
PREDICTION_COLUMN    = "prediction"
PRED_PROBA_COLUMN    = "survival_probability"

# Baseline accuracy from training (from model/metrics.json)
BASELINE_ACCURACY    = 0.80


def build_column_mapping() -> ColumnMapping:
    return ColumnMapping(
        target=TARGET_COLUMN,
        prediction=PREDICTION_COLUMN,
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )


# ── Loaders ───────────────────────────────────────────────────────

def load_predictions(predictions_path: str) -> pd.DataFrame:
    p = Path(predictions_path)
    csv = (p / "predictions.csv") if p.is_dir() else p
    df = pd.read_csv(csv)
    logger.info("Loaded predictions: %d rows from %s", len(df), csv)
    return df


def load_reference_from_baseline(baseline_file: str) -> pd.DataFrame:
    """
    Load the original training dataset as the Evidently reference.
    Accepts either the raw CSV or a JSON stats file.
    """
    p = Path(baseline_file)
    if p.suffix == ".csv":
        return pd.read_csv(p)

    # JSON fallback — reconstruct minimal reference
    with open(p) as f:
        stats = json.load(f)

    n = 500
    records = {}
    feature_stats = stats.get("feature_stats", {})
    for feat, s in feature_stats.items():
        if isinstance(s, dict) and "mean" in s:
            records[feat] = np.random.normal(s["mean"], s.get("std", 1), n)

    baseline_pos_rate = stats.get("target_positive_rate", 0.38)
    records[TARGET_COLUMN] = np.random.binomial(1, baseline_pos_rate, n)

    df = pd.DataFrame(records)
    logger.warning("Reference reconstructed from JSON stats (n=%d). Prefer original CSV.", n)
    return df


def load_ground_truth(ground_truth_path: str) -> Optional[pd.DataFrame]:
    """Load matched feedback data (predictions + actual_label)."""
    if not ground_truth_path:
        return None
    p = Path(ground_truth_path)
    candidates = list(p.glob("matched_ground_truth.csv")) if p.is_dir() else [p]
    if not candidates or not candidates[0].exists():
        logger.warning("No ground truth file found at %s", ground_truth_path)
        return None
    df = pd.read_csv(candidates[0])
    logger.info("Loaded ground truth: %d rows", len(df))
    return df


# ── Analysis 1: Classification Performance (requires ground truth) ─

def run_classification_analysis(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    baseline_accuracy: float,
    concept_drift_threshold: float,
    output_dir: Path,
) -> dict:
    """
    Use Evidently ClassificationPreset to compare production performance
    against the training baseline.
    """
    column_mapping = build_column_mapping()

    # Ensure required columns exist
    for col in [TARGET_COLUMN, PREDICTION_COLUMN]:
        if col not in reference_df.columns:
            reference_df[col] = 0
        if col not in current_df.columns:
            logger.warning("Column '%s' missing from current data — skipping classification analysis", col)
            return {"skipped": True, "reason": f"Missing column: {col}"}

    report = Report(metrics=[
        ClassificationPreset(),
        ClassificationQualityMetric(),
    ])

    try:
        report.run(
            reference_data=reference_df,
            current_data=current_df,
            column_mapping=column_mapping,
        )
    except Exception as e:
        logger.warning("ClassificationPreset failed: %s", e)
        return {"skipped": True, "reason": str(e)}

    result_dict = report.as_dict()

    # Extract current accuracy from result
    current_accuracy = None
    for m in result_dict.get("metrics", []):
        if "ClassificationQualityMetric" in m.get("metric", ""):
            current_accuracy = m.get("result", {}).get("current", {}).get("accuracy")
            break

    if current_accuracy is None:
        return {"skipped": True, "reason": "Could not extract accuracy from Evidently result"}

    performance_drop = baseline_accuracy - float(current_accuracy)
    concept_drift_detected = performance_drop > concept_drift_threshold

    html_path = output_dir / "classification_report.html"
    report.save_html(str(html_path))

    return {
        "concept_drift_detected": bool(concept_drift_detected),
        "baseline_accuracy":      baseline_accuracy,
        "current_accuracy":       float(current_accuracy),
        "performance_drop":       round(float(performance_drop), 4),
        "threshold":              concept_drift_threshold,
        "html_report":            str(html_path),
        "skipped":                False,
    }


# ── Analysis 2: Target / Prediction Drift ─────────────────────────

def run_target_drift_analysis(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """
    Use Evidently TargetDriftPreset to detect shifts in the prediction
    (or target) distribution — works even without ground truth.
    """
    column_mapping = build_column_mapping()

    # Prefer using the raw prediction column
    pred_col = PREDICTION_COLUMN if PREDICTION_COLUMN in current_df.columns else TARGET_COLUMN
    ref_col   = pred_col if pred_col in reference_df.columns else None

    if not ref_col:
        return {"skipped": True, "reason": "Neither prediction nor target column in reference"}

    report = Report(metrics=[
        TargetDriftPreset(),
        ColumnDriftMetric(column_name=pred_col),
    ])

    try:
        report.run(
            reference_data=reference_df,
            current_data=current_df,
            column_mapping=column_mapping,
        )
    except Exception as e:
        logger.warning("TargetDriftPreset failed: %s", e)
        return {"skipped": True, "reason": str(e)}

    result_dict = report.as_dict()

    # Extract column drift for prediction column
    col_drift = {}
    for m in result_dict.get("metrics", []):
        if "ColumnDriftMetric" in m.get("metric", ""):
            res = m.get("result", {})
            col_drift = {
                "drift_detected": bool(res.get("drift_detected", False)),
                "drift_score":    float(res.get("drift_score", 0.0)),
                "stattest_name":  res.get("stattest_name", ""),
            }

    html_path = output_dir / "target_drift.html"
    report.save_html(str(html_path))

    return {
        "prediction_drift_detected": col_drift.get("drift_detected", False),
        "prediction_drift_score":    col_drift.get("drift_score", 0.0),
        "stattest":                  col_drift.get("stattest_name", ""),
        "html_report":               str(html_path),
        "skipped":                   False,
    }


# ── Analysis 3: Data Quality ──────────────────────────────────────

def run_data_quality_analysis(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """Detect schema violations and missing value anomalies in production."""
    report = Report(metrics=[DataQualityPreset()])

    try:
        report.run(reference_data=reference_df, current_data=current_df)
    except Exception as e:
        return {"skipped": True, "reason": str(e)}

    html_path = output_dir / "data_quality.html"
    report.save_html(str(html_path))

    return {"html_report": str(html_path), "skipped": False}


# ── Public API ────────────────────────────────────────────────────

def main(
    baseline_stats_file: str,
    predictions_path: str,
    ground_truth_path: Optional[str],
    threshold: float = 0.03,
    output_file: str = "concept_drift_report.json",
) -> dict:
    """
    Orchestrate all three Evidently analyses and write the combined report.

    Parameters
    ----------
    baseline_stats_file : str
        Original training CSV (preferred) or JSON baseline stats
    predictions_path : str
        Folder or CSV with production predictions
    ground_truth_path : str | None
        Folder with matched_ground_truth.csv (from FeedbackStore)
    threshold : float
        Accuracy drop threshold to flag concept drift (default 3%)
    output_file : str
        Path to write consolidated JSON report
    """
    logger.info("── Evidently Concept Drift Detection ───────────────")

    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    reference_df  = load_reference_from_baseline(baseline_stats_file)
    predictions_df = load_predictions(predictions_path)
    ground_truth_df = load_ground_truth(ground_truth_path)

    # Prepare current_df: merge predictions with ground truth if available
    if ground_truth_df is not None and "actual_label" in ground_truth_df.columns:
        # Rename for Evidently compatibility
        current_df = ground_truth_df.rename(columns={"actual_label": TARGET_COLUMN})
        has_ground_truth = True
        logger.info("Ground truth available — running full classification analysis")
    else:
        current_df = predictions_df.copy()
        has_ground_truth = False
        logger.info("No ground truth — running prediction drift analysis only")

    # ── Run analyses ──────────────────────────────────────────────
    classification_result = {}
    if has_ground_truth:
        classification_result = run_classification_analysis(
            reference_df=reference_df,
            current_df=current_df,
            baseline_accuracy=BASELINE_ACCURACY,
            concept_drift_threshold=threshold,
            output_dir=output_dir,
        )

    target_drift_result  = run_target_drift_analysis(reference_df, current_df, output_dir)
    data_quality_result  = run_data_quality_analysis(reference_df, current_df, output_dir)

    # ── Consolidate ───────────────────────────────────────────────
    concept_drift_detected = (
        classification_result.get("concept_drift_detected", False)
        if not classification_result.get("skipped", True)
        else False
    )
    prediction_drift_detected = target_drift_result.get("prediction_drift_detected", False)

    report = {
        "timestamp":                  datetime.now(timezone.utc).isoformat(),
        "concept_drift_detected":     concept_drift_detected,
        "prediction_drift_detected":  prediction_drift_detected,
        "has_ground_truth":           has_ground_truth,
        "threshold":                  threshold,
        "evidently_version":          _evidently_version(),
        "classification_analysis":    classification_result,
        "target_drift_analysis":      target_drift_result,
        "data_quality_analysis":      data_quality_result,
        # Top-level convenience fields for analyze_drift_results.py
        "current_accuracy":           classification_result.get("current_accuracy"),
        "baseline_accuracy":          BASELINE_ACCURACY,
        "performance_drop":           classification_result.get("performance_drop"),
        "samples_evaluated":          len(current_df),
        "reason": (
            "Concept drift detected via Evidently ClassificationPreset"
            if concept_drift_detected
            else "No ground truth" if not has_ground_truth
            else "Performance within acceptable range"
        ),
    }

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Concept drift report written to %s", output_file)
    logger.info(
        "concept_drift=%s  prediction_drift=%s  ground_truth=%s",
        concept_drift_detected, prediction_drift_detected, has_ground_truth,
    )

    return report


def _evidently_version() -> str:
    try:
        import evidently
        return evidently.__version__
    except Exception:
        return "unknown"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evidently AI — Concept Drift Detection")
    parser.add_argument("--baseline-stats",    required=True)
    parser.add_argument("--predictions-path",  required=True)
    parser.add_argument("--ground-truth-path", default=None)
    parser.add_argument("--threshold",         type=float, default=0.03)
    parser.add_argument("--output",            default="concept_drift_report.json")
    args = parser.parse_args()

    main(
        baseline_stats_file=args.baseline_stats,
        predictions_path=args.predictions_path,
        ground_truth_path=args.ground_truth_path,
        threshold=args.threshold,
        output_file=args.output,
    )
