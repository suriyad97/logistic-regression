"""
detect_data_drift.py  [Evidently AI edition]
─────────────────────────────────────────────
Detects data/feature drift between the training baseline and
production inference data using Evidently AI's DataDriftPreset.

Evidently gives us:
  ─ Per-feature drift detection with the statistically appropriate test
    (KS for continuous, chi-squared for categorical, by default)
  ─ A rich HTML report viewable in AML Studio or Azure DevOps artifacts
  ─ A JSON summary that downstream scripts parse for severity grading

Inputs:
  baseline_data_path  — path to training CSV (the reference dataset)
  production_data_path — path to predictions CSV (the current dataset)

Outputs:
  drift_report.json   — machine-readable drift summary
  data_drift.html     — human-readable Evidently HTML report

Called by:
  cm-monitoring-drift.yml  →  Detect Data Distribution Drift step
  mlpipelines/monitoring/monitoring_job.py
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ── Evidently ─────────────────────────────────────────────────────
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetDriftMetric, DataDriftTable
from evidently import ColumnMapping

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Feature definitions ───────────────────────────────────────────
# Columns that contain model input features (match DataProcessor.feature_names)
NUMERICAL_FEATURES   = ["Age", "Fare", "SibSp", "Parch", "FamilySize", "IsAlone"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Title", "AgeBand", "FareBand", "Embarked"]
TARGET_COLUMN        = "Survived"
PREDICTION_COLUMN    = "prediction"

# Prediction columns written by inference_batch.py — exclude from feature drift
INFERENCE_META_COLS  = {
    "prediction", "survival_probability", "non_survival_probability",
    "prediction_label", "inference_timestamp",
}


def build_column_mapping() -> ColumnMapping:
    """Tell Evidently which columns are features vs target vs predictions."""
    return ColumnMapping(
        target=TARGET_COLUMN,
        prediction=PREDICTION_COLUMN,
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )


def load_reference(baseline_data_path: str) -> pd.DataFrame:
    """
    Load the training dataset as the Evidently reference (baseline).
    If a CSV is provided directly, use it. If a JSON stats file is provided,
    fall back to synthetic reconstruction (less accurate but usable).
    """
    p = Path(baseline_data_path)

    if p.suffix == ".csv":
        df = pd.read_csv(p)
        logger.info("Loaded reference from CSV: %d rows × %d cols", *df.shape)
        return df

    # JSON baseline stats — reconstruct approximate reference from stats
    if p.suffix == ".json":
        with open(p) as f:
            stats = json.load(f)
        feature_stats = stats.get("feature_stats", stats)
        logger.warning(
            "Reconstructing reference from JSON stats (less accurate). "
            "Prefer passing the original training CSV."
        )
        n = 500
        records = {}
        for feat, s in feature_stats.items():
            if isinstance(s, dict) and "mean" in s:
                records[feat] = np.random.normal(s["mean"], s.get("std", 1), size=n)
        return pd.DataFrame(records)

    raise ValueError(f"Unsupported baseline format: {p.suffix}. Expected .csv or .json")


def load_current(production_data_path: str) -> pd.DataFrame:
    """Load the current (production) inference data."""
    p = Path(production_data_path)

    if p.is_dir():
        candidates = list(p.glob("*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No CSV files found in {production_data_path}")
        csv_path = candidates[0]
    else:
        csv_path = p

    df = pd.read_csv(csv_path)
    logger.info("Loaded current data: %d rows from %s", len(df), csv_path)
    return df


def filter_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only columns that are model input features + known outputs."""
    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    available = [c for c in feature_cols if c in df.columns]
    return df[available]


def run_evidently_data_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    drift_share_threshold: float = 0.5,
    output_dir: Path = Path("."),
) -> dict:
    """
    Run Evidently DataDriftPreset and return a structured summary.

    drift_share_threshold: fraction of features that must drift to flag
                           overall_drift_detected = True (default: 50%)
    """
    # Filter to feature columns only
    ref_features = filter_feature_columns(reference)
    cur_features  = filter_feature_columns(current)

    # Align columns — only test features present in both datasets
    common_cols = [c for c in ref_features.columns if c in cur_features.columns]
    if not common_cols:
        raise ValueError("No common feature columns between reference and current datasets.")

    logger.info("Running Evidently drift on %d features: %s", len(common_cols), common_cols)

    ref_aligned = ref_features[common_cols].copy()
    cur_aligned  = cur_features[common_cols].copy()

    # ── Build the Evidently Report ────────────────────────────────
    report = Report(metrics=[
        DatasetDriftMetric(drift_share=drift_share_threshold),
        DataDriftTable(),
    ])
    report.run(reference_data=ref_aligned, current_data=cur_aligned)

    # ── Extract JSON result ───────────────────────────────────────
    result_dict = report.as_dict()

    # Parse Evidently output into our standard format
    dataset_drift_metric = None
    feature_drift_table  = None

    for metric_result in result_dict.get("metrics", []):
        metric_id = metric_result.get("metric", "")
        if "DatasetDriftMetric" in metric_id:
            dataset_drift_metric = metric_result.get("result", {})
        elif "DataDriftTable" in metric_id:
            feature_drift_table = metric_result.get("result", {})

    overall_drift = bool(
        dataset_drift_metric.get("dataset_drift", False)
        if dataset_drift_metric else False
    )
    drift_share  = float(dataset_drift_metric.get("share_drifted_features", 0.0) if dataset_drift_metric else 0.0)
    n_drifted    = int(dataset_drift_metric.get("number_of_drifted_features", 0) if dataset_drift_metric else 0)
    n_features   = int(dataset_drift_metric.get("number_of_features", len(common_cols)) if dataset_drift_metric else len(common_cols))

    # Per-feature detail
    features_detail = {}
    if feature_drift_table:
        for feat_name, feat_info in feature_drift_table.get("drift_by_columns", {}).items():
            features_detail[feat_name] = {
                "drift_detected":  bool(feat_info.get("drift_detected", False)),
                "drift_score":     float(feat_info.get("drift_score", 0.0)),
                "stattest_name":   feat_info.get("stattest_name", ""),
                "threshold":       float(feat_info.get("threshold", 0.05)),
                "reference_mean":  float(ref_aligned[feat_name].mean()) if feat_name in ref_aligned else None,
                "current_mean":    float(cur_aligned[feat_name].mean())  if feat_name in cur_aligned  else None,
            }

    # ── Save HTML report ──────────────────────────────────────────
    html_path = output_dir / "data_drift.html"
    report.save_html(str(html_path))
    logger.info("Evidently HTML report saved to %s", html_path)

    return {
        "timestamp":              datetime.now(timezone.utc).isoformat(),
        "overall_drift_detected": overall_drift,
        "drift_share":            drift_share,
        "n_drifted_features":     n_drifted,
        "n_features_tested":      n_features,
        "drift_share_threshold":  drift_share_threshold,
        "evidently_version":      _evidently_version(),
        "features":               features_detail,
        "html_report":            str(html_path),
    }


def _evidently_version() -> str:
    try:
        import evidently
        return evidently.__version__
    except Exception:
        return "unknown"


# ── Public API (called by monitoring_job.py) ──────────────────────
def main(
    baseline_stats_file: str,
    production_data_path: str,
    threshold: float = 0.05,
    output_file: str = "drift_report.json",
) -> dict:
    """
    Entry point compatible with the monitoring_job.py orchestrator.

    baseline_stats_file  — can be the original training CSV OR the JSON baseline
    production_data_path — folder or file with production predictions CSV
    threshold            — drift_share threshold (fraction of features)
    output_file          — path to write JSON summary
    """
    logger.info("── Evidently Data Drift Detection ──────────────────")

    reference = load_reference(baseline_stats_file)
    current   = load_current(production_data_path)

    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    drift_report = run_evidently_data_drift(
        reference=reference,
        current=current,
        drift_share_threshold=threshold,
        output_dir=output_dir,
    )

    with open(output_file, "w") as f:
        json.dump(drift_report, f, indent=2)

    logger.info("Data drift report written to %s", output_file)
    logger.info(
        "Overall drift: %s  |  Drifted features: %d / %d  (%.0f%%)",
        drift_report["overall_drift_detected"],
        drift_report["n_drifted_features"],
        drift_report["n_features_tested"],
        drift_report["drift_share"] * 100,
    )

    return drift_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evidently AI — Data Drift Detection")
    parser.add_argument("--baseline-stats",    required=True,
                        help="Training CSV (preferred) or JSON baseline stats")
    parser.add_argument("--production-data",   required=True,
                        help="Production predictions CSV or folder")
    parser.add_argument("--threshold",         type=float, default=0.5,
                        help="Fraction of features that must drift to flag overall drift")
    parser.add_argument("--output",            default="drift_report.json")
    args = parser.parse_args()

    main(
        baseline_stats_file=args.baseline_stats,
        production_data_path=args.production_data,
        threshold=args.threshold,
        output_file=args.output,
    )
