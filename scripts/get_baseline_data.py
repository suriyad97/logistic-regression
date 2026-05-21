"""
get_baseline_data.py
─────────────────────
Pulls baseline statistics from the DEV workspace training run
and writes them to a local JSON file consumed by drift detectors.

Baseline stats include:
  - Per-feature mean/std/min/max (from training data)
  - Model training metrics (accuracy, roc_auc …)
  - Feature importance (model coefficients)

Called by: cm-monitoring-drift.yml → Retrieve Baseline Statistics step
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def get_ml_client(subscription_id: str, resource_group: str, workspace_name: str) -> MLClient:
    credential = DefaultAzureCredential()
    return MLClient(credential, subscription_id, resource_group, workspace_name)


def pull_from_local_model(model_dir: str = "model") -> dict:
    """
    Fallback: compute baseline stats directly from local model artifacts.
    Used when AML workspace connection is not available (e.g., local dev).
    """
    model_path = Path(model_dir)
    metrics_file = model_path / "metrics.json"

    if not metrics_file.exists():
        raise FileNotFoundError(f"metrics.json not found at {metrics_file}")

    with open(metrics_file) as f:
        saved_metrics = json.load(f)

    # Also load the training data for feature stats
    data_file = Path("data/titanic.csv")
    if not data_file.exists():
        data_file = Path("../data/titanic.csv")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from data_processing import DataProcessor

    processor = DataProcessor()
    processor.load_preprocessor(str(model_path / "preprocessor.pkl"))

    df = pd.read_csv(data_file)
    df = processor.handle_missing_values(df)
    df = processor.feature_engineering(df, is_training=False)
    df = processor.encode_categorical_features(df, is_training=False)
    X = df[processor.feature_names]

    feature_stats = {}
    for col in X.columns:
        feature_stats[col] = {
            "mean":   float(X[col].mean()),
            "std":    float(X[col].std()),
            "min":    float(X[col].min()),
            "max":    float(X[col].max()),
            "median": float(X[col].median()),
        }

    # Feature importance from model coefficients
    model = joblib.load(model_path / "model.pkl")
    feature_importance = {
        feat: float(abs(coef))
        for feat, coef in zip(processor.feature_names, model.coef_[0])
    }

    return {
        "feature_stats": feature_stats,
        "feature_importance": feature_importance,
        "training_metrics": saved_metrics.get("training_metrics", {}),
        "test_metrics":     saved_metrics.get("test_metrics", {}),
        "model_params":     saved_metrics.get("model_params", {}),
        "data_info":        saved_metrics.get("data_info", {}),
        "feature_names":    processor.feature_names,
    }


def pull_from_aml(ml_client: MLClient, model_name: str = "titanic-logistic-regression") -> dict:
    """
    Pull the latest model's metrics from AML.
    Falls back to local if AML call fails.
    """
    try:
        models = list(ml_client.models.list(name=model_name))
        if not models:
            raise ValueError(f"No models found with name '{model_name}'")

        latest = sorted(models, key=lambda m: int(m.version))[-1]
        logger.info("Found model: %s @ version %s", latest.name, latest.version)

        # Tags contain training run id — we can pull MLflow metrics from that run
        tags = latest.tags or {}
        return {
            "model_name":    latest.name,
            "model_version": latest.version,
            "tags":          tags,
            "note": "Full baseline stats require local model artifacts or MLflow run details",
        }
    except Exception as exc:
        logger.warning("AML model lookup failed (%s) — falling back to local", exc)
        return pull_from_local_model()


def main():
    parser = argparse.ArgumentParser(description="Retrieve baseline statistics for drift detection")
    parser.add_argument("--resource-group",  default=None)
    parser.add_argument("--workspace-name",  default=None)
    parser.add_argument("--subscription-id", default=None)
    parser.add_argument("--model-name",      default="titanic-logistic-regression")
    parser.add_argument("--model-dir",       default="model",
                        help="Local model directory (fallback if AML not available)")
    parser.add_argument("--output",          default="baseline_stats.json")
    args = parser.parse_args()

    import os
    subscription_id = args.subscription_id or os.environ.get("AZURE_SUBSCRIPTION_ID")

    if subscription_id and args.resource_group and args.workspace_name:
        logger.info("Pulling baseline from AML workspace %s", args.workspace_name)
        ml_client = get_ml_client(subscription_id, args.resource_group, args.workspace_name)
        stats = pull_from_aml(ml_client, args.model_name)
        # If AML only returned tags, also merge local stats for full feature distribution
        if "feature_stats" not in stats:
            local_stats = pull_from_local_model(args.model_dir)
            stats.update(local_stats)
    else:
        logger.info("No AML credentials — computing baseline from local artifacts")
        stats = pull_from_local_model(args.model_dir)

    with open(args.output, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Baseline stats written to %s", args.output)


if __name__ == "__main__":
    main()
