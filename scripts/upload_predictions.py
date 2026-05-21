"""
upload_predictions.py
─────────────────────
Uploads inference predictions as a versioned AML Data Asset so the
monitoring pipeline can compare production feature distributions
against the training baseline.

Called by: cd-inference-pipeline.yml → Log Predictions for Monitoring step

This keeps a complete, versioned history of every prediction batch —
critical for computing concept drift over time.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def get_ml_client(subscription_id: str, resource_group: str, workspace_name: str) -> MLClient:
    credential = DefaultAzureCredential()
    return MLClient(credential, subscription_id, resource_group, workspace_name)


def upload_predictions(
    ml_client: MLClient,
    predictions_path: str,
    environment: str = "prod",
    asset_name: str = "titanic-predictions",
) -> Data:
    """Register the predictions folder as a new version of the predictions data asset."""
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    asset = Data(
        name=asset_name,
        description=f"Batch inference predictions — {environment} — {run_ts}",
        path=str(Path(predictions_path).resolve()),
        type=AssetTypes.URI_FOLDER,
        tags={
            "environment":    environment,
            "run_timestamp":  run_ts,
            "pipeline":       "cd-inference",
            "purpose":        "monitoring-input",
        },
    )

    registered = ml_client.data.create_or_update(asset)
    logger.info(
        "Predictions registered as data asset: %s @ version %s",
        registered.name,
        registered.version,
    )
    return registered


def write_prediction_manifest(predictions_path: str, registered_version: str) -> None:
    """Write a manifest JSON alongside predictions for traceability."""
    manifest = {
        "predictions_path": predictions_path,
        "registered_asset_version": registered_version,
        "upload_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = Path(predictions_path) / "upload_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest written to %s", manifest_path)


def main():
    parser = argparse.ArgumentParser(description="Upload batch predictions to AML Data Asset")
    parser.add_argument("--predictions-path", required=True)
    parser.add_argument("--resource-group",   required=True)
    parser.add_argument("--workspace-name",   required=True)
    parser.add_argument("--subscription-id",  default=None)
    parser.add_argument("--environment",      default="prod")
    parser.add_argument("--asset-name",       default="titanic-predictions")
    args = parser.parse_args()

    subscription_id = args.subscription_id or os.environ.get("AZURE_SUBSCRIPTION_ID")
    if not subscription_id:
        logger.error("AZURE_SUBSCRIPTION_ID not set")
        sys.exit(1)

    ml_client = get_ml_client(subscription_id, args.resource_group, args.workspace_name)
    registered = upload_predictions(
        ml_client,
        args.predictions_path,
        args.environment,
        args.asset_name,
    )
    write_prediction_manifest(args.predictions_path, registered.version)
    print(f"Uploaded predictions as: {registered.name} @ version {registered.version}")


if __name__ == "__main__":
    main()
