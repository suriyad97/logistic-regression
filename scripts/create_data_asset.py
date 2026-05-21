"""
create_data_asset.py
────────────────────
Registers (or creates a new version of) the Titanic training dataset
as an Azure ML Data Asset.

Called by: CT pipeline (ct-train-register-model.yml)

Why data assets?
  - Every new version is immutable & traceable in AML Studio
  - Training jobs reference them by name@version — reproducible runs
  - AML auto-increments the version on each upload
"""

import argparse
import logging
import sys
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


def create_or_update_data_asset(
    ml_client: MLClient,
    data_path: str,
    asset_name: str = "titanic-dataset",
    description: str = "Titanic passenger survival dataset — training split",
) -> Data:
    """
    Register the local CSV as a versioned AML Data Asset.
    AML auto-increments the version number on each call.
    """
    data_asset = Data(
        name=asset_name,
        description=description,
        path=str(Path(data_path).resolve()),
        type=AssetTypes.URI_FILE,
        tags={
            "source": "titanic",
            "format": "csv",
            "task": "binary_classification",
        },
    )

    registered = ml_client.data.create_or_update(data_asset)
    logger.info(
        "Data asset registered: %s  version=%s",
        registered.name,
        registered.version,
    )
    return registered


def main():
    parser = argparse.ArgumentParser(description="Register training data as AML Data Asset")
    parser.add_argument("--subscription-id", default=None)
    parser.add_argument("--resource-group", required=True)
    parser.add_argument("--workspace-name", required=True)
    parser.add_argument("--data-path", default="data/titanic.csv")
    parser.add_argument("--asset-name", default="titanic-dataset")
    args = parser.parse_args()

    # Fall back to env var if not passed as argument
    import os
    subscription_id = args.subscription_id or os.environ.get("AZURE_SUBSCRIPTION_ID")
    if not subscription_id:
        logger.error("subscription-id not provided and AZURE_SUBSCRIPTION_ID env var not set")
        sys.exit(1)

    ml_client = get_ml_client(subscription_id, args.resource_group, args.workspace_name)
    asset = create_or_update_data_asset(ml_client, args.data_path, args.asset_name)
    print(f"Registered data asset: {asset.name} @ version {asset.version}")


if __name__ == "__main__":
    main()
