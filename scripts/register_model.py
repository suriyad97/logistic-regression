"""
Model registration script.
Registers trained model to Azure ML model registry.
"""

import os
import argparse
import logging
import json
from pathlib import Path
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def register_model(subscription_id: str, resource_group: str, workspace_name: str,
                  model_path: str, preprocessor_path: str, model_name: str,
                  model_version: str) -> str:
    """Register model to Azure ML model registry."""
    
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name
    )
    
    logger.info(f"Registering model: {model_name} v{model_version}")
    
    try:
        # Create model
        model = Model(
            path=model_path,
            name=model_name,
            version=model_version,
            description="Logistic regression model for Titanic survival prediction",
            type="mlflow_model",
            properties={
                "task": "classification",
                "dataset": "titanic",
                "algorithm": "logistic_regression",
                "framework": "scikit-learn"
            }
        )
        
        # Register model
        registered_model = ml_client.models.create_or_update(model)
        logger.info(f"Model registered successfully")
        logger.info(f"Model ID: {registered_model.id}")
        logger.info(f"Model version: {registered_model.version}")
        
        return registered_model.id
        
    except Exception as e:
        logger.error(f"Error registering model: {e}")
        raise


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Register model to Azure ML')
    parser.add_argument('--subscription-id', type=str, required=True,
                       help='Azure subscription ID')
    parser.add_argument('--resource-group', type=str, required=True,
                       help='Azure resource group')
    parser.add_argument('--workspace-name', type=str, required=True,
                       help='Azure ML workspace name')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to model file')
    parser.add_argument('--preprocessor-path', type=str, required=True,
                       help='Path to preprocessor file')
    parser.add_argument('--model-name', type=str, required=True,
                       help='Model name')
    parser.add_argument('--model-version', type=str, required=True,
                       help='Model version')
    
    args = parser.parse_args()
    
    model_id = register_model(
        args.subscription_id,
        args.resource_group,
        args.workspace_name,
        args.model_path,
        args.preprocessor_path,
        args.model_name,
        args.model_version
    )
    
    logger.info(f"Model registration completed: {model_id}")


if __name__ == '__main__':
    main()
