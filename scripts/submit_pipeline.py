"""
Azure ML pipeline submission and management script.
Submits training pipeline to Azure ML and manages run lifecycle.
"""

import os
import argparse
import logging
from pathlib import Path
from azure.ai.ml import MLClient
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import command, load_component
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_ml_client(subscription_id: str, resource_group: str, workspace_name: str) -> MLClient:
    """Get Azure ML client with proper authentication."""
    credential = DefaultAzureCredential()
    
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name
    )
    
    return ml_client


def create_environment(ml_client: MLClient, env_name: str, conda_path: str) -> str:
    """Create Azure ML environment from conda file."""
    logger.info(f"Creating environment: {env_name}")
    
    try:
        # Check if environment already exists
        try:
            env = ml_client.environments.get(name=env_name, label='latest')
            logger.info(f"Environment {env_name} already exists")
            return env.name
        except:
            pass
        
        # Create new environment
        with open(conda_path, 'r') as f:
            conda_config = yaml.safe_load(f)
        
        env = Environment(
            name=env_name,
            description="Environment for Titanic logistic regression",
            conda_file=conda_path,
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
        )
        
        env = ml_client.environments.create_or_update(env)
        logger.info(f"Environment {env_name} created successfully")
        
        return env.name
        
    except Exception as e:
        logger.error(f"Error creating environment: {e}")
        raise


def submit_pipeline(ml_client: MLClient, pipeline_path: str, 
                   experiment_name: str) -> str:
    """Submit pipeline job to Azure ML."""
    logger.info(f"Submitting pipeline from {pipeline_path}")
    
    try:
        # Load pipeline
        with open(pipeline_path, 'r') as f:
            pipeline_job = yaml.safe_load(f)
        
        # Submit job
        submitted_job = ml_client.jobs.create_or_update(pipeline_job)
        logger.info(f"Pipeline submitted successfully")
        logger.info(f"Job ID: {submitted_job.name}")
        logger.info(f"Job status: {submitted_job.status}")
        
        return submitted_job.name
        
    except Exception as e:
        logger.error(f"Error submitting pipeline: {e}")
        raise


def wait_for_completion(ml_client: MLClient, job_name: str, timeout: int = 3600):
    """Wait for job to complete."""
    logger.info(f"Waiting for job {job_name} to complete...")
    
    import time
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        job = ml_client.jobs.get(job_name)
        
        if job.status in ['Completed', 'Failed', 'Canceled']:
            logger.info(f"Job {job_name} completed with status: {job.status}")
            return job.status
        
        logger.info(f"Current status: {job.status}")
        time.sleep(30)
    
    logger.warning(f"Job did not complete within {timeout} seconds")
    return None


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Submit Azure ML pipeline')
    parser.add_argument('--subscription-id', type=str, required=True,
                       help='Azure subscription ID')
    parser.add_argument('--resource-group', type=str, required=True,
                       help='Azure resource group')
    parser.add_argument('--workspace-name', type=str, required=True,
                       help='Azure ML workspace name')
    parser.add_argument('--pipeline-path', type=str, 
                       default='pipelines/pipeline.yml',
                       help='Path to pipeline YAML')
    parser.add_argument('--conda-path', type=str,
                       default='config/environment.yml',
                       help='Path to conda environment file')
    parser.add_argument('--experiment-name', type=str,
                       default='titanic-logistic-regression-exp',
                       help='Experiment name')
    parser.add_argument('--wait', action='store_true',
                       help='Wait for pipeline to complete')
    
    args = parser.parse_args()
    
    try:
        # Get ML client
        ml_client = get_ml_client(
            args.subscription_id,
            args.resource_group,
            args.workspace_name
        )
        logger.info("Connected to Azure ML workspace")
        
        # Create environment
        env_name = create_environment(ml_client, 'titanic-env', args.conda_path)
        
        # Submit pipeline
        job_id = submit_pipeline(ml_client, args.pipeline_path, args.experiment_name)
        
        # Optionally wait for completion
        if args.wait:
            status = wait_for_completion(ml_client, job_id)
            logger.info(f"Final status: {status}")
        
        logger.info("Pipeline submission completed")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == '__main__':
    main()
