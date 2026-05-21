"""
Azure ML Workspace Connection and Configuration Script
Connects to existing Azure ML workspace and configures required settings.
"""

import os
import logging
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.storage.blob import BlobServiceClient
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_workspace_config():
    """Get Azure ML workspace configuration from environment or user input."""
    
    subscription_id = os.getenv('AZURE_SUBSCRIPTION_ID')
    resource_group = os.getenv('AZURE_RESOURCE_GROUP')
    workspace_name = os.getenv('AZURE_ML_WORKSPACE')
    
    if not all([subscription_id, resource_group, workspace_name]):
        print("\n=== Azure ML Workspace Configuration ===\n")
        subscription_id = subscription_id or input("Enter Azure Subscription ID: ").strip()
        resource_group = resource_group or input("Enter Resource Group name: ").strip()
        workspace_name = workspace_name or input("Enter Azure ML Workspace name: ").strip()
    
    return {
        'subscription_id': subscription_id,
        'resource_group': resource_group,
        'workspace_name': workspace_name
    }


def connect_to_workspace(subscription_id: str, resource_group: str, workspace_name: str) -> MLClient:
    """Connect to existing Azure ML workspace."""
    
    logger.info(f"Connecting to workspace: {workspace_name}")
    logger.info(f"Resource Group: {resource_group}")
    logger.info(f"Subscription: {subscription_id}")
    
    try:
        credential = DefaultAzureCredential()
        
        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name
        )
        
        # Verify connection
        workspace = ml_client.workspaces.get(workspace_name)
        logger.info(f"✓ Connected successfully to workspace: {workspace.display_name}")
        logger.info(f"  Storage Account: {workspace.storage_account}")
        logger.info(f"  Key Vault: {workspace.key_vault}")
        
        return ml_client
        
    except Exception as e:
        logger.error(f"Failed to connect to workspace: {e}")
        raise


def get_storage_connection_string(ml_client: MLClient) -> str:
    """Get storage account connection string from workspace."""
    
    logger.info("Retrieving storage account details...")
    
    try:
        workspace = ml_client.workspaces.get(ml_client._workspace_name)
        storage_account_id = workspace.storage_account
        
        # Extract storage account name from resource ID
        # Format: /subscriptions/{id}/resourceGroups/{rg}/providers/Microsoft.Storage/storageAccounts/{name}
        storage_account_name = storage_account_id.split('/')[-1]
        
        logger.info(f"Storage Account: {storage_account_name}")
        
        # Get storage account keys using Azure CLI
        import subprocess
        result = subprocess.run([
            'az', 'storage', 'account', 'keys', 'list',
            '--resource-group', ml_client._resource_group_name,
            '--account-name', storage_account_name,
            '--query', '[0].value',
            '--output', 'tsv'
        ], capture_output=True, text=True, check=True)
        
        storage_key = result.stdout.strip()
        
        connection_string = f"DefaultEndpointsProtocol=https;AccountName={storage_account_name};AccountKey={storage_key};EndpointSuffix=core.windows.net"
        
        logger.info("✓ Storage connection string retrieved")
        return connection_string
        
    except Exception as e:
        logger.error(f"Failed to get storage connection string: {e}")
        raise


def upload_data_to_blob(connection_string: str, local_file: str, 
                       container_name: str = 'amldata', blob_name: str = None) -> str:
    """Upload local file to Azure Blob Storage."""
    
    if blob_name is None:
        blob_name = os.path.basename(local_file)
    
    logger.info(f"Uploading {local_file} to blob storage...")
    logger.info(f"  Container: {container_name}")
    logger.info(f"  Blob name: {blob_name}")
    
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # Get or create container
        container_client = blob_service_client.get_container_client(container_name)
        
        # Upload file
        with open(local_file, 'rb') as data:
            container_client.upload_blob(blob_name, data, overwrite=True)
        
        logger.info(f"✓ Successfully uploaded {blob_name}")
        
        # Return blob URI
        blob_uri = f"azureml://datastores/workspaceblobstore/paths/{blob_name}"
        logger.info(f"  Blob URI: {blob_uri}")
        
        return blob_uri
        
    except Exception as e:
        logger.error(f"Failed to upload file: {e}")
        raise


def verify_compute_cluster(ml_client: MLClient, compute_name: str = 'cpu-cluster') -> bool:
    """Verify that compute cluster exists."""
    
    logger.info(f"Checking for compute cluster: {compute_name}")
    
    try:
        compute = ml_client.compute.get(compute_name)
        logger.info(f"✓ Compute cluster found: {compute.name}")
        logger.info(f"  Type: {compute.type}")
        logger.info(f"  Status: {compute.status}")
        return True
        
    except Exception as e:
        logger.warning(f"Compute cluster not found: {e}")
        return False


def create_environment(ml_client: MLClient, env_name: str = 'titanic-env') -> str:
    """Create or get Azure ML environment."""
    
    logger.info(f"Setting up environment: {env_name}")
    
    try:
        # Check if environment already exists
        try:
            env = ml_client.environments.get(name=env_name, label='latest')
            logger.info(f"✓ Environment already exists: {env.name}")
            return env.id
        except:
            pass
        
        # Create new environment from conda file
        from azure.ai.ml.entities import Environment
        
        env_path = Path('config/environment.yml')
        if not env_path.exists():
            logger.error(f"Environment file not found: {env_path}")
            raise FileNotFoundError(f"No environment file at {env_path}")
        
        env = Environment(
            name=env_name,
            description="Environment for Titanic logistic regression",
            conda_file=str(env_path),
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
        )
        
        created_env = ml_client.environments.create_or_update(env)
        logger.info(f"✓ Environment created: {created_env.name}")
        
        return created_env.id
        
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        raise


def main():
    """Main setup function."""
    
    print("\n" + "="*60)
    print("Azure ML Workspace Setup & Configuration")
    print("="*60 + "\n")
    
    try:
        # Step 1: Get workspace configuration
        config = get_workspace_config()
        
        # Step 2: Connect to workspace
        logger.info("\n[1/5] Connecting to Azure ML workspace...")
        ml_client = connect_to_workspace(
            config['subscription_id'],
            config['resource_group'],
            config['workspace_name']
        )
        
        # Step 3: Verify compute cluster
        logger.info("\n[2/5] Verifying compute cluster...")
        has_compute = verify_compute_cluster(ml_client)
        
        # Step 4: Create/verify environment
        logger.info("\n[3/5] Setting up environment...")
        env_id = create_environment(ml_client)
        
        # Step 5: Upload data to blob storage
        logger.info("\n[4/5] Uploading data to blob storage...")
        local_data_path = 'titanic.csv'
        if os.path.exists(local_data_path):
            connection_string = get_storage_connection_string(ml_client)
            blob_uri = upload_data_to_blob(connection_string, local_data_path)
        else:
            logger.warning(f"Data file not found: {local_data_path}")
        
        # Summary
        logger.info("\n[5/5] Configuration complete!")
        
        print("\n" + "="*60)
        print("✓ Setup Summary")
        print("="*60)
        print(f"Workspace: {config['workspace_name']}")
        print(f"Resource Group: {config['resource_group']}")
        print(f"Subscription: {config['subscription_id']}")
        print(f"Compute Cluster Ready: {'Yes' if has_compute else 'No'}")
        print(f"Environment: {env_name if 'env_name' in locals() else 'N/A'}")
        print("="*60 + "\n")
        
        print("Next steps:")
        print("1. Review the configuration above")
        print("2. Run: python scripts/submit_pipeline.py")
        print("3. Monitor training in Azure ML Studio")
        print("\n")
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        raise


if __name__ == '__main__':
    main()
