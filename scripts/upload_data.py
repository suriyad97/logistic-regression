"""
Upload data to Azure ML workspace blob storage
"""

import os
import argparse
import logging
from pathlib import Path
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.storage.blob import BlobServiceClient
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_storage_account_name(ml_client: MLClient) -> str:
    """Extract storage account name from workspace."""
    try:
        workspace = ml_client.workspaces.get(ml_client._workspace_name)
        storage_account_id = workspace.storage_account
        storage_account_name = storage_account_id.split('/')[-1]
        return storage_account_name
    except Exception as e:
        logger.error(f"Failed to get storage account: {e}")
        raise


def get_storage_account_key(subscription_id: str, resource_group: str, 
                            storage_account_name: str) -> str:
    """Get storage account key using Azure CLI."""
    try:
        result = subprocess.run([
            'az', 'storage', 'account', 'keys', 'list',
            '--subscription', subscription_id,
            '--resource-group', resource_group,
            '--account-name', storage_account_name,
            '--query', '[0].value',
            '--output', 'tsv'
        ], capture_output=True, text=True, check=True)
        
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to get storage key: {e.stderr}")
        raise
    except FileNotFoundError:
        logger.error("Azure CLI not found. Please install it first.")
        raise


def upload_file(subscription_id: str, resource_group: str, storage_account_name: str,
               local_file: str, container_name: str = 'amldata', 
               blob_name: str = None) -> str:
    """Upload file to Azure Blob Storage."""
    
    if not os.path.exists(local_file):
        raise FileNotFoundError(f"Local file not found: {local_file}")
    
    if blob_name is None:
        blob_name = os.path.basename(local_file)
    
    logger.info(f"Uploading {local_file}...")
    logger.info(f"  Storage Account: {storage_account_name}")
    logger.info(f"  Container: {container_name}")
    logger.info(f"  Blob: {blob_name}")
    
    try:
        # Get storage account key
        storage_key = get_storage_account_key(subscription_id, resource_group, 
                                            storage_account_name)
        
        # Create connection string
        connection_string = f"DefaultEndpointsProtocol=https;AccountName={storage_account_name};AccountKey={storage_key};EndpointSuffix=core.windows.net"
        
        # Upload file
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        
        with open(local_file, 'rb') as data:
            container_client.upload_blob(blob_name, data, overwrite=True)
        
        logger.info(f"✓ Upload successful")
        
        # Return blob path for Azure ML
        return f"azureml://datastores/workspaceblobstore/paths/{blob_name}"
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='Upload data to Azure ML blob storage')
    parser.add_argument('--subscription-id', type=str, required=True,
                       help='Azure subscription ID')
    parser.add_argument('--resource-group', type=str, required=True,
                       help='Azure resource group')
    parser.add_argument('--workspace-name', type=str, required=True,
                       help='Azure ML workspace name')
    parser.add_argument('--file', type=str, default='titanic.csv',
                       help='Local file to upload')
    parser.add_argument('--container', type=str, default='amldata',
                       help='Blob container name')
    parser.add_argument('--blob-name', type=str, default=None,
                       help='Name for blob (default: original filename)')
    
    args = parser.parse_args()
    
    try:
        # Connect to workspace to get storage account name
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id=args.subscription_id,
            resource_group_name=args.resource_group,
            workspace_name=args.workspace_name
        )
        
        storage_account_name = get_storage_account_name(ml_client)
        
        # Upload file
        blob_path = upload_file(
            args.subscription_id,
            args.resource_group,
            storage_account_name,
            args.file,
            args.container,
            args.blob_name
        )
        
        logger.info(f"Blob path: {blob_path}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == '__main__':
    main()
