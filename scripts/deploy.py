"""
Quick deployment script for existing Azure ML infrastructure
Runs complete pipeline: upload data -> submit training -> evaluate -> deploy
"""

import os
import sys
import logging
import argparse
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner(text):
    """Print formatted banner."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def check_prerequisites():
    """Check if all required tools are installed."""
    logger.info("Checking prerequisites...")
    
    required_tools = ['az', 'python', 'pip']
    missing = []
    
    for tool in required_tools:
        import shutil
        if shutil.which(tool) is None:
            missing.append(tool)
    
    if missing:
        logger.error(f"Missing required tools: {', '.join(missing)}")
        logger.error("Please install them before proceeding")
        return False
    
    logger.info("✓ All prerequisites installed")
    return True


def check_files():
    """Check if all required files exist."""
    logger.info("Checking required files...")
    
    required_files = [
        'titanic.csv',
        'requirements.txt',
        'src/train.py',
        'src/evaluate.py',
        'config/environment.yml',
        'scripts/submit_pipeline.py',
        'scripts/upload_data.py'
    ]
    
    missing = []
    for file in required_files:
        if not os.path.exists(file):
            missing.append(file)
    
    if missing:
        logger.error(f"Missing files: {', '.join(missing)}")
        return False
    
    logger.info("✓ All required files present")
    return True


def install_dependencies():
    """Install Python dependencies."""
    logger.info("Installing Python dependencies...")
    
    import subprocess
    try:
        subprocess.run(['pip', 'install', '-q', '-r', 'requirements.txt'], 
                      check=True)
        logger.info("✓ Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False


def authenticate_azure():
    """Authenticate with Azure."""
    logger.info("Authenticating with Azure...")
    
    import subprocess
    try:
        # Check if already authenticated
        result = subprocess.run(['az', 'account', 'show'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✓ Already authenticated with Azure")
            return True
        
        # Authenticate
        subprocess.run(['az', 'login'], check=True)
        logger.info("✓ Azure authentication successful")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Azure authentication failed: {e}")
        return False


def upload_data(subscription_id: str, resource_group: str, workspace_name: str):
    """Upload data to blob storage."""
    print_banner("Step 1: Uploading Data to Blob Storage")
    
    logger.info("Uploading titanic.csv to blob storage...")
    
    import subprocess
    try:
        result = subprocess.run([
            'python', 'scripts/upload_data.py',
            '--subscription-id', subscription_id,
            '--resource-group', resource_group,
            '--workspace-name', workspace_name,
            '--file', 'titanic.csv'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✓ Data upload successful")
            return True
        else:
            logger.error(f"Data upload failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error uploading data: {e}")
        return False


def submit_pipeline(subscription_id: str, resource_group: str, workspace_name: str):
    """Submit training pipeline to Azure ML."""
    print_banner("Step 2: Submitting Training Pipeline")
    
    logger.info("Submitting pipeline to Azure ML...")
    
    import subprocess
    try:
        result = subprocess.run([
            'python', 'scripts/submit_pipeline.py',
            '--subscription-id', subscription_id,
            '--resource-group', resource_group,
            '--workspace-name', workspace_name,
            '--wait'
        ], capture_output=True, text=True)
        
        print(result.stdout)
        
        if result.returncode == 0:
            logger.info("✓ Pipeline submitted successfully")
            return True
        else:
            logger.error(f"Pipeline submission failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error submitting pipeline: {e}")
        return False


def train_locally(use_local: bool = False):
    """Option to train locally first for testing."""
    print_banner("Training Locally (Optional)")
    
    if not use_local:
        logger.info("Skipping local training")
        return True
    
    logger.info("Training model locally...")
    
    import subprocess
    try:
        result = subprocess.run([
            'python', 'src/train.py',
            '--data', 'titanic.csv',
            '--output', './model',
            '--test-size', '0.2'
        ], capture_output=True, text=True)
        
        print(result.stdout)
        
        if result.returncode == 0:
            logger.info("✓ Local training successful")
            
            # Evaluate
            logger.info("Evaluating local model...")
            result = subprocess.run([
                'python', 'src/evaluate.py',
                '--model', './model/model.pkl',
                '--preprocessor', './model/preprocessor.pkl',
                '--data', 'titanic.csv',
                '--output', './evaluation'
            ], capture_output=True, text=True)
            
            print(result.stdout)
            logger.info("✓ Local evaluation complete")
            return True
        else:
            logger.error(f"Local training failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error in local training: {e}")
        return False


def deploy_api(docker: bool = False):
    """Deploy API server."""
    print_banner("Step 3: Deploying API")
    
    if docker:
        logger.info("Building Docker image...")
        import subprocess
        
        try:
            result = subprocess.run(['docker', 'build', '-t', 'titanic:latest', '.'],
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✓ Docker image built successfully")
                logger.info("\nTo run the API:")
                logger.info("  docker run -p 5000:5000 -v $(pwd)/model:/model titanic:latest")
                return True
            else:
                logger.error(f"Docker build failed: {result.stderr}")
                return False
                
        except FileNotFoundError:
            logger.warning("Docker not installed. Skipping Docker deployment")
            return False
    else:
        logger.info("To run the API locally:")
        logger.info("  python scripts/app.py")
        return True


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description='Deploy Titanic ML pipeline to Azure')
    parser.add_argument('--subscription-id', type=str, required=True,
                       help='Azure subscription ID')
    parser.add_argument('--resource-group', type=str, required=True,
                       help='Azure resource group')
    parser.add_argument('--workspace-name', type=str, required=True,
                       help='Azure ML workspace name')
    parser.add_argument('--train-local', action='store_true',
                       help='Train model locally first for testing')
    parser.add_argument('--docker', action='store_true',
                       help='Build Docker image for API')
    parser.add_argument('--skip-upload', action='store_true',
                       help='Skip data upload step')
    
    args = parser.parse_args()
    
    print_banner("Titanic ML Pipeline - Azure Deployment")
    
    try:
        # Check prerequisites
        if not check_prerequisites():
            return 1
        
        # Check files
        if not check_files():
            return 1
        
        # Install dependencies
        if not install_dependencies():
            return 1
        
        # Authenticate
        if not authenticate_azure():
            return 1
        
        # Train locally if requested
        if args.train_local:
            if not train_locally(use_local=True):
                return 1
        
        # Upload data
        if not args.skip_upload:
            if not upload_data(args.subscription_id, args.resource_group, 
                             args.workspace_name):
                return 1
        
        # Submit pipeline
        if not submit_pipeline(args.subscription_id, args.resource_group, 
                             args.workspace_name):
            return 1
        
        # Deploy API
        if not deploy_api(args.docker):
            return 1
        
        # Success
        print_banner("Deployment Complete! ✓")
        
        print("\nNext steps:")
        print("1. Monitor training in Azure ML Studio:")
        print(f"   https://ml.azure.com/workspaces/{args.workspace_name}")
        print("\n2. Run the API server:")
        print("   python scripts/app.py")
        print("\n3. Test predictions:")
        print("   curl http://localhost:5000/health")
        print("\n4. View evaluation results in your workspace")
        print("\n" + "="*70 + "\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
