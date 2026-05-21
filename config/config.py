"""
Configuration module for Azure ML pipeline.
Contains all configuration parameters for training and deployment.
"""

from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class DataConfig:
    """Data configuration."""
    data_path: str = 'titanic.csv'
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class ModelConfig:
    """Model configuration."""
    model_type: str = 'logistic_regression'
    max_iter: int = 1000
    solver: str = 'lbfgs'
    class_weight: str = 'balanced'
    random_state: int = 42


@dataclass
class TrainingConfig:
    """Training configuration."""
    data: DataConfig = None
    model: ModelConfig = None
    output_dir: str = './model'
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()


@dataclass
class AzureMLConfig:
    """Azure ML configuration."""
    subscription_id: str = os.getenv('AZURE_SUBSCRIPTION_ID', '')
    resource_group: str = os.getenv('AZURE_RESOURCE_GROUP', '')
    workspace_name: str = os.getenv('AZURE_ML_WORKSPACE', '')
    compute_target: str = 'cpu-cluster'
    environment_name: str = 'titanic-logistic-regression'
    experiment_name: str = 'titanic-logistic-regression-exp'
    
    
@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    model_name: str = 'titanic-logistic-regression'
    model_version: str = '1.0.0'
    inference_config_name: str = 'inference-config'
    deployment_config_name: str = 'deployment-config'
    deployment_name: str = 'titanic-logistic-reg-endpoint'
    sku: str = 'Standard_F2s_v2'
    instance_count: int = 1


def get_config() -> TrainingConfig:
    """Get default training configuration."""
    return TrainingConfig()
