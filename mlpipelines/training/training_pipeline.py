"""
Training ML Pipeline for Titanic Logistic Regression Model
Includes feature engineering, model training, and evaluation
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from azure.ai.ml import MLClient, Input, Output, dsl
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

# Load environment variables
load_dotenv('config/.env')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dsl.component(
    base_image="python:3.12",
    name="data_preparation",
    display_name="Data Preparation",
    code="./src",
)
def data_preparation(
    input_data: Input(type="uri_file"),
    output_data: Output(type="uri_file"),
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Prepare data for training"""
    import pandas as pd
    from data_processing import DataProcessor
    import joblib
    
    processor = DataProcessor(random_state=random_state)
    X_train, X_test, y_train, y_test = processor.process_data(
        input_data, is_training=True, test_size=test_size
    )
    
    # Save processed data
    processed_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    joblib.dump(processed_data, output_data)
    
    print(f"Data prepared: Train={X_train.shape}, Test={X_test.shape}")


@dsl.component(
    base_image="python:3.12",
    name="model_training",
    display_name="Model Training",
    code="./src",
)
def model_training(
    processed_data: Input(type="uri_file"),
    model_output: Output(type="uri_folder"),
    max_iter: int = 1000,
    random_state: int = 42,
):
    """Train logistic regression model"""
    import joblib
    from sklearn.linear_model import LogisticRegression
    import json
    
    # Load processed data
    data = joblib.load(processed_data)
    X_train = data['X_train']
    y_train = data['y_train']
    
    # Train model
    model = LogisticRegression(max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Save model
    model_path = f"{model_output}/model.pkl"
    joblib.dump(model, model_path)
    
    # Log feature importance
    feature_importance = {
        f"Feature_{i}": float(coef) 
        for i, coef in enumerate(model.coef_[0])
    }
    
    with open(f"{model_output}/feature_importance.json", "w") as f:
        json.dump(feature_importance, f)
    
    print(f"Model trained and saved to {model_path}")


@dsl.component(
    base_image="python:3.12",
    name="model_evaluation",
    display_name="Model Evaluation",
    code="./src",
)
def model_evaluation(
    processed_data: Input(type="uri_file"),
    model_path: Input(type="uri_folder"),
    metrics_output: Output(type="uri_file"),
):
    """Evaluate model performance"""
    import joblib
    import json
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, roc_auc_score, confusion_matrix
    )
    
    # Load data and model
    data = joblib.load(processed_data)
    model = joblib.load(f"{model_path}/model.pkl")
    
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    }
    
    # Save metrics
    with open(metrics_output, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics: {metrics}")


@dsl.pipeline(
    name="titanic-training-pipeline",
    description="End-to-end training pipeline for Titanic model",
    compute="logistic12",
)
def training_pipeline(
    input_data: Input(type="uri_file") = "data/titanic.csv",
    test_size: float = 0.2,
    random_state: int = 42,
    max_iter: int = 1000,
):
    """Define the training pipeline"""
    
    # Data preparation
    prep = data_preparation(
        input_data=input_data,
        test_size=test_size,
        random_state=random_state,
    )
    
    # Model training
    training = model_training(
        processed_data=prep.outputs.output_data,
        max_iter=max_iter,
        random_state=random_state,
    )
    
    # Model evaluation
    evaluation = model_evaluation(
        processed_data=prep.outputs.output_data,
        model_path=training.outputs.model_output,
    )
    
    return {
        "model": training.outputs.model_output,
        "metrics": evaluation.outputs.metrics_output,
    }


def submit_pipeline():
    """Submit the pipeline to Azure ML"""
    
    # Initialize MLClient
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=os.getenv('AZURE_SUBSCRIPTION_ID'),
        resource_group_name=os.getenv('AZURE_RESOURCE_GROUP') + '-dev',
        workspace_name=os.getenv('AZURE_ML_WORKSPACE') + '-dev',
    )
    
    # Create pipeline job
    pipeline_job = training_pipeline()
    
    # Submit pipeline
    submitted_job = ml_client.jobs.create_or_update(pipeline_job)
    logger.info(f"Pipeline submitted with ID: {submitted_job.name}")
    
    # Stream logs
    ml_client.jobs.stream(submitted_job.name)
    
    return submitted_job


if __name__ == "__main__":
    submit_pipeline()
