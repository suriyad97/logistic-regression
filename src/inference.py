"""
Inference module for making predictions with trained model.
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Union, Dict, List

from data_processing import DataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelInference:
    """Handles model inference and predictions."""
    
    def __init__(self, model_path: str, preprocessor_path: str):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to trained model
            preprocessor_path: Path to preprocessor
        """
        logger.info("Loading model and preprocessor for inference...")
        self.model = joblib.load(model_path)
        self.processor = DataProcessor()
        self.processor.load_preprocessor(preprocessor_path)
        logger.info("Model and preprocessor loaded successfully")
    
    def predict(self, data: Union[str, pd.DataFrame]) -> Dict:
        """
        Make predictions on new data.
        
        Args:
            data: Either path to CSV file or DataFrame
            
        Returns:
            Dictionary with predictions and probabilities
        """
        logger.info("Starting inference")
        
        try:
            # Load or process data
            if isinstance(data, str):
                df = self.processor.load_data(data, is_training=False)
            else:
                df = data.copy()
            
            logger.info(f"Processing {df.shape[0]} samples")
            
            # Handle missing values and feature engineering
            df = self.processor.handle_missing_values(df)
            df = self.processor.feature_engineering(df, is_training=False)
            df = self.processor.encode_categorical_features(df, is_training=False)
            
            # Select features
            X = df[self.processor.feature_names]
            
            # Scale features
            X_scaled = self.processor.scale_features(X, is_training=False)
            
            # Make predictions
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)
            
            logger.info(f"Predictions completed for {len(predictions)} samples")
            
            # Format results
            results = {
                'predictions': predictions.tolist(),
                'survival_probability': probabilities[:, 1].tolist(),
                'non_survival_probability': probabilities[:, 0].tolist(),
                'prediction_labels': ['Survived' if p == 1 else 'Did not survive' 
                                     for p in predictions]
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}", exc_info=True)
            raise
    
    def predict_single(self, features: Dict) -> Dict:
        """
        Make prediction for a single instance.
        
        Args:
            features: Dictionary with feature values
            
        Returns:
            Prediction for single instance
        """
        logger.info("Making prediction for single instance")
        
        try:
            # Create DataFrame from features
            df = pd.DataFrame([features])
            
            # Process data
            df = self.processor.handle_missing_values(df)
            df = self.processor.feature_engineering(df, is_training=False)
            df = self.processor.encode_categorical_features(df, is_training=False)
            
            # Select features
            X = df[self.processor.feature_names]
            
            # Scale features
            X_scaled = self.processor.scale_features(X, is_training=False)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            probability = self.model.predict_proba(X_scaled)[0]
            
            result = {
                'prediction': int(prediction),
                'prediction_label': 'Survived' if prediction == 1 else 'Did not survive',
                'survival_probability': float(probability[1]),
                'non_survival_probability': float(probability[0])
            }
            
            logger.info(f"Prediction: {result['prediction_label']} "
                       f"({result['survival_probability']:.2%} confidence)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during single inference: {str(e)}", exc_info=True)
            raise


def batch_inference(model_path: str, preprocessor_path: str, input_data_path: str,
                   output_path: str) -> None:
    """
    Run batch inference and save results.
    
    Args:
        model_path: Path to trained model
        preprocessor_path: Path to preprocessor
        input_data_path: Path to input data CSV
        output_path: Path to save predictions CSV
    """
    logger.info("Starting batch inference")
    
    try:
        # Initialize inference engine
        inference = ModelInference(model_path, preprocessor_path)
        
        # Make predictions
        results = inference.predict(input_data_path)
        
        # Load original data and add predictions
        df = pd.read_csv(input_data_path)
        df['prediction'] = results['predictions']
        df['survival_probability'] = results['survival_probability']
        df['prediction_label'] = results['prediction_labels']
        
        # Save results
        df.to_csv(output_path, index=False)
        logger.info(f"Batch inference completed. Results saved to {output_path}")
        
        # Print summary statistics
        print(f"\nBatch Inference Summary:")
        print(f"Total predictions: {len(results['predictions'])}")
        print(f"Predicted survivors: {sum(results['predictions'])}")
        print(f"Predicted non-survivors: {len(results['predictions']) - sum(results['predictions'])}")
        print(f"Mean survival probability: {np.mean(results['survival_probability']):.2%}")

        
    except Exception as e:
        logger.error(f"Error during batch inference: {str(e)}", exc_info=True)
        raise


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch inference using a trained model and preprocessor.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (e.g., model.pkl)")
    parser.add_argument("--preprocessor", type=str, required=True, help="Path to DataProcessor (e.g., preprocessor.joblib)")
    parser.add_argument("--data", type=str, required=True, help="Path to raw inference data CSV")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Path to save predictions")
    args = parser.parse_args()

    batch_inference(
        os.path.abspath(args.model),
        os.path.abspath(args.preprocessor),
        os.path.abspath(args.data),
        os.path.abspath(args.output)
    )

if __name__ == '__main__':
    main()
