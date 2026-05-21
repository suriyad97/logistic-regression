"""
Batch Inference Pipeline with A/B Testing
Generates predictions and logs for monitoring
"""

import os
import json
import logging
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv('config/.env')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_preprocessor(model_path: str):
    """Load trained model and preprocessor"""
    model = joblib.load(f"{model_path}/model.pkl")
    preprocessor = joblib.load(f"{model_path}/preprocessor.pkl")
    return model, preprocessor


def run_inference(
    model,
    preprocessor,
    data: pd.DataFrame,
    ab_test_enabled: bool = False,
    ab_test_split: float = 0.1,
):
    """Run inference on data"""
    
    # Prepare data
    X = preprocessor.transform(data)
    
    # Generate predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Create results dataframe
    results = pd.DataFrame({
        'prediction': predictions,
        'probability_negative': probabilities[:, 0],
        'probability_positive': probabilities[:, 1],
        'confidence': np.max(probabilities, axis=1),
    })
    
    # A/B testing: mark some records for challenger model
    if ab_test_enabled:
        n_records = len(results)
        ab_indices = np.random.choice(
            n_records, 
            size=int(n_records * ab_test_split), 
            replace=False
        )
        results['model_variant'] = 'champion'
        results.loc[ab_indices, 'model_variant'] = 'challenger'
    else:
        results['model_variant'] = 'champion'
    
    # Add metadata
    results['inference_timestamp'] = pd.Timestamp.now()
    results['model_version'] = os.getenv('MODEL_VERSION', 'unknown')
    
    return results


def main(
    model_path: str,
    input_data: str,
    output_path: str,
    ab_test_enabled: bool = False,
    ab_test_split: float = 0.1,
):
    """Main inference function"""
    
    logger.info(f"Loading model from {model_path}")
    model, preprocessor = load_model_and_preprocessor(model_path)
    
    logger.info(f"Loading data from {input_data}")
    data = pd.read_csv(input_data)
    
    logger.info("Running inference")
    results = run_inference(
        model,
        preprocessor,
        data,
        ab_test_enabled=ab_test_enabled,
        ab_test_split=ab_test_split,
    )
    
    # Save results
    Path(output_path).mkdir(parents=True, exist_ok=True)
    results.to_csv(f"{output_path}/predictions.csv", index=False)
    
    # Log summary statistics
    summary = {
        'total_predictions': len(results),
        'positive_predictions': int((results['prediction'] == 1).sum()),
        'negative_predictions': int((results['prediction'] == 0).sum()),
        'average_confidence': float(results['confidence'].mean()),
        'min_confidence': float(results['confidence'].min()),
        'max_confidence': float(results['confidence'].max()),
    }
    
    with open(f"{output_path}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Inference completed. Results saved to {output_path}")
    logger.info(f"Summary: {summary}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True, help='Path to model directory')
    parser.add_argument('--input-data', required=True, help='Path to input data CSV')
    parser.add_argument('--output-path', required=True, help='Path to save predictions')
    parser.add_argument('--ab-test-enabled', type=bool, default=False)
    parser.add_argument('--ab-test-split', type=float, default=0.1)
    
    args = parser.parse_args()
    
    main(
        model_path=args.model_path,
        input_data=args.input_data,
        output_path=args.output_path,
        ab_test_enabled=args.ab_test_enabled,
        ab_test_split=args.ab_test_split,
    )
