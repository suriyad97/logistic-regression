"""
Flask API for model serving and inference.
Provides HTTP endpoints for predictions.
"""

import os
import json
import logging
from flask import Flask, request, jsonify
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference import ModelInference

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global inference engine
inference_engine = None


def load_model():
    """Load model on startup."""
    global inference_engine
    
    model_path = os.getenv('MODEL_PATH', '/model/model.pkl')
    preprocessor_path = os.getenv('PREPROCESSOR_PATH', '/model/preprocessor.pkl')
    
    logger.info(f"Loading model from {model_path}")
    logger.info(f"Loading preprocessor from {preprocessor_path}")
    
    try:
        inference_engine = ModelInference(model_path, preprocessor_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'model_loaded': inference_engine is not None})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Make predictions on new data.
    Expects JSON with passenger features.
    """
    try:
        if inference_engine is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Make prediction
        result = inference_engine.predict_single(data)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 400


@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """
    Make batch predictions on multiple instances.
    Expects JSON with list of passenger features.
    """
    try:
        if inference_engine is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        
        if not data or not isinstance(data, list):
            return jsonify({'error': 'Expected list of instances'}), 400
        
        # Make predictions for each instance
        results = []
        for instance in data:
            try:
                result = inference_engine.predict_single(instance)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting instance: {e}")
                results.append({'error': str(e)})
        
        return jsonify({'predictions': results})
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': str(e)}), 400


@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information."""
    if inference_engine is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_type': 'logistic_regression',
        'dataset': 'titanic',
        'features': inference_engine.processor.feature_names,
        'feature_count': len(inference_engine.processor.feature_names) if inference_engine.processor.feature_names else 0
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run Flask app
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
