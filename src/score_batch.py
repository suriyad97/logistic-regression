import os
import pandas as pd
import logging
from inference import ModelInference

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def init():
    """
    This function is called once when the batch endpoint container starts up.
    We load the model and preprocessor here.
    """
    global engine
    
    aml_model_dir = os.getenv("AZUREML_MODEL_DIR")
    if not aml_model_dir:
        raise ValueError("AZUREML_MODEL_DIR environment variable not set")
        
    logger.info(f"Looking for models in: {aml_model_dir}")
    
    # Depending on how the model is registered, check subfolders
    if os.path.exists(os.path.join(aml_model_dir, "model", "model.pkl")):
        model_path = os.path.join(aml_model_dir, "model", "model.pkl")
        preprocessor_path = os.path.join(aml_model_dir, "model", "preprocessor.pkl")
    else:
        model_path = os.path.join(aml_model_dir, "model.pkl")
        preprocessor_path = os.path.join(aml_model_dir, "preprocessor.pkl")
        
    logger.info(f"Loading model from {model_path}")
    engine = ModelInference(model_path, preprocessor_path)

def run(mini_batch):
    """
    This function is called for every mini-batch of files.
    """
    logger.info(f"Processing mini_batch of size {len(mini_batch)}")
    
    results = []
    
    for file_path in mini_batch:
        logger.info(f"Processing file: {file_path}")
        try:
            df = pd.read_csv(file_path)
            # Perform inference
            preds = engine.predict(df)
            
            # Format predictions line by line
            for i in range(len(preds["predictions"])):
                passenger_id = str(df["PassengerId"].iloc[i]) if "PassengerId" in df.columns else str(i)
                results.append(
                    f"{passenger_id},{preds['predictions'][i]},{preds['survival_probability'][i]:.4f},{preds['prediction_labels'][i]}"
                )
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            results.append(f"ERROR: {file_path} - {str(e)}")
            
    return results
