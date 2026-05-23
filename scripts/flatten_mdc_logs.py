import argparse
import json
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

def flatten_jsonl(input_dir: str, output_csv: str):
    """
    Reads all JSON Lines (.jsonl) files in the input directory,
    extracts the raw input features and the model's prediction,
    and writes a flattened tabular CSV that mirrors the batch predictions format.
    """
    input_path = Path(input_dir)
    jsonl_files = list(input_path.rglob("*.jsonl"))
    
    if not jsonl_files:
        logger.warning(f"No .jsonl files found in {input_dir}")
        pd.DataFrame().to_csv(output_csv, index=False)
        return

    logger.info(f"Found {len(jsonl_files)} JSONL files to process.")
    
    records = []
    for file_path in jsonl_files:
        with open(file_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    # Azure MDC schema wraps the payload in 'data'
                    # We assume the user payload contains features, and 'response' contains the prediction.
                    
                    # This is a generic flattening. A real implementation would parse the specific
                    # request/response schema configured in the scoring script.
                    row = {}
                    
                    # Extract inputs (assuming standard MDC input logging)
                    inputs = data.get("request", {}).get("body", {})
                    if isinstance(inputs, dict):
                        row.update(inputs)
                        
                    # Extract output (assuming standard MDC output logging)
                    outputs = data.get("response", {}).get("body", {})
                    if isinstance(outputs, dict):
                        row.update({"prediction": outputs.get("prediction", None)})
                        
                    # Also tag with deployment name if available in headers
                    row["DeploymentName"] = data.get("request", {}).get("headers", {}).get("azureml-model-deployment", "unknown")
                    
                    records.append(row)
                except Exception as e:
                    logger.error(f"Error parsing line in {file_path}: {e}")

    df = pd.DataFrame(records)
    logger.info(f"Flattened {len(df)} records.")
    
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved flattened data to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flatten Azure MDC JSONL logs into a tabular CSV.")
    parser.add_argument("--input", required=True, help="Directory containing raw JSONL files.")
    parser.add_argument("--output", required=True, help="Path to save the output CSV.")
    args = parser.parse_args()

    flatten_jsonl(args.input, args.output)
